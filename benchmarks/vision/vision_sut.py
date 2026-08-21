"""Vision benchmark: an image classifier as the system under test.

Uses the sklearn ``digits`` dataset (1797 8x8 grayscale images, 10 classes) and a
small MLP classifier -- dependency-free (no torch/download), fully reproducible.
The Reverse N-Wise pipeline builds a covering array over the classifier's
DECISION-BOUNDARY behaviour and inverts to synthesise images that realise each
target behaviour.

Because raw pixel space is high-dimensional (64), the inverse-map search operates
in a compact PCA latent space (default 8 components) and decodes back to pixels;
this is the standard way to make gradient-free inversion tractable for vision.

Output factors (q components of f(x), all ternary):
  0. class_group     -- predicted digit bucketed into 3 groups {0-2, 3-6, 7-9}
                        (coarse decision region)
  1. confidence       -- max-softmax probability band (low / med / high)
  2. margin_region    -- top1-top2 probability gap (near_boundary / mid / far)
  3. top2_relation    -- whether the runner-up class is in the same class_group
                        as the winner (intra / adjacent / cross) -- confusability
  4. bright_robust    -- does the decision survive a global brightness shift?
                        (robust / shifts_conf / flips) -- perturbation behaviour
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from rnwise.model import StochasticOutput, SystemUnderTest
from rnwise.optimizers.base import SearchSpace
from rnwise.partition import CategoricalPassthrough, PartitionScheme, ThresholdBands

N_COMPONENTS = 8  # PCA latent dimensionality for the inverse-map search space


def _class_group(label: int) -> str:
    if label <= 2:
        return "low"
    if label <= 6:
        return "mid"
    return "high"


@dataclass
class VisionData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    pca: Any
    latent_bounds: tuple[np.ndarray, np.ndarray]
    pixel_lo: float
    pixel_hi: float


@lru_cache(maxsize=1)
def load_vision(seed: int = 0) -> VisionData:
    """Load digits, fit a PCA latent space, and split train/test (cached)."""
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    ssl._create_default_https_context = ssl._create_unverified_context
    d = load_digits()
    X = d.data.astype(float)
    y = d.target.astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    pca = PCA(n_components=N_COMPONENTS, random_state=seed).fit(X_tr)
    z_tr = pca.transform(X_tr)
    lo = np.quantile(z_tr, 0.02, axis=0)
    hi = np.quantile(z_tr, 0.98, axis=0)
    return VisionData(
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        pca=pca,
        latent_bounds=(lo, hi),
        pixel_lo=float(X.min()),
        pixel_hi=float(X.max()),
    )


def train_mlp(data: VisionData, seed: int = 0):
    """Train the MLP image classifier used as the SUT."""
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=400,
        random_state=seed,
    )
    clf.fit(data.X_train, data.y_train)
    return clf


@dataclass
class VisionSUT:
    """Wrap the trained image classifier as a structured-output SUT.

    Inputs are PCA latent vectors ``z``; :meth:`decode` maps them to clipped
    pixel images, which the classifier scores. ``predict`` returns the 5-factor
    behavioural raw output.
    """

    model: Any
    data: VisionData
    _q: int = field(default=5, init=False)

    @property
    def q(self) -> int:
        return self._q

    def decode(self, z: np.ndarray) -> np.ndarray:
        """PCA latent -> clipped pixel image (the model-native input)."""
        img = self.data.pca.inverse_transform(np.asarray(z, dtype=float).reshape(1, -1))[0]
        return np.clip(img, self.data.pixel_lo, self.data.pixel_hi)

    def _proba(self, img: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(img.reshape(1, -1))[0]

    def predict(self, z: Any) -> StochasticOutput:
        img = self.decode(np.asarray(z, dtype=float))
        p = self._proba(img)
        order = np.argsort(p)[::-1]
        top1, top2 = int(order[0]), int(order[1])
        conf = float(p[top1])
        margin = float(p[top1] - p[top2])

        class_group = _class_group(top1)

        # top2 relation: same coarse group / adjacent / cross
        g1, g2 = _class_group(top1), _class_group(top2)
        if g1 == g2:
            top2_relation = "intra"
        elif {g1, g2} in ({"low", "mid"}, {"mid", "high"}):
            top2_relation = "adjacent"
        else:
            top2_relation = "cross"

        # brightness-robustness: shift pixels by +25% of range, re-classify
        shift = 0.25 * (self.data.pixel_hi - self.data.pixel_lo)
        bright = np.clip(img + shift, self.data.pixel_lo, self.data.pixel_hi)
        p_b = self._proba(bright)
        new_top = int(np.argmax(p_b))
        if new_top != top1:
            bright_robust = "flips"
        elif abs(float(p_b[new_top]) - conf) > 0.15:
            bright_robust = "shifts_conf"
        else:
            bright_robust = "robust"

        raw = (class_group, conf, margin, top2_relation, bright_robust)
        return StochasticOutput.deterministic(raw)


def vision_partition_scheme() -> PartitionScheme:
    """5-factor behavioural output partition for the vision SUT."""
    return PartitionScheme(
        (
            CategoricalPassthrough(alphabet=("low", "mid", "high")),
            ThresholdBands(edges=(0.5, 0.8), labels=("low", "med", "high")),
            ThresholdBands(edges=(0.1, 0.4), labels=("near_boundary", "mid", "far")),
            CategoricalPassthrough(alphabet=("intra", "adjacent", "cross")),
            CategoricalPassthrough(alphabet=("robust", "shifts_conf", "flips")),
        )
    )


def vision_search_space(data: VisionData) -> SearchSpace:
    """Continuous box over the PCA latent space (2nd-98th percentile bounds)."""
    lo, hi = data.latent_bounds
    return SearchSpace(lower=lo.copy(), upper=hi.copy())


def vision_probe_latents(data: VisionData, n: int = 3000) -> list[np.ndarray]:
    """Probe set = PCA-projected training images (real, feasible latents)."""
    z = data.pca.transform(data.X_train)
    return [z[i] for i in range(min(n, len(z)))]
