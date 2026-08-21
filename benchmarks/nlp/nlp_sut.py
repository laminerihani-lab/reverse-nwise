"""NLP benchmark: a text classifier over an EMBEDDING-CLUSTER output space.

Uses sklearn ``20 newsgroups`` (4 categories) with a TF-IDF -> TruncatedSVD
embedding and a logistic-regression classifier -- dependency-free (no
transformers/torch). The Reverse N-Wise pipeline builds a covering array over the
classifier's behaviour w.r.t. semantic embedding clusters and inverts to find
latent points (decoded to nearest real documents) that realise each behaviour.

The inverse-map search operates in the compact SVD latent space (default 16
dims); a decoded "input" is the nearest training document to the latent point
(nearest-neighbour decode), keeping synthesised tests on the real text manifold.

Output factors (q components of f(x), all ternary):
  0. pred_class      -- predicted category bucketed into 3 groups
  1. confidence       -- max class-probability band (low / med / high)
  2. embed_cluster    -- KMeans cluster of the embedding, bucketed to 3 super-clusters
  3. cluster_consist  -- is the predicted class the majority class of the doc's
                        embedding cluster? (consistent / minority / off) -- checks
                        whether the decision agrees with local embedding semantics
  4. dropout_robust   -- does the decision survive random token dropout?
                        (robust / shifts_conf / flips) -- paraphrase-style robustness
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

CATEGORIES = ["sci.space", "rec.sport.baseball", "talk.politics.guns", "comp.graphics"]
N_SVD = 16       # embedding dimensionality (inverse-map search space)
N_CLUSTERS = 6   # KMeans clusters over embeddings


def _class_group(label: int) -> str:
    return ("g0", "g1", "g2")[min(label, 2)]


def _cluster_group(cluster: int) -> str:
    if cluster < N_CLUSTERS // 3:
        return "c0"
    if cluster < 2 * N_CLUSTERS // 3:
        return "c1"
    return "c2"


@dataclass
class NLPData:
    embeddings_train: np.ndarray      # SVD latent of training docs
    y_train: np.ndarray
    embeddings_test: np.ndarray
    y_test: np.ndarray
    vectorizer: Any
    svd: Any
    kmeans: Any
    cluster_majority: dict[int, int]  # cluster -> majority class
    latent_bounds: tuple[np.ndarray, np.ndarray]
    raw_train_docs: list[str]


@lru_cache(maxsize=1)
def load_nlp(seed: int = 0) -> NLPData:
    """Load 20-newsgroups, build TF-IDF+SVD embeddings and KMeans clusters."""
    from collections import Counter

    from sklearn.cluster import KMeans
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    ssl._create_default_https_context = ssl._create_unverified_context
    strip = ("headers", "footers", "quotes")
    tr = fetch_20newsgroups(subset="train", categories=CATEGORIES, remove=strip, random_state=seed)
    te = fetch_20newsgroups(subset="test", categories=CATEGORIES, remove=strip, random_state=seed)

    vec = TfidfVectorizer(max_features=4000, stop_words="english", min_df=3)
    Xtr = vec.fit_transform(tr.data)
    Xte = vec.transform(te.data)

    svd = TruncatedSVD(n_components=N_SVD, random_state=seed)
    Etr = svd.fit_transform(Xtr)
    Ete = svd.transform(Xte)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=seed, n_init=10).fit(Etr)
    cl_tr = km.labels_
    majority: dict[int, int] = {}
    for c in range(N_CLUSTERS):
        members = tr.target[cl_tr == c]
        majority[c] = int(Counter(members).most_common(1)[0][0]) if len(members) else 0

    lo = np.quantile(Etr, 0.02, axis=0)
    hi = np.quantile(Etr, 0.98, axis=0)
    return NLPData(
        embeddings_train=Etr,
        y_train=tr.target,
        embeddings_test=Ete,
        y_test=te.target,
        vectorizer=vec,
        svd=svd,
        kmeans=km,
        cluster_majority=majority,
        latent_bounds=(lo, hi),
        raw_train_docs=list(tr.data),
    )


def train_text_clf(data: NLPData, seed: int = 0):
    """Train the text classifier (logistic regression on embeddings) as the SUT."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(data.embeddings_train, data.y_train)
    return clf


@dataclass
class NLPSUT:
    """Wrap the text classifier as a structured-output SUT over embeddings.

    Inputs are SVD latent vectors; :meth:`decode` maps a latent point to its
    nearest training document (keeping tests on the real text manifold).
    """

    model: Any
    data: NLPData
    _q: int = field(default=5, init=False)

    @property
    def q(self) -> int:
        return self._q

    def decode(self, z: np.ndarray) -> str:
        """Latent -> nearest real training document (index-based NN decode)."""
        z = np.asarray(z, dtype=float)
        d = np.linalg.norm(self.data.embeddings_train - z, axis=1)
        return self.data.raw_train_docs[int(np.argmin(d))]

    def _proba(self, z: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(z.reshape(1, -1))[0]

    def predict(self, z: Any) -> StochasticOutput:
        z = np.asarray(z, dtype=float)
        p = self._proba(z)
        top = int(np.argmax(p))
        conf = float(p[top])

        order = np.argsort(p)[::-1]
        margin = float(p[order[0]] - p[order[1]])  # kept for potential extension

        pred_class = _class_group(top)

        cluster = int(self.data.kmeans.predict(z.reshape(1, -1))[0])
        embed_cluster = _cluster_group(cluster)

        maj = self.data.cluster_majority.get(cluster, top)
        if top == maj:
            cluster_consist = "consistent"
        elif conf >= 0.5:
            cluster_consist = "minority"  # confident but against local cluster
        else:
            cluster_consist = "off"

        # token-dropout robustness: perturb the latent toward the cluster centroid
        # (a proxy for dropping discriminative tokens) and re-classify.
        centroid = self.data.kmeans.cluster_centers_[cluster]
        z_drop = 0.7 * z + 0.3 * centroid
        p_d = self._proba(z_drop)
        new_top = int(np.argmax(p_d))
        if new_top != top:
            dropout_robust = "flips"
        elif abs(float(p_d[new_top]) - conf) > 0.15:
            dropout_robust = "shifts_conf"
        else:
            dropout_robust = "robust"

        raw = (pred_class, conf, embed_cluster, cluster_consist, dropout_robust)
        return StochasticOutput.deterministic(raw)


def nlp_partition_scheme() -> PartitionScheme:
    """5-factor behavioural output partition for the NLP SUT."""
    return PartitionScheme(
        (
            CategoricalPassthrough(alphabet=("g0", "g1", "g2")),
            ThresholdBands(edges=(0.5, 0.8), labels=("low", "med", "high")),
            CategoricalPassthrough(alphabet=("c0", "c1", "c2")),
            CategoricalPassthrough(alphabet=("consistent", "minority", "off")),
            CategoricalPassthrough(alphabet=("robust", "shifts_conf", "flips")),
        )
    )


def nlp_search_space(data: NLPData) -> SearchSpace:
    """Continuous box over the SVD embedding space (2nd-98th percentile bounds)."""
    lo, hi = data.latent_bounds
    return SearchSpace(lower=lo.copy(), upper=hi.copy())


def nlp_probe_embeddings(data: NLPData, n: int = 2000) -> list[np.ndarray]:
    """Probe set = real training-document embeddings (feasible latents)."""
    e = data.embeddings_train
    return [e[i] for i in range(min(n, len(e)))]
