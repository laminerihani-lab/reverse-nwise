"""UCI Adult Income benchmark: XGBoost classifier as the system under test.

This is the paper's primary empirical bed (Table 1). We train an XGBoost binary
classifier on the UCI Adult dataset, then expose a structured *behavioural output*
space over which Reverse N-Wise builds an output covering array:

Output factors (q = 9 components of f(x), all ternary):
  0. decision        -- income class as neg / boundary / pos (0.4, 0.6 cuts)
  1. confidence       -- calibrated max-class probability band (low/med/high)
  2. margin_region    -- signed distance to the 0.5 boundary (far_neg/near/far_pos)
  3. sex_consistency  -- effect of flipping the ``sex`` feature on the decision:
                         consistent / weak (prob shift) / inconsistent (flip)
                         (fairness-relevant behavioural property)
  4. age_band         -- coarse age bucket at the queried point (young/mid/senior)
  5. workclass_grp    -- employment behavioural class (private/gov/other)
  6. edu_response     -- decision sensitivity to a +2 shift in education-num
                         (drop/stable/rise of P(>50K))
  7. hours_response   -- decision sensitivity to a +10 shift in hours-per-week
                         (drop/stable/rise)
  8. capgain_regime   -- capital-gain operating regime (none/moderate/high)

These behavioural partitions turn a single scalar-probability classifier into a
9-factor, all-ternary output space (pairwise universe = C(9,2)*3^2 = 324), which
is what makes output-oriented combinatorial coverage meaningful at the scale of
the paper's Table 1.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from rnwise.model import StochasticOutput, SystemUnderTest
from rnwise.optimizers.base import SearchSpace
from rnwise.partition import (
    CategoricalPassthrough,
    PartitionScheme,
    ThresholdBands,
)

TARGET_POS = ">50K"
TARGET_NEG = "<=50K"

# Numeric feature columns we expose to the input search space (the categorical
# columns are held at reference modes to keep the inverse-map space continuous).
NUMERIC_FEATURES = [
    "age",
    "education-num",
    "hours-per-week",
    "capital-gain",
    "capital-loss",
]


@dataclass
class AdultData:
    """Preprocessed Adult dataset split + encoders."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    categorical_modes: dict[str, Any]
    numeric_bounds: dict[str, tuple[float, float]]
    workclass_gov_codes: frozenset[int] = field(default_factory=frozenset)


@lru_cache(maxsize=1)
def load_adult(seed: int = 0) -> AdultData:
    """Load and preprocess UCI Adult (cached). Label-encodes categoricals; splits
    80/20 stratified."""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    ssl._create_default_https_context = ssl._create_unverified_context
    ds = fetch_openml("adult", version=2, as_frame=True)
    df = ds.data.copy()
    y = (ds.target.astype(str) == TARGET_POS).astype(int).to_numpy()

    df = df.replace("?", np.nan)
    for col in df.select_dtypes(include=["category", "object"]).columns:
        df[col] = df[col].astype("object").fillna(df[col].mode(dropna=True)[0])
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = [c for c in df.columns if df[c].dtype == object]
    encoders: dict[str, LabelEncoder] = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    # government workclass codes (encoded ints whose label mentions "gov")
    gov_codes: set[int] = set()
    if "workclass" in encoders:
        wc_le = encoders["workclass"]
        for i, label in enumerate(wc_le.classes_):
            if "gov" in str(label).lower():
                gov_codes.add(i)

    feature_names = list(df.columns)
    cat_modes = {c: int(df[c].mode()[0]) for c in cat_cols}
    num_bounds = {
        c: (float(df[c].quantile(0.01)), float(df[c].quantile(0.99)))
        for c in NUMERIC_FEATURES
    }

    X_tr, X_te, y_tr, y_te = train_test_split(
        df, y, test_size=0.2, random_state=seed, stratify=y
    )
    return AdultData(
        X_train=X_tr.reset_index(drop=True),
        X_test=X_te.reset_index(drop=True),
        y_train=y_tr,
        y_test=y_te,
        feature_names=feature_names,
        categorical_modes=cat_modes,
        numeric_bounds=num_bounds,
        workclass_gov_codes=frozenset(gov_codes),
    )


def train_xgb(data: AdultData, seed: int = 0):
    """Train the XGBoost classifier used as the SUT."""
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    clf.fit(data.X_train.to_numpy(), data.y_train)
    return clf


@dataclass
class AdultSUT:
    """Wrap the trained XGBoost model as a structured-output SUT.

    ``predict`` returns the 6-factor behavioural abstract-output *raw* vector; the
    accompanying :func:`adult_partition_scheme` maps it to symbols.
    """

    model: Any
    data: AdultData
    _q: int = field(default=9, init=False)

    @property
    def q(self) -> int:
        return self._q

    def _template_row(self) -> np.ndarray:
        row = np.zeros(len(self.data.feature_names), dtype=float)
        for c, m in self.data.categorical_modes.items():
            row[self.data.feature_names.index(c)] = m
        for c in NUMERIC_FEATURES:
            lo, hi = self.data.numeric_bounds[c]
            row[self.data.feature_names.index(c)] = (lo + hi) / 2.0
        return row

    def _full_input(self, x_numeric: np.ndarray) -> np.ndarray:
        """Embed the numeric search vector into a full feature row."""
        row = self._template_row()
        for val, c in zip(np.asarray(x_numeric, dtype=float), NUMERIC_FEATURES):
            row[self.data.feature_names.index(c)] = val
        return row

    def _proba(self, full_row: np.ndarray) -> float:
        p = self.model.predict_proba(full_row.reshape(1, -1))[0, 1]
        return float(p)

    def predict(self, x: Any) -> StochasticOutput:
        full = self._full_input(x)
        p = self._proba(full)
        conf = max(p, 1.0 - p)
        margin = p - 0.5
        # 3-way decision: negative / boundary / positive (ternary behavioural class)
        decision = "neg" if p < 0.4 else ("pos" if p > 0.6 else "boundary")

        # sex-flip consistency (fairness behavioural property), 3-way by magnitude
        sex_idx = self.data.feature_names.index("sex")
        flipped = full.copy()
        flipped[sex_idx] = 1.0 - flipped[sex_idx]
        p_flip = self._proba(flipped)
        delta_sex = abs(p_flip - p)
        if (p_flip >= 0.5) != (p >= 0.5):
            consistent = "inconsistent"  # decision flips
        elif delta_sex > 0.05:
            consistent = "weak"  # same decision but material probability shift
        else:
            consistent = "consistent"

        # age band at the queried point
        age = full[self.data.feature_names.index("age")]
        age_band = "young" if age < 30 else ("mid" if age < 50 else "senior")

        # workclass behavioural group (private / gov / other), 3-way
        wc = full[self.data.feature_names.index("workclass")]
        wc_ref = self.data.categorical_modes["workclass"]
        gov_codes = self.data.workclass_gov_codes
        if wc == wc_ref:
            workclass_grp = "private"
        elif wc in gov_codes:
            workclass_grp = "gov"
        else:
            workclass_grp = "other"

        # edu_response: response of P(>50K) to a +2 shift in education-num
        edu_idx = self.data.feature_names.index("education-num")
        edu_up = full.copy()
        edu_up[edu_idx] = edu_up[edu_idx] + 2.0
        d_edu = self._proba(edu_up) - p
        edu_response = "drop" if d_edu < -0.02 else ("rise" if d_edu > 0.02 else "stable")

        # hours_response: response of P(>50K) to a +10 shift in hours-per-week
        hrs_idx = self.data.feature_names.index("hours-per-week")
        hrs_up = full.copy()
        hrs_up[hrs_idx] = hrs_up[hrs_idx] + 10.0
        d_hrs = self._proba(hrs_up) - p
        hours_response = "drop" if d_hrs < -0.02 else ("rise" if d_hrs > 0.02 else "stable")

        # capgain_regime: operating regime of the capital-gain feature at this point
        cg = full[self.data.feature_names.index("capital-gain")]
        capgain_regime = "none" if cg <= 0.0 else ("moderate" if cg < 5000.0 else "high")

        raw = (
            decision,
            conf,
            margin,
            consistent,
            age_band,
            workclass_grp,
            edu_response,
            hours_response,
            capgain_regime,
        )
        return StochasticOutput.deterministic(raw)


def adult_partition_scheme() -> PartitionScheme:
    """The 9-factor, all-ternary partition scheme f_hat for the Adult SUT.

    All factors have 3 semantic levels, giving a pairwise output universe of
    ``C(9,2) * 3^2 = 324`` interactions -- the scale of the paper's Table 1.
    """
    return PartitionScheme(
        (
            CategoricalPassthrough(alphabet=("neg", "boundary", "pos")),
            ThresholdBands(edges=(0.6, 0.85), labels=("low", "med", "high")),
            ThresholdBands(edges=(-0.15, 0.15), labels=("far_neg", "near", "far_pos")),
            CategoricalPassthrough(alphabet=("consistent", "weak", "inconsistent")),
            CategoricalPassthrough(alphabet=("young", "mid", "senior")),
            CategoricalPassthrough(alphabet=("private", "gov", "other")),
            CategoricalPassthrough(alphabet=("drop", "stable", "rise")),
            CategoricalPassthrough(alphabet=("drop", "stable", "rise")),
            CategoricalPassthrough(alphabet=("none", "moderate", "high")),
        )
    )


def adult_search_space(data: AdultData) -> SearchSpace:
    """Continuous box over the numeric features (1st-99th percentile bounds)."""
    lower = np.array([data.numeric_bounds[c][0] for c in NUMERIC_FEATURES])
    upper = np.array([data.numeric_bounds[c][1] for c in NUMERIC_FEATURES])
    # widen age lower bound so the "young" band is reachable
    age_pos = NUMERIC_FEATURES.index("age")
    lower[age_pos] = min(lower[age_pos], 17.0)
    return SearchSpace(lower=lower, upper=upper)
