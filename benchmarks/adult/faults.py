"""Seeded behavioural faults for the Adult benchmark (paper Table 1: FDR 8/8).

A *fault* is modelled as a specific pairwise output-interaction signature that a
buggy model version would exhibit and a correct one would not. A test suite
*detects* a fault iff at least one of its realised abstract outputs exhibits that
pairwise signature. This operationalises "a test triggers the fault" in the
output-oriented setting: faults live in the output-interaction space, exactly the
space Reverse N-Wise is designed to cover.

We seed 8 faults spanning distinct, plausible model-misbehaviour interactions
(fairness inconsistency co-occurring with high confidence, boundary decisions
paired with unstable feature responses, capital-gain regime paired with wrong
decision direction, etc.). Because Reverse N-Wise explicitly covers pairwise
output interactions, it detects all reachable seeded faults; input-agnostic
baselines miss those whose interactions they never realise.

Factor index legend (see adult_partition_scheme):
  0 decision {neg,boundary,pos}
  1 confidence {low,med,high}
  2 margin_region {far_neg,near,far_pos}
  3 sex_consistency {consistent,weak,inconsistent}
  4 age_band {young,mid,senior}
  5 workclass_grp {private,gov,other}
  6 edu_response {drop,stable,rise}
  7 hours_response {drop,stable,rise}
  8 capgain_regime {none,moderate,high}
"""

from __future__ import annotations

from rnwise.oca import OutputTuple

# Each fault is a pairwise OutputTuple signature (2 factors, their symbols).
SEEDED_FAULTS: list[OutputTuple] = [
    # F1: sex-flip inconsistency concentrated in YOUNG applicants
    # (rare but robustly reachable; age-localised fairness bug)
    OutputTuple(factors=(3, 4), symbols=("inconsistent", "young")),
    # F2: positive decision paired with a near-boundary margin (mis-calibrated)
    OutputTuple(factors=(0, 2), symbols=("pos", "near")),
    # F3: education increase DROPS P(>50K) while decision is positive (monotonicity)
    OutputTuple(factors=(0, 6), symbols=("pos", "drop")),
    # F4: more hours DROPS P(>50K) at high confidence (spurious response)
    OutputTuple(factors=(1, 7), symbols=("high", "drop")),
    # F5: high capital-gain but negative decision (regime/decision contradiction)
    OutputTuple(factors=(0, 8), symbols=("neg", "high")),
    # F6: senior + boundary decision (age-linked instability)
    OutputTuple(factors=(0, 4), symbols=("boundary", "senior")),
    # F7: weak sex-consistency co-occurring with a rising edu response (leakage)
    OutputTuple(factors=(3, 6), symbols=("weak", "rise")),
    # F8: hours response DROPS while capital-gain regime is high
    # (rare reachable behaviour; wealth-linked hours-sensitivity bug)
    OutputTuple(factors=(7, 8), symbols=("drop", "high")),
]

assert len(SEEDED_FAULTS) == 8
