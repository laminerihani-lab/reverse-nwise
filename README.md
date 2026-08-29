# Reverse N-Wise Output-Oriented Testing — Experimental Artifact

Reference implementation and empirical evaluation of **Reverse N-Wise
Output-Oriented Testing for AI/ML and Quantum Computing Systems**.

Classical combinatorial testing builds covering arrays over *inputs*. Reverse
N-Wise inverts this: it builds covering arrays over abstracted **outputs**
(behavioural properties of the system under test), then recovers concrete inputs
by gradient-free inverse mapping. This targets *output* interaction coverage —
the behaviours a model actually exhibits — rather than input combinations that
may all collapse to the same output.

The package implements Algorithm 1 end-to-end and reproduces every table/figure
in the empirical section from scratch on commodity hardware.

## What's here

```
rnwise/            core method (the tool)
  model.py           SUT protocol  f: X -> P(Y)
  partition.py       semantic output partitioners  pi_j: Y_j -> W_j
  oca.py             Output Covering Array OCA(M; s, q, w) + coverage math
  feasibility.py     Algorithm 1 Step 1  (feasible output enumeration)
  covering_array.py  Algorithm 1 Step 2  (greedy / exhaustive / ACTS / PICT)
  inverse_map.py     Algorithm 1 Step 3  (optimizer dispatch + warm start)
  prioritize.py      Algorithm 1 Step 4  (greedy incremental OCov_s ordering)
  pipeline.py        end-to-end orchestrator
  metrics.py         OCov_s, eta_s, FDR
  optimizers/        Jaya, Whale, Firefly, Bayesian Opt, VQE (quantum)
baselines/         Input CT, Random, Property-Based, Metamorphic, DeepCT
benchmarks/
  adult/             UCI Adult + XGBoost SUT, 9-factor output space, 8 faults
  vision/            digits MLP over a PCA latent space (decision-boundary output)
  nlp/               20-newsgroups TF-IDF+SVD text classifier (embedding-cluster output)
  quantum/           self-contained NISQ statevector SUT (no qiskit needed)
experiments/       exp01..exp06 (see below)
analysis/          stats (MWU / bootstrap / Cliff's delta), LaTeX tables, plots
tests/             76 fast unit tests + 5 slow integration tests
configs/           declarative experiment configs + fixed seeds
```

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[analysis]"       # core + matplotlib/seaborn
# macOS + XGBoost also needs the OpenMP runtime:  brew install libomp
```

Optional extras: `.[bayesopt]` (scikit-optimize), `.[quantum]` (qiskit — NOT
required; the bundled NISQ simulator is dependency-free), `.[dev]` (pytest/ruff).

## Reproduce the results

```bash
# exp01 — single-run method vs baselines on Adult + XGBoost (pairwise)
python -m experiments.exp01_table1_adult --seed 0 --json results/exp01.json

# exp04 — 30-run significance study (means, 95% bootstrap CIs, MWU + Cliff's d)
python -m experiments.exp04_significance --runs 30 --json results/exp04.json

# exp03 — coverage-strength scaling (s = 2, 3, 4) + Corollary 1 bound check
python -m experiments.exp03_strength --strengths 2 3 4 --json results/exp03.json

# exp05 — ablations (optimizer / output-granularity / lambda)
python -m experiments.exp05_ablations --seed 0 --json results/exp05.json

# exp06 — cost profile (per-step timing, memory, budget ratio)
python -m experiments.exp06_cost --seed 0 --json results/exp06.json

# exp02 — multi-domain generality: quantum + vision + NLP SUTs
python -m experiments.exp02_multidomain --domains quantum vision nlp --json results/exp02.json

# regenerate LaTeX tables + figures from the JSON artifacts
python -m analysis.tables --exp04 results/exp04.json --exp03 results/exp03.json --out results/tables
python -m analysis.plots  --exp04 results/exp04.json --exp03 results/exp03.json --out results/figures
```

## Headline measured results (30 runs, Adult + XGBoost, pairwise s=2)

| Method | Tests | OCov₂ (95% CI) | FDR (95% CI) |
|---|---:|---|---|
| **Reverse N-Wise** | 20 | **100.0% [100.0, 100.0]** | **98.3% [96.7, 99.6]** |
| DeepCT | 17 | 91.2% [91.1, 91.3] | 99.6% [98.8, 100.0] |
| Property-Based | 20 | 58.6% [56.6, 60.7] | 34.2% [28.3, 40.0] |
| Input CT | 13 | 53.9% [51.5, 56.4] | 35.0% [29.6, 40.4] |
| Metamorphic | 19 | 53.4% [49.8, 57.3] | 28.7% [22.9, 34.6] |
| Random | 20 | 52.5% [50.4, 54.9] | 31.2% [24.6, 37.9] |

Reverse N-Wise reaches **perfect output coverage on every run** (Mann-Whitney U
vs. every baseline, including DeepCT: p ≈ 6×10⁻¹³, Cliff's δ = 1.0). On fault
detection it is statistically tied with the strongest baseline (DeepCT,
p = 0.92) and dominates the other four (p < 10⁻¹²), while using direct inversion
at **~8% of an exhaustive-input search budget** rather than a large random pool.

Strength scaling (exp03): OCov_s stays at 100% for s = 2, 3, 4 while the suite
grows sub-linearly with the output universe (324 → 2,268 → 10,206), and the
Corollary 1 bound M ≥ vˢ holds at every strength.

## Multi-domain generality (exp02)

The *same* Algorithm 1 pipeline, applied across three structurally different
domains (each with its own SUT, output partition, search space, and inverse-map
optimizer), vs. a matched-budget random baseline:

| Domain | Optimizer | RNWise OCov₂ / FDR | Random OCov₂ / FDR |
|---|---|---|---|
| Quantum (NISQ circuit) | VQE | **100% / 5/5** | 73.0% / 2/5 |
| Vision (digits MLP) | Jaya | **100% / 5/5** | 61.6% / 0/5 |
| NLP (20-newsgroups) | Jaya | **100% / 5/5** | 74.1% / 1/5 |

Together with the Adult study, this exercises the method on tabular ML, vision,
NLP, and quantum systems — output-oriented coverage is domain-agnostic.

## Tests

```bash
pytest              # 76 fast unit tests (slow tests skipped by default)
pytest -m ""        # all 81, incl. slow integration tests (train models, full pipeline)
```

## Determinism

All experiments take an explicit `--seed`; `configs/seeds.yaml` pins the 30 seeds
used for the significance study. Results above were produced on Python 3.12,
macOS/arm64; exact floating values may shift slightly across BLAS/XGBoost builds
but the qualitative and statistical conclusions are stable.
