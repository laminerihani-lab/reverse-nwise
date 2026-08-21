"""End-to-end orchestrator for Algorithm 1 (Reverse N-Wise Output-Oriented Testing).

Wires the four steps together:

    1. FeasibilitySAT      -> feasible abstract outputs + feasible s-way tuples
    2. CoveringArrayGen    -> OCA(M; s, q, w) over the feasible universe
    3. InverseMap          -> concrete inputs realising each OCA row (L(x;w*))
    4. Prioritize          -> greedy OCov_s ordering of the realised test suite

and returns a bundle carrying the artefacts and headline metrics (OCov_s, eta_s,
M, realisation rate) used to populate the paper's tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from rnwise.covering_array import GreedyGenerator
from rnwise.feasibility import EmpiricalFeasibility, FeasibilityResult
from rnwise.inverse_map import InverseMapper, InverseMapResult
from rnwise.metrics import output_coverage_from_rows, tuple_efficiency
from rnwise.model import SystemUnderTest
from rnwise.oca import OutputCoveringArray
from rnwise.optimizers.base import Optimizer, Regularizer, SearchSpace
from rnwise.optimizers.jaya import Jaya
from rnwise.partition import PartitionScheme
from rnwise.prioritize import PrioritizationResult, prioritize


@dataclass
class PipelineResult:
    """All artefacts and metrics from one Algorithm 1 run."""

    feasibility: FeasibilityResult
    oca: OutputCoveringArray
    inverse: InverseMapResult
    prioritization: PrioritizationResult
    strength: int

    # headline metrics
    n_tests: int
    ocov_s: float
    eta_s: float
    realisation_rate: float

    def summary(self) -> dict[str, Any]:
        return {
            "s": self.strength,
            "n_tests": self.n_tests,
            "OCov_s": self.ocov_s,
            "eta_s": self.eta_s,
            "realisation_rate": self.realisation_rate,
            "feasible_outputs": self.feasibility.n_feasible_outputs,
            "feasible_tuples": self.feasibility.n_feasible_tuples,
            "oca_rows": self.oca.M,
            "oca_lower_bound": self.oca.lower_bound(),
            "total_inverse_evals": self.inverse.total_evals,
        }


@dataclass
class ReverseNWisePipeline:
    """Configurable Algorithm 1 runner.

    Attributes:
        sut: System under test.
        scheme: Partition scheme (``f_hat`` definition).
        space: Input search space for InverseMap.
        strength: Coverage strength ``s`` (pairwise = 2).
        optimizer: InverseMap optimizer (default parameter-free Jaya).
        lam / regularizer: Inverse-loss regularisation.
        generator: Covering-array generator (default self-contained greedy).
    """

    sut: SystemUnderTest
    scheme: PartitionScheme
    space: SearchSpace
    strength: int = 2
    optimizer: Optimizer = field(default_factory=lambda: Jaya(max_iter=200))
    lam: float = 0.0
    regularizer: Regularizer | None = None
    generator: Any = None
    use_warm_start: bool = True

    def run(
        self,
        probe_inputs: Sequence,
        seed: int = 0,
    ) -> PipelineResult:
        rng = np.random.default_rng(seed)

        # Step 1: feasibility (empirical, model-agnostic).
        feas = EmpiricalFeasibility(self.scheme, self.strength).analyze(
            self.sut, probe_inputs
        )

        # Step 2: covering array over the feasible universe.
        gen = self.generator or GreedyGenerator(self.scheme, self.strength)
        oca = gen.generate(feas, rng)

        # Step 3: invert each row to a concrete input (warm-started by feasibility
        # exemplars when available, so realisation reflects true reachability).
        warm = feas.exemplars if self.use_warm_start else None
        mapper = InverseMapper(
            sut=self.sut,
            scheme=self.scheme,
            space=self.space,
            optimizer=self.optimizer,
            lam=self.lam,
            regularizer=self.regularizer,
            warm_start=warm,
        )
        inv = mapper.run(oca, rng)

        # Step 4: greedy OCov_s prioritisation of the realised suite.
        prio = prioritize(
            inv.test_cases, self.scheme.q, self.strength, feas.feasible_tuples
        )

        realised_rows = inv.realised_abstract_outputs()
        ocov = output_coverage_from_rows(
            realised_rows,
            self.scheme.symbol_space,
            self.strength,
            feasible=feas.feasible_tuples,
        )
        eta = tuple_efficiency(realised_rows, self.scheme.symbol_space, self.strength)

        return PipelineResult(
            feasibility=feas,
            oca=oca,
            inverse=inv,
            prioritization=prio,
            strength=self.strength,
            n_tests=len(inv.test_cases),
            ocov_s=ocov,
            eta_s=eta,
            realisation_rate=inv.realisation_rate,
        )
