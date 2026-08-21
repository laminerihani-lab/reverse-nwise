"""Reverse N-Wise Output-Oriented Testing.

A tool implementing Algorithm 1 from Rihani, "Reverse N-Wise Output-Oriented
Testing for AI/ML and Quantum Computing Systems". The method builds covering
arrays over abstracted *outputs* and inverts them to recover input test cases.
"""

from __future__ import annotations

__version__ = "0.1.0"

from rnwise.model import SystemUnderTest, Output, StochasticOutput
from rnwise.partition import Partitioner, PartitionScheme, register_partitioner, get_partitioner
from rnwise.oca import OutputCoveringArray, OutputTuple, AbstractOutput
from rnwise.metrics import output_coverage, tuple_efficiency, fault_detection_rate

__all__ = [
    "__version__",
    "SystemUnderTest",
    "Output",
    "StochasticOutput",
    "Partitioner",
    "PartitionScheme",
    "register_partitioner",
    "get_partitioner",
    "OutputCoveringArray",
    "OutputTuple",
    "AbstractOutput",
    "output_coverage",
    "tuple_efficiency",
    "fault_detection_rate",
]
