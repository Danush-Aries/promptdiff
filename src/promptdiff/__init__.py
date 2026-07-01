"""promptdiff — pytest for LLM prompts.

A regression-testing harness that catches when a prompt or model change silently
regresses outputs. Snapshot/baseline diffing is the core primitive, cases run with
native pytest ergonomics, and every run is hermetic: offline-by-default VCR-style
cassettes make CI real-but-free with zero API spend.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .models import (
    AssertionResult,
    AssertionSpec,
    Case,
    Cassette,
    CassetteEntry,
    CaseResult,
    Completion,
    Report,
    RunResult,
    Suite,
)

__all__ = [
    "__version__",
    "AssertionSpec",
    "AssertionResult",
    "Case",
    "CaseResult",
    "Cassette",
    "CassetteEntry",
    "Completion",
    "Report",
    "RunResult",
    "Suite",
    "run_suite",
    "case",
]


def run_suite(*args, **kwargs):
    """Lazy re-export of :func:`promptdiff.runner.run_suite`."""
    from .runner import run_suite as _run_suite

    return _run_suite(*args, **kwargs)


def case(*args, **kwargs):
    """Lazy re-export of the ``@case`` decorator for Python-defined cases."""
    from .pytest_plugin import case as _case

    return _case(*args, **kwargs)
