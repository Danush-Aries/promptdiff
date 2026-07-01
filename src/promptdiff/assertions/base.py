"""Assertion ABC and the evaluation context.

Every assertion is a pure function of ``(output_text, ctx) -> AssertionResult``.
The base class applies generic ``negate`` after ``evaluate`` so that, e.g.,
``not_contains`` is just ``contains`` with ``negate=True``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ..models import AssertionResult

if TYPE_CHECKING:
    from ..models import AssertionSpec, Case


class AssertionConfigError(Exception):
    """Raised at construction time for a malformed assertion spec (e.g. bad regex)."""


@dataclass
class EvalContext:
    """Side-channel state an assertion may need (snapshot dirs, the owning case)."""
    case: Case | None = None
    snapshot_dir: str = "__snapshots__"
    baseline_dir: str = "baselines"
    update_snapshots: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class Assertion(ABC):
    """Base class. Subclasses implement ``evaluate`` and ``from_spec``."""

    type_name: str = "base"

    def __init__(self, negate: bool = False, description: str | None = None) -> None:
        self.negate = negate
        self.description = description

    @classmethod
    @abstractmethod
    def from_spec(cls, spec: AssertionSpec) -> Assertion:
        """Construct from a parsed :class:`AssertionSpec`. Validation happens here."""

    @abstractmethod
    def evaluate(self, output: str, ctx: EvalContext) -> AssertionResult:
        """Return an :class:`AssertionResult`; must not apply ``negate`` itself."""

    def run(self, output: str, ctx: EvalContext) -> AssertionResult:
        """Public entry: evaluate then apply generic negation."""
        result = self.evaluate(output, ctx)
        if not self.negate:
            return result
        return replace(
            result,
            passed=not result.passed,
            message=f"[negated] {result.message}",
            detail={**result.detail, "negated": True},
        )
