"""`exact` — string equality assertion (with optional strip)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import AssertionResult
from .base import Assertion, AssertionConfigError, EvalContext

if TYPE_CHECKING:
    from ..models import AssertionSpec


class ExactAssertion(Assertion):
    type_name = "exact"

    def __init__(self, value: str, strip: bool = True,
                 negate: bool = False, description: str | None = None) -> None:
        super().__init__(negate=negate, description=description)
        self.value = value
        self.strip = strip

    @classmethod
    def from_spec(cls, spec: AssertionSpec) -> ExactAssertion:
        cfg = spec.config
        if "value" not in cfg:
            raise AssertionConfigError("exact: missing 'value'")
        return cls(
            value=str(cfg["value"]),
            strip=bool(cfg.get("strip", True)),
            negate=spec.negate,
            description=spec.description,
        )

    def evaluate(self, output: str, ctx: EvalContext) -> AssertionResult:
        got = output.strip() if self.strip else output
        want = self.value.strip() if self.strip else self.value
        passed = got == want
        msg = "output equals expected value" if passed else "output differs from expected value"
        return AssertionResult(
            type=self.type_name,
            passed=passed,
            message=msg,
            detail={"expected": want, "got": got, "strip": self.strip},
        )
