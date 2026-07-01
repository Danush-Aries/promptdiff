"""`contains` / `not_contains` — substring membership assertions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import AssertionResult
from .base import Assertion, AssertionConfigError, EvalContext

if TYPE_CHECKING:
    from ..models import AssertionSpec


class ContainsAssertion(Assertion):
    type_name = "contains"

    def __init__(
        self,
        values: list[str],
        case_sensitive: bool = True,
        mode: str = "all",
        negate: bool = False,
        description: str | None = None,
    ) -> None:
        super().__init__(negate=negate, description=description)
        if not values:
            raise AssertionConfigError("contains: 'value' is required")
        if mode not in {"all", "any"}:
            raise AssertionConfigError(f"contains: mode must be 'all'|'any', got {mode!r}")
        self.values = values
        self.case_sensitive = case_sensitive
        self.mode = mode

    @classmethod
    def from_spec(cls, spec: AssertionSpec) -> ContainsAssertion:
        cfg = spec.config
        value = cfg.get("value")
        if value is None:
            raise AssertionConfigError("contains: missing 'value'")
        values = [str(v) for v in value] if isinstance(value, list) else [str(value)]
        mode = "any" if cfg.get("any") else ("all" if cfg.get("all") else cfg.get("mode", "all"))
        return cls(
            values=values,
            case_sensitive=bool(cfg.get("case_sensitive", True)),
            mode=str(mode),
            negate=spec.negate,
            description=spec.description,
        )

    def evaluate(self, output: str, ctx: EvalContext) -> AssertionResult:
        hay = output if self.case_sensitive else output.lower()
        needles = self.values if self.case_sensitive else [v.lower() for v in self.values]
        present = [n for n in needles if n in hay]
        missing = [n for n in needles if n not in hay]
        passed = (len(present) == len(needles)) if self.mode == "all" else bool(present)
        if passed:
            msg = f"output contains {self.mode} of {self.values!r}"
        else:
            msg = f"output missing (mode={self.mode}) {missing!r}"
        return AssertionResult(
            type=self.type_name,
            passed=passed,
            message=msg,
            detail={"present": present, "missing": missing, "mode": self.mode},
        )
