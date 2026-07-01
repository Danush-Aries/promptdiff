"""`regex` — pattern match assertion (search or full_match)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..models import AssertionResult
from .base import Assertion, AssertionConfigError, EvalContext

if TYPE_CHECKING:
    from ..models import AssertionSpec

_FLAG_MAP = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}


class RegexAssertion(Assertion):
    type_name = "regex"

    def __init__(
        self,
        pattern: str,
        flags: str = "",
        full_match: bool = False,
        negate: bool = False,
        description: str | None = None,
    ) -> None:
        super().__init__(negate=negate, description=description)
        flagbits = 0
        for ch in flags:
            if ch not in _FLAG_MAP:
                raise AssertionConfigError(f"regex: unknown flag {ch!r} (use i/m/s/x)")
            flagbits |= _FLAG_MAP[ch]
        try:
            self.regex = re.compile(pattern, flagbits)
        except re.error as e:
            raise AssertionConfigError(f"regex: invalid pattern {pattern!r}: {e}") from e
        self.pattern = pattern
        self.full_match = full_match

    @classmethod
    def from_spec(cls, spec: AssertionSpec) -> RegexAssertion:
        cfg = spec.config
        pattern = cfg.get("pattern")
        if pattern is None:
            raise AssertionConfigError("regex: missing 'pattern'")
        return cls(
            pattern=str(pattern),
            flags=str(cfg.get("flags", "")),
            full_match=bool(cfg.get("full_match", False)),
            negate=spec.negate,
            description=spec.description,
        )

    def evaluate(self, output: str, ctx: EvalContext) -> AssertionResult:
        m = self.regex.fullmatch(output) if self.full_match else self.regex.search(output)
        kind = "fullmatch" if self.full_match else "search"
        passed = m is not None
        msg = f"pattern {self.pattern!r} {kind} " + ("matched" if passed else "did not match")
        return AssertionResult(
            type=self.type_name,
            passed=passed,
            message=msg,
            detail={
                "pattern": self.pattern,
                "kind": kind,
                "matched_span": list(m.span()) if m else None,
            },
        )
