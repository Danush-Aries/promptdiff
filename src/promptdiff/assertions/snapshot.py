"""`snapshot` — record-once / compare-forever baseline assertion.

First run with no stored snapshot records the output and passes (marked "new").
Subsequent runs compare byte-for-byte and fail with a unified diff on drift.
``update=True`` (from ``--update-snapshots``) rewrites the snapshot and passes.
"""
from __future__ import annotations

import difflib
import os
from typing import TYPE_CHECKING

from ..models import AssertionResult
from .base import Assertion, EvalContext

if TYPE_CHECKING:
    from ..models import AssertionSpec


class SnapshotAssertion(Assertion):
    type_name = "snapshot"

    def __init__(self, snapshot_id: str | None = None, update: bool = False,
                 negate: bool = False, description: str | None = None) -> None:
        super().__init__(negate=negate, description=description)
        self.snapshot_id = snapshot_id
        self.update = update

    @classmethod
    def from_spec(cls, spec: AssertionSpec) -> SnapshotAssertion:
        cfg = spec.config
        return cls(
            snapshot_id=cfg.get("id"),
            update=bool(cfg.get("update", False)),
            negate=spec.negate,
            description=spec.description,
        )

    def _path(self, ctx: EvalContext) -> str:
        case = ctx.case
        suite = "suite"
        case_name = self.snapshot_id or "snapshot"
        if case is not None:
            case_name = self.snapshot_id or case.name
            if case.source_file:
                suite = os.path.splitext(os.path.basename(case.source_file))[0]
        return os.path.join(ctx.snapshot_dir, suite, f"{case_name}.txt")

    def evaluate(self, output: str, ctx: EvalContext) -> AssertionResult:
        path = self._path(ctx)
        should_update = self.update or ctx.update_snapshots

        if not os.path.exists(path) or should_update:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(output)
            status = "updated" if (should_update and os.path.exists(path)) else "new"
            return AssertionResult(
                type=self.type_name,
                passed=True,
                message=f"snapshot {status}: {path}",
                detail={"status": status, "path": path},
            )

        with open(path, encoding="utf-8") as fh:
            stored = fh.read()
        if stored == output:
            return AssertionResult(
                type=self.type_name,
                passed=True,
                message=f"snapshot match: {path}",
                detail={"status": "match", "path": path},
            )
        diff = "".join(
            difflib.unified_diff(
                stored.splitlines(keepends=True),
                output.splitlines(keepends=True),
                fromfile="snapshot",
                tofile="output",
            )
        )
        return AssertionResult(
            type=self.type_name,
            passed=False,
            message=f"snapshot drift: {path} (run with --update-snapshots to accept)",
            detail={"status": "drift", "path": path, "diff": diff},
        )
