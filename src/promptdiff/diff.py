"""Baseline-vs-current report comparison — the CI regression gate.

Aligns two reports by ``case_name`` and classifies each case:
  * ``regression`` — was passing, now failing (this is what fails the build)
  * ``fixed``      — was failing, now passing
  * ``new``        — present only in current
  * ``removed``    — present only in baseline
  * ``unchanged``  — same pass/fail status
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Report


@dataclass(frozen=True)
class CaseDelta:
    case_name: str
    classification: str            # regression | fixed | new | removed | unchanged
    baseline_passed: bool | None
    current_passed: bool | None


@dataclass
class DiffReport:
    deltas: list[CaseDelta] = field(default_factory=list)

    @property
    def regressions(self) -> list[CaseDelta]:
        return [d for d in self.deltas if d.classification == "regression"]

    @property
    def fixed(self) -> list[CaseDelta]:
        return [d for d in self.deltas if d.classification == "fixed"]

    @property
    def new(self) -> list[CaseDelta]:
        return [d for d in self.deltas if d.classification == "new"]

    @property
    def removed(self) -> list[CaseDelta]:
        return [d for d in self.deltas if d.classification == "removed"]

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)

    def counts(self) -> dict[str, int]:
        out = {"regression": 0, "fixed": 0, "new": 0, "removed": 0, "unchanged": 0}
        for d in self.deltas:
            out[d.classification] += 1
        return out


def diff_reports(baseline: Report, current: Report) -> DiffReport:
    """Compare two reports and classify each case."""
    base = {c.case_name: c.passed for c in baseline.run.case_results}
    curr = {c.case_name: c.passed for c in current.run.case_results}
    names = sorted(set(base) | set(curr))

    deltas: list[CaseDelta] = []
    for name in names:
        b = base.get(name)
        c = curr.get(name)
        if b is None:
            klass = "new"
        elif c is None:
            klass = "removed"
        elif b and not c:
            klass = "regression"
        elif not b and c:
            klass = "fixed"
        else:
            klass = "unchanged"
        deltas.append(CaseDelta(case_name=name, classification=klass,
                                baseline_passed=b, current_passed=c))
    return DiffReport(deltas=deltas)


def render_diff_console(diff: DiffReport, console=None) -> None:
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    table = Table(title="promptdiff · regression diff")
    table.add_column("case", style="bold")
    table.add_column("change")
    styles = {"regression": "[red]REGRESSION[/red]", "fixed": "[green]fixed[/green]",
              "new": "[cyan]new[/cyan]", "removed": "[yellow]removed[/yellow]",
              "unchanged": "unchanged"}
    for d in diff.deltas:
        if d.classification == "unchanged":
            continue
        table.add_row(d.case_name, styles[d.classification])
    console.print(table)
    counts = diff.counts()
    console.print(
        f"[red]{counts['regression']} regressions[/red], "
        f"[green]{counts['fixed']} fixed[/green], "
        f"{counts['new']} new, {counts['removed']} removed"
    )


__all__ = ["CaseDelta", "DiffReport", "diff_reports", "render_diff_console"]
