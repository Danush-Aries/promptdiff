"""Reporting — console (rich), JUnit XML, and JSON views of a run."""

from __future__ import annotations

from xml.sax.saxutils import escape

from rich.console import Console
from rich.table import Table

from .models import RunResult


def render_console(run: RunResult, console: Console | None = None) -> None:
    """Print a pass/fail table plus per-assertion detail for failures."""
    console = console or Console()
    counts = run.counts
    table = Table(title=f"promptdiff · {run.suite_name} ({run.provider}/{run.model})")
    table.add_column("case", style="bold")
    table.add_column("result")
    table.add_column("assertions")
    for c in run.case_results:
        if c.error:
            status = "[red]ERROR[/red]"
            detail = c.error
        elif c.passed:
            status = "[green]PASS[/green]"
            detail = f"{len(c.assertion_results)} ok"
        else:
            status = "[red]FAIL[/red]"
            detail = "; ".join(f"{a.type}: {a.message}" for a in c.failed_assertions)
        table.add_row(c.case_name, status, detail)
    console.print(table)

    for c in run.case_results:
        for a in c.failed_assertions:
            if a.detail.get("diff"):
                console.print(f"\n[bold]{c.case_name}[/bold] · {a.type} diff:")
                console.print(a.detail["diff"])

    summary = (
        f"[green]{counts['passed']} passed[/green], "
        f"[red]{counts['failed']} failed[/red], "
        f"[yellow]{counts['errors']} errors[/yellow] "
        f"({counts['total']} total)"
    )
    console.print(summary)


def render_console_str(run: RunResult) -> str:
    """Render the console view into a string (used for ``--out`` / capture)."""
    from io import StringIO

    buf = StringIO()
    render_console(run, Console(file=buf, force_terminal=False, width=100))
    return buf.getvalue()


def to_junit_xml(run: RunResult) -> str:
    """Serialize a run as a JUnit XML testsuite for CI reporters."""
    counts = run.counts
    lines = ['<?xml version="1.0" encoding="utf-8"?>']
    lines.append(
        f'<testsuite name="{escape(run.suite_name)}" tests="{counts["total"]}" '
        f'failures="{counts["failed"]}" errors="{counts["errors"]}">'
    )
    for c in run.case_results:
        lines.append(
            f'  <testcase name="{escape(c.case_name)}" time="{(c.duration_ms or 0) / 1000:.4f}">'
        )
        if c.error:
            lines.append(f'    <error message="{escape(c.error)}"/>')
        elif not c.passed:
            msg = "; ".join(f"{a.type}: {a.message}" for a in c.failed_assertions)
            lines.append(f'    <failure message="{escape(msg)}"/>')
        lines.append("  </testcase>")
    lines.append("</testsuite>")
    return "\n".join(lines) + "\n"


__all__ = ["render_console", "render_console_str", "to_junit_xml"]
