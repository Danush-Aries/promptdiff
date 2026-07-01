"""promptdiff CLI.

    promptdiff run   PATHS...   # load suites, run assertions offline, report
    promptdiff record PATHS...  # capture live outputs into cassettes (needs API key)
    promptdiff diff  BASE CUR   # regression-gate two JSON reports
    promptdiff list  PATHS...   # list cases / assertion types
    promptdiff init             # scaffold an example suite
    promptdiff version

Exit codes: 0 all pass · 1 assertion failure/regression · 2 config/load error
            · 3 cassette miss in offline mode.
"""

from __future__ import annotations

from pathlib import Path

import typer

from . import __version__, assertions
from .diff import diff_reports, render_diff_console
from .loader import LoaderError, discover_suites
from .models import Report
from .providers import list_providers
from .reporting import render_console, to_junit_xml
from .runner import build_report, run_suite

app = typer.Typer(add_completion=False, help="pytest for LLM prompts — offline regression testing.")


def _load(paths: list[str]):
    try:
        suites = discover_suites(paths)
    except LoaderError as e:
        typer.secho(f"error: {e}", fg="red", err=True)
        raise typer.Exit(2) from e
    if not suites:
        typer.secho(f"error: no suites found in {paths}", fg="red", err=True)
        raise typer.Exit(2)
    return suites


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(f"promptdiff {__version__}")


@app.command(name="list")
def list_cmd(
    paths: list[str] = typer.Argument(None, help="suite files/dirs to inspect"),
    show_assertions: bool = typer.Option(False, "--assertions", help="list assertion types"),
    show_cases: bool = typer.Option(False, "--cases", help="list discovered cases"),
) -> None:
    """List discovered cases and/or the registered assertion types."""
    if show_assertions or not (show_cases or paths):
        typer.echo("assertion types:")
        for name in assertions.list_types():
            typer.echo(f"  {name}")
        typer.echo(f"providers: {', '.join(list_providers())}")
    if paths and (show_cases or not show_assertions):
        for suite in _load(paths):
            typer.echo(f"\n{suite.name}  ({suite.source_file})")
            for c in suite.cases:
                tags = f" [{', '.join(c.tags)}]" if c.tags else ""
                typer.echo(f"  {c.name}  ({len(c.assertions)} assertions){tags}")


@app.command()
def run(
    paths: list[str] = typer.Argument(..., help="suite YAML files or directories"),
    provider: str | None = typer.Option(None, "--provider", help="override provider"),
    record_mode: str = typer.Option("once", "--record-mode", help="none|once|new_episodes|all"),
    cassette_dir: str = typer.Option("tests/cassettes", "--cassette-dir"),
    snapshot_dir: str = typer.Option("__snapshots__", "--snapshot-dir"),
    update_snapshots: bool = typer.Option(False, "--update-snapshots"),
    report: str = typer.Option("console", "--report", help="console|json|junit"),
    out: Path | None = typer.Option(None, "--out", help="write report to FILE"),
    k: str | None = typer.Option(None, "-k", help="filter cases by name/tag substring"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    offline: bool = typer.Option(
        True, "--offline/--online", help="force ReplayProvider (default: on)"
    ),
) -> None:
    """Run suites and report. Offline by default — zero network, zero cost."""
    suites = _load(paths)
    all_pass = True
    saw_miss = False
    reports: list[Report] = []

    for suite in suites:
        run_result = run_suite(
            suite,
            provider=provider,
            record_mode=record_mode,
            cassette_dir=cassette_dir,
            offline=offline,
            snapshot_dir=snapshot_dir,
            update_snapshots=update_snapshots,
            name_filter=k,
            fail_fast=fail_fast,
        )
        reports.append(build_report(run_result))
        all_pass = all_pass and run_result.passed
        for c in run_result.case_results:
            if c.error and "CassetteMiss" in c.error:
                saw_miss = True

    if report == "json":
        payload = "\n".join(r.to_json() for r in reports)
        _emit(payload, out)
    elif report == "junit":
        payload = "\n".join(to_junit_xml(r.run) for r in reports)
        _emit(payload, out)
    else:
        for r in reports:
            render_console(r.run)

    if saw_miss and offline:
        raise typer.Exit(3)
    raise typer.Exit(0 if all_pass else 1)


@app.command()
def record(
    paths: list[str] = typer.Argument(..., help="suite YAML files or directories"),
    provider: str | None = typer.Option(None, "--provider"),
    cassette_dir: str = typer.Option("tests/cassettes", "--cassette-dir"),
    overwrite: bool = typer.Option(False, "--overwrite", help="re-record existing entries"),
) -> None:
    """Capture live provider outputs into cassettes (needs an API key)."""
    suites = _load(paths)
    mode = "all" if overwrite else "new_episodes"
    for suite in suites:
        run_result = run_suite(
            suite,
            provider=provider,
            record_mode=mode,
            cassette_dir=cassette_dir,
            offline=False,
        )
        typer.echo(f"recorded {suite.name}: {run_result.counts}")


@app.command(name="diff")
def diff_cmd(
    baseline: Path = typer.Argument(..., help="baseline report JSON"),
    current: Path = typer.Argument(..., help="current report JSON"),
    fail_on_regression: bool = typer.Option(True, "--fail-on-regression/--no-fail-on-regression"),
    format: str = typer.Option("console", "--format", help="console|json"),
) -> None:
    """Compare two JSON reports; non-zero exit on any regression."""
    try:
        base = Report.from_json(baseline.read_text())
        curr = Report.from_json(current.read_text())
    except (OSError, ValueError) as e:
        typer.secho(f"error: {e}", fg="red", err=True)
        raise typer.Exit(2) from e

    d = diff_reports(base, curr)
    if format == "json":
        import json

        typer.echo(
            json.dumps(
                {"counts": d.counts(), "regressions": [x.case_name for x in d.regressions]},
                indent=2,
            )
        )
    else:
        render_diff_console(d)

    if fail_on_regression and d.has_regressions:
        raise typer.Exit(1)
    raise typer.Exit(0)


@app.command()
def init(
    directory: Path = typer.Option(Path("."), "--dir", help="target directory"),
) -> None:
    """Scaffold an example suite and cassette directory."""
    prompts_dir = directory / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (directory / "tests" / "cassettes").mkdir(parents=True, exist_ok=True)
    example = prompts_dir / "example.promptdiff.yaml"
    example.write_text(_EXAMPLE_SUITE)
    typer.echo(f"wrote {example}")
    typer.echo("next: `promptdiff record prompts/` then `promptdiff run prompts/`")


def _emit(payload: str, out: Path | None) -> None:
    if out:
        out.write_text(payload)
        typer.echo(f"wrote report to {out}")
    else:
        typer.echo(payload)


_EXAMPLE_SUITE = """\
name: example
defaults:
  provider: replay
  model: gpt-4o-mini
  params:
    temperature: 0
cases:
  - name: capital_of_france
    prompt: "What is the capital of France? Answer in one word."
    assertions:
      - type: contains
        value: Paris
        case_sensitive: false
  - name: json_output
    prompt: "Return a JSON object with name and age."
    assertions:
      - type: json_schema
        schema:
          type: object
          required: [name, age]
          properties:
            name: { type: string }
            age: { type: integer, minimum: 0 }
"""

if __name__ == "__main__":
    app()
