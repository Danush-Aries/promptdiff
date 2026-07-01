"""Runner — orchestrates load -> resolve provider -> run cases -> assert.

Provider/render errors are captured as failing :class:`CaseResult`s (never crash the
whole run). Ordering is deterministic. Offline replay is the default so the same code
path is exercised in CI with zero network.
"""

from __future__ import annotations

import datetime as _dt
import os
import time

from . import __version__, assertions
from .assertions import AssertionConfigError, EvalContext
from .models import (
    AssertionResult,
    Case,
    CaseResult,
    Cassette,
    Completion,
    Report,
    RunResult,
    Suite,
)
from .providers import Provider, get_provider
from .providers.cassette import RecordingProvider
from .providers.replay import ReplayProvider


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def cassette_path_for(suite: Suite, cassette_dir: str) -> str:
    return os.path.join(cassette_dir, f"{suite.name}.json")


def resolve_provider(
    suite: Suite,
    provider_name: str | None,
    record_mode: str,
    cassette: Cassette,
    offline: bool,
) -> tuple[Provider, str]:
    """Return ``(provider, effective_name)`` per the run configuration.

    * ``offline`` or ``record_mode == 'none'`` with a replay provider -> ReplayProvider.
    * a live provider name with a real record mode -> RecordingProvider over the SDK.
    """
    name = provider_name or suite.defaults.get("provider", "replay")

    # Hermetic path: forced offline, an explicit replay provider, or "no recording".
    if offline or name == "replay" or record_mode == "none":
        return ReplayProvider(cassette), "replay"

    # Live/recording path (only reached when explicitly recording).
    inner = get_provider(name, suite.defaults.get("model", ""))
    return RecordingProvider(inner, cassette, mode=record_mode), name


def _run_assertions(case: Case, text: str, ctx: EvalContext) -> list[AssertionResult]:
    results: list[AssertionResult] = []
    for spec in case.assertions:
        try:
            assertion = assertions.build(spec)
        except AssertionConfigError as e:
            results.append(
                AssertionResult(
                    type=spec.type,
                    passed=False,
                    message=f"config error: {e}",
                    detail={"error": "config"},
                )
            )
            continue
        results.append(assertion.run(text, ctx))
    return results


def run_case(
    case: Case,
    provider: Provider,
    effective_provider: str,
    snapshot_dir: str = "__snapshots__",
    update_snapshots: bool = False,
) -> CaseResult:
    """Execute a single case end-to-end, capturing errors as a failing result."""
    t0 = time.perf_counter()
    try:
        prompt = case.rendered_prompt
    except (KeyError, ValueError) as e:
        return CaseResult(
            case_name=case.name,
            passed=False,
            completion=Completion(text="", model=case.model, provider=effective_provider),
            error=f"prompt render error: {e}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    params = dict(case.params)
    params["_model"] = case.model
    params["_provider"] = case.provider
    try:
        completion = provider.generate(prompt, params)
    except Exception as e:  # provider errors are failing results, not crashes
        return CaseResult(
            case_name=case.name,
            passed=False,
            completion=Completion(text="", model=case.model, provider=effective_provider),
            error=f"{type(e).__name__}: {e}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    ctx = EvalContext(case=case, snapshot_dir=snapshot_dir, update_snapshots=update_snapshots)
    results = _run_assertions(case, completion.text, ctx)
    passed = all(r.passed for r in results)
    return CaseResult(
        case_name=case.name,
        passed=passed,
        completion=completion,
        assertion_results=results,
        error=None,
        duration_ms=(time.perf_counter() - t0) * 1000,
    )


def run_suite(
    suite: Suite,
    provider: str | None = None,
    record_mode: str = "once",
    cassette_dir: str = "tests/cassettes",
    offline: bool = True,
    snapshot_dir: str = "__snapshots__",
    update_snapshots: bool = False,
    name_filter: str | None = None,
    fail_fast: bool = False,
    cassette: Cassette | None = None,
) -> RunResult:
    """Run every (optionally filtered) case in ``suite`` and return a :class:`RunResult`."""
    if cassette is None:
        cassette = Cassette.load(cassette_path_for(suite, cassette_dir))

    prov, eff_name = resolve_provider(suite, provider, record_mode, cassette, offline)

    cases = suite.cases
    if name_filter:
        needle = name_filter.lower()
        cases = [
            c for c in cases if needle in c.name.lower() or any(needle in t.lower() for t in c.tags)
        ]

    model = suite.defaults.get("model", cases[0].model if cases else "")
    run = RunResult(suite_name=suite.name, provider=eff_name, model=model, started_at=_now_iso())
    for case in cases:
        result = run_case(
            case, prov, eff_name, snapshot_dir=snapshot_dir, update_snapshots=update_snapshots
        )
        run.case_results.append(result)
        if fail_fast and not result.passed:
            break

    # Persist any newly recorded episodes.
    if isinstance(prov, RecordingProvider) and record_mode != "none":
        cassette.save()

    return run


def build_report(run: RunResult) -> Report:
    return Report(run=run, promptdiff_version=__version__)


__all__ = ["run_suite", "run_case", "resolve_provider", "build_report", "cassette_path_for"]
