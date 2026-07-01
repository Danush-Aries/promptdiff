"""Baseline-vs-current diff classification tests."""
from __future__ import annotations

from promptdiff.diff import diff_reports
from promptdiff.models import CaseResult, Completion, Report, RunResult


def _report(statuses: dict[str, bool]) -> Report:
    results = [
        CaseResult(case_name=name, passed=ok,
                   completion=Completion(text="", model="m", provider="replay"))
        for name, ok in statuses.items()
    ]
    run = RunResult(suite_name="s", provider="replay", model="m",
                    started_at="2026-01-01T00:00:00+00:00", case_results=results)
    return Report(run=run, promptdiff_version="0.1.0")


def test_diff_detects_regression():
    base = _report({"a": True, "b": True})
    curr = _report({"a": True, "b": False})
    d = diff_reports(base, curr)
    assert d.has_regressions
    assert [x.case_name for x in d.regressions] == ["b"]


def test_diff_classifies_new_removed_fixed():
    base = _report({"keep": True, "gone": True, "brokethenfixed": False})
    curr = _report({"keep": True, "brokethenfixed": True, "fresh": True})
    d = diff_reports(base, curr)
    counts = d.counts()
    assert counts["fixed"] == 1
    assert counts["new"] == 1
    assert counts["removed"] == 1
    assert counts["regression"] == 0
    assert [x.case_name for x in d.new] == ["fresh"]
    assert [x.case_name for x in d.removed] == ["gone"]
    assert [x.case_name for x in d.fixed] == ["brokethenfixed"]


def test_report_json_roundtrip():
    rep = _report({"a": True, "b": False})
    restored = Report.from_json(rep.to_json())
    assert restored.run.counts == rep.run.counts
    assert [c.case_name for c in restored.run.case_results] == ["a", "b"]
