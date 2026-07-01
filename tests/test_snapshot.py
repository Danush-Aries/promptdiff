"""snapshot assertion: record-once, compare-forever, diff-on-drift, update flag."""
from __future__ import annotations

from promptdiff import assertions
from promptdiff.assertions import EvalContext
from promptdiff.models import AssertionSpec, Case


def _snap(negate=False):
    return assertions.build(AssertionSpec(type="snapshot", config={}, negate=negate))


def _ctx(tmp_path, update=False):
    case = Case(name="c1", provider="replay", model="m", prompt="p",
                source_file="mysuite.yaml")
    return EvalContext(case=case, snapshot_dir=str(tmp_path / "__snapshots__"),
                       update_snapshots=update)


def test_snapshot_first_run_records(tmp_path):
    ctx = _ctx(tmp_path)
    res = _snap().run("first output", ctx)
    assert res.passed
    assert res.detail["status"] == "new"
    assert (tmp_path / "__snapshots__" / "mysuite" / "c1.txt").exists()


def test_snapshot_match_passes(tmp_path):
    ctx = _ctx(tmp_path)
    _snap().run("stable output", ctx)          # records
    res = _snap().run("stable output", ctx)    # compares
    assert res.passed
    assert res.detail["status"] == "match"


def test_snapshot_drift_fails_with_diff(tmp_path):
    ctx = _ctx(tmp_path)
    _snap().run("original line\n", ctx)
    res = _snap().run("changed line\n", ctx)
    assert not res.passed
    assert res.detail["status"] == "drift"
    assert "changed line" in res.detail["diff"]
    assert "original line" in res.detail["diff"]


def test_snapshot_update_flag_rewrites(tmp_path):
    ctx = _ctx(tmp_path)
    _snap().run("v1", ctx)
    upd = _ctx(tmp_path, update=True)
    res = _snap().run("v2", upd)
    assert res.passed
    # subsequent normal compare now matches the updated snapshot
    res2 = _snap().run("v2", ctx)
    assert res2.passed and res2.detail["status"] == "match"
