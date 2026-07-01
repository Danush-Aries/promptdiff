"""Loader, template rendering, and runner aggregation tests."""
from __future__ import annotations

from typing import Any

import pytest

from promptdiff.loader import LoaderError, parse_suite
from promptdiff.models import Case, Cassette, Completion, Suite, cassette_key
from promptdiff.providers.base import Provider
from promptdiff.providers.cassette import make_entry
from promptdiff.runner import run_case, run_suite

_SUITE_DOC = {
    "name": "demo",
    "defaults": {"provider": "replay", "model": "gpt-4o-mini", "params": {"temperature": 0}},
    "cases": [
        {"name": "a", "prompt": "hello", "assertions": [{"type": "contains", "value": "hi"}]},
        {"name": "b", "template": "hi {who}", "vars": {"who": "world"},
         "params": {"max_tokens": 10},
         "assertions": [{"type": "contains", "value": "x"}]},
    ],
}


def test_loader_yaml_to_cases():
    suite = parse_suite(_SUITE_DOC, source_file="demo.yaml")
    assert isinstance(suite, Suite)
    assert [c.name for c in suite.cases] == ["a", "b"]
    a = suite.cases[0]
    assert a.provider == "replay" and a.model == "gpt-4o-mini"
    assert a.params == {"temperature": 0}          # defaults merged in
    assert a.assertions[0].type == "contains"
    b = suite.cases[1]
    assert b.params == {"temperature": 0, "max_tokens": 10}   # case params override/merge


def test_loader_rejects_duplicate_and_missing():
    with pytest.raises(LoaderError):
        parse_suite({"cases": [{"name": "x"}, {"name": "x"}]})
    with pytest.raises(LoaderError):
        parse_suite({"cases": [{"prompt": "no name"}]})
    with pytest.raises(LoaderError):
        parse_suite({"name": "e", "cases": []})


def test_template_rendering_with_vars():
    c = Case(name="t", provider="replay", model="m", template="hi {who}", vars={"who": "sky"})
    assert c.rendered_prompt == "hi sky"


def test_template_missing_var_is_failure_not_crash():
    c = Case(name="t", provider="replay", model="m", template="hi {missing}", vars={})
    cas = Cassette(path="x.json")

    class _P(Provider):
        name = "replay"

        def generate(self, prompt, params):  # pragma: no cover - never reached
            return Completion(text="", model="m", provider="replay")

    res = run_case(c, _P(), "replay")
    assert not res.passed
    assert "render error" in res.error


class DictProvider(Provider):
    """Serves canned text per prompt; unknown prompt -> raises to prove error handling."""
    name = "replay"

    def __init__(self, table: dict[str, str]):
        self.table = table

    def generate(self, prompt: str, params: dict[str, Any]) -> Completion:
        if prompt not in self.table:
            raise RuntimeError(f"boom for {prompt!r}")
        return Completion(text=self.table[prompt], model="m", provider="replay")


def test_runner_aggregates_pass_fail():
    doc = {
        "name": "mix", "defaults": {"provider": "replay", "model": "m"},
        "cases": [
            {"name": "pass", "prompt": "p1", "assertions": [{"type": "contains", "value": "ok"}]},
            {"name": "fail", "prompt": "p2", "assertions": [{"type": "contains", "value": "zzz"}]},
        ],
    }
    suite = parse_suite(doc, "mix.yaml")
    cas = Cassette(path="mix.json")
    cas.put(make_entry("replay", "m", "p1", {}, Completion("ok here", "m", "replay")))
    cas.put(make_entry("replay", "m", "p2", {}, Completion("nope", "m", "replay")))
    run = run_suite(suite, cassette=cas, offline=True)
    assert run.counts == {"passed": 1, "failed": 1, "errors": 0, "total": 2}
    assert not run.passed


def test_runner_provider_error_is_failure_not_crash():
    doc = {"name": "err", "defaults": {"provider": "replay", "model": "m"},
           "cases": [{"name": "boom", "prompt": "unknown",
                      "assertions": [{"type": "contains", "value": "x"}]}]}
    suite = parse_suite(doc, "err.yaml")
    # empty cassette -> ReplayProvider raises CassetteMiss, captured as an error result
    run = run_suite(suite, cassette=Cassette(path="err.json"), offline=True)
    assert run.counts["errors"] == 1
    assert not run.passed
    assert "CassetteMiss" in run.case_results[0].error


def test_name_filter_by_tag_and_name():
    doc = {
        "name": "f", "defaults": {"provider": "replay", "model": "m"},
        "cases": [
            {"name": "geo_q", "prompt": "p", "tags": ["geo"],
             "assertions": [{"type": "contains", "value": "ok"}]},
            {"name": "math_q", "prompt": "p2", "tags": ["math"],
             "assertions": [{"type": "contains", "value": "ok"}]},
        ],
    }
    suite = parse_suite(doc, "f.yaml")
    cas = Cassette(path="f.json")
    cas.put(make_entry("replay", "m", "p", {}, Completion("ok", "m", "replay")))
    cas.put(make_entry("replay", "m", "p2", {}, Completion("ok", "m", "replay")))
    run = run_suite(suite, cassette=cas, offline=True, name_filter="geo")
    assert [c.case_name for c in run.case_results] == ["geo_q"]


def test_replay_key_matches_recorded_openai_key():
    # end-to-end key agreement: recorded under openai, replayed via runner injection
    prompt = "What is 2+2?"
    params = {"temperature": 0}
    k = cassette_key("openai", "gpt-4o-mini", prompt, params)
    cas = Cassette(path="k.json")
    cas.put(make_entry("openai", "gpt-4o-mini", prompt, params,
                       Completion("4", "gpt-4o-mini", "openai")))
    assert k in cas.entries
    doc = {"name": "k", "defaults": {"provider": "openai", "model": "gpt-4o-mini",
                                     "params": {"temperature": 0}},
           "cases": [{"name": "math", "prompt": prompt,
                      "assertions": [{"type": "contains", "value": "4"}]}]}
    suite = parse_suite(doc, "k.yaml")
    run = run_suite(suite, cassette=cas, offline=True)
    assert run.passed
