"""Assertion unit tests against known literal strings (no provider, no network)."""
from __future__ import annotations

import pytest

from promptdiff import assertions
from promptdiff.assertions import AssertionConfigError, EvalContext
from promptdiff.models import AssertionSpec

CTX = EvalContext()


def _build(type_, **cfg):
    negate = cfg.pop("negate", False)
    return assertions.build(AssertionSpec(type=type_, config=cfg, negate=negate))


def test_contains_pass_and_fail():
    a = _build("contains", value="sky")
    assert a.run("the sky is blue", CTX).passed
    b = _build("contains", value="green")
    assert not b.run("the sky is blue", CTX).passed


def test_contains_case_insensitive_and_list_all_any():
    ci = _build("contains", value="SKY", case_sensitive=False)
    assert ci.run("the sky is blue", CTX).passed

    all_ = _build("contains", value=["sky", "blue"], all=True)
    assert all_.run("the sky is blue", CTX).passed
    all_fail = _build("contains", value=["sky", "red"], all=True)
    assert not all_fail.run("the sky is blue", CTX).passed

    any_ = _build("contains", value=["red", "blue"], any=True)
    assert any_.run("the sky is blue", CTX).passed
    any_fail = _build("contains", value=["red", "green"], any=True)
    assert not any_fail.run("the sky is blue", CTX).passed


def test_not_contains():
    ok = _build("not_contains", value="password")
    assert ok.run("nothing secret here", CTX).passed
    bad = _build("not_contains", value="secret")
    assert not bad.run("nothing secret here", CTX).passed


def test_regex_search_vs_fullmatch():
    search = _build("regex", pattern=r"\d{3}")
    assert search.run("order 123 shipped", CTX).passed
    full = _build("regex", pattern=r"\d{3}", full_match=True)
    assert not full.run("order 123 shipped", CTX).passed
    full_ok = _build("regex", pattern=r"\d{3}", full_match=True)
    assert full_ok.run("123", CTX).passed


def test_regex_flags_ignorecase():
    a = _build("regex", pattern=r"hello", flags="i")
    assert a.run("HELLO world", CTX).passed


def test_regex_invalid_pattern_errors_clearly():
    with pytest.raises(AssertionConfigError) as exc:
        _build("regex", pattern="([")
    assert "invalid pattern" in str(exc.value)


def test_exact_strip_behavior():
    stripped = _build("exact", value="hello", strip=True)
    assert stripped.run("  hello\n", CTX).passed
    strict = _build("exact", value="hello", strip=False)
    assert not strict.run("  hello\n", CTX).passed
    assert strict.run("hello", CTX).passed


def test_negate_wrapper_on_exact():
    a = _build("exact", value="yes", negate=True)
    assert a.run("no", CTX).passed
    assert not a.run("yes", CTX).passed
