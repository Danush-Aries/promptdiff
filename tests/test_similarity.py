"""semantic_similarity tests using the forced offline token-overlap backend."""
from __future__ import annotations

from promptdiff import assertions
from promptdiff.assertions import EvalContext
from promptdiff.assertions.similarity import SimilarityAssertion, token_overlap
from promptdiff.models import AssertionSpec

CTX = EvalContext()


def _sim(reference, threshold, metric="token_overlap"):
    return assertions.build(AssertionSpec(
        type="semantic_similarity",
        config={"reference": reference, "threshold": threshold, "metric": metric},
    ))


def test_similarity_identical_is_one():
    a = _sim("the cat sat on the mat", 0.9)
    res = a.run("the cat sat on the mat", CTX)
    assert res.detail["score"] == 1.0
    assert res.passed


def test_similarity_disjoint_is_low():
    a = _sim("alpha beta gamma", 0.5)
    res = a.run("xylophone quartz jigsaw", CTX)
    assert res.detail["score"] < 0.1
    assert not res.passed


def test_similarity_threshold_boundary():
    # 2 shared of 3-union style overlap -> tune to land exactly at threshold.
    score = token_overlap("red green blue", "red green yellow")
    # tokens: {red,green,blue} vs {red,green,yellow} -> inter 2 union 4 -> 0.5
    assert abs(score - 0.5) < 1e-9
    at_boundary = _sim("red green yellow", 0.5)
    assert at_boundary.run("red green blue", CTX).passed          # 0.5 >= 0.5
    above = _sim("red green yellow", 0.6)
    assert not above.run("red green blue", CTX).passed            # 0.5 < 0.6


def test_similarity_default_metric_is_offline_backend():
    a = _sim("hello world", 0.5)
    assert isinstance(a, SimilarityAssertion)
    assert a.run("hello world", CTX).detail["backend"] == "token_overlap"


def test_similarity_custom_backend_seam_is_mockable():
    a = SimilarityAssertion(reference="x", threshold=0.5, backend=lambda o, r: 0.99)
    assert a.run("anything", CTX).passed
