"""json_schema assertion + JSON extractor tests (real schema, real jsonschema lib)."""
from __future__ import annotations

from promptdiff import assertions
from promptdiff.assertions import EvalContext
from promptdiff.assertions._text import extract_json
from promptdiff.models import AssertionSpec

CTX = EvalContext()

_SCHEMA = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
    },
}


def _schema_assertion():
    return assertions.build(AssertionSpec(type="json_schema", config={"schema": _SCHEMA}))


def test_json_schema_valid_case():
    a = _schema_assertion()
    res = a.run('{"name": "Ada", "age": 36}', CTX)
    assert res.passed


def test_json_schema_type_violation():
    a = _schema_assertion()
    res = a.run('{"name": "Ada", "age": "old"}', CTX)
    assert not res.passed
    assert "age" in res.detail["path"]


def test_json_schema_missing_required():
    a = _schema_assertion()
    res = a.run('{"name": "Ada"}', CTX)
    assert not res.passed


def test_json_schema_non_json_output():
    a = _schema_assertion()
    res = a.run("Sorry, I cannot help with that.", CTX)
    assert not res.passed
    assert res.detail.get("error") == "not_json"


def test_json_extractor_handles_fences_and_prose():
    assert extract_json('{"a": 1}') == {"a": 1}
    fenced = "Here you go:\n```json\n{\"a\": 2}\n```\nthanks"
    assert extract_json(fenced) == {"a": 2}
    prose = 'The answer is {"a": 3} as requested.'
    assert extract_json(prose) == {"a": 3}


def test_json_schema_extracts_from_fenced_output():
    a = _schema_assertion()
    res = a.run('```json\n{"name": "Grace", "age": 45}\n```', CTX)
    assert res.passed
