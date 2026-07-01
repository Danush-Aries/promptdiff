"""`json_schema` — parse output as JSON and validate against a JSON Schema."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from ..models import AssertionResult
from ._text import extract_json
from .base import Assertion, AssertionConfigError, EvalContext

if TYPE_CHECKING:
    from ..models import AssertionSpec


class JsonSchemaAssertion(Assertion):
    type_name = "json_schema"

    def __init__(
        self, schema: dict[str, Any], negate: bool = False, description: str | None = None
    ) -> None:
        super().__init__(negate=negate, description=description)
        self.schema = schema

    @classmethod
    def from_spec(cls, spec: AssertionSpec) -> JsonSchemaAssertion:
        cfg = spec.config
        schema = cfg.get("schema")
        if schema is None and cfg.get("schema_file"):
            path = str(cfg["schema_file"])
            if not os.path.exists(path):
                raise AssertionConfigError(f"json_schema: schema_file not found: {path}")
            with open(path, encoding="utf-8") as fh:
                schema = json.load(fh)
        if not isinstance(schema, dict):
            raise AssertionConfigError("json_schema: 'schema' (dict) or 'schema_file' required")
        return cls(schema=schema, negate=spec.negate, description=spec.description)

    def evaluate(self, output: str, ctx: EvalContext) -> AssertionResult:
        try:
            payload = extract_json(output)
        except ValueError as e:
            return AssertionResult(
                type=self.type_name,
                passed=False,
                message=f"output is not valid JSON: {e}",
                detail={"error": "not_json"},
            )

        import jsonschema

        validator = jsonschema.Draft202012Validator(self.schema)
        errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
        if not errors:
            return AssertionResult(
                type=self.type_name,
                passed=True,
                message="output JSON conforms to schema",
                detail={"payload": payload},
            )
        first = errors[0]
        path = "$" + "".join(f"[{p!r}]" if isinstance(p, str) else f"[{p}]" for p in first.path)
        return AssertionResult(
            type=self.type_name,
            passed=False,
            message=f"schema violation at {path}: {first.message}",
            detail={
                "path": path,
                "error": first.message,
                "all_errors": [e.message for e in errors],
            },
        )
