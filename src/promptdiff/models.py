"""Shared data contract for promptdiff.

Every module — loader, assertions, providers, runner, reporting, diff, cli — codes
against these types. Kept dependency-free (stdlib only) so it stays the stable center
of the package. Inputs are frozen; only accumulator types (RunResult, Cassette) are
mutable while a run is in flight.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

REPORT_SCHEMA_VERSION = 1
CASSETTE_VERSION = 1

# Params that are allowed to influence the cassette hash. Everything else (client
# timeouts, streaming flags, api keys, etc.) is stripped before hashing so that
# record/replay stays stable across environments.
CANONICAL_PARAM_ALLOWLIST = frozenset(
    {
        "temperature",
        "max_tokens",
        "top_p",
        "seed",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "n",
        "system",
    }
)


# ---- Inputs (parsed from YAML/Python) -------------------------------------


@dataclass(frozen=True)
class AssertionSpec:
    type: str  # "contains" | "regex" | "json_schema" | ...
    config: dict[str, Any] = field(default_factory=dict)
    negate: bool = False  # generic negation (powers not_contains)
    weight: float = 1.0
    description: str | None = None


@dataclass(frozen=True)
class Case:
    name: str  # unique within a suite; test id + cassette key part
    provider: str  # "openai" | "anthropic" | "replay"
    model: str
    prompt: str | None = None  # literal prompt OR ...
    template: str | None = None  # ... a template string with {var} placeholders
    vars: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    assertions: list[AssertionSpec] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source_file: str | None = None
    line: int | None = None

    @property
    def rendered_prompt(self) -> str:
        """Substitute ``vars`` into ``template`` (or return the literal ``prompt``).

        Raises ``KeyError`` for a missing template variable — surfaced by the runner
        as a failing case, never a crash.
        """
        if self.template is not None:
            return self.template.format(**self.vars)
        if self.prompt is not None:
            return self.prompt
        raise ValueError(f"case '{self.name}' has neither prompt nor template")


@dataclass(frozen=True)
class Suite:
    name: str
    cases: list[Case]
    defaults: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""


# ---- Provider I/O ---------------------------------------------------------


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    provider: str
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Completion:
        return cls(
            text=d["text"],
            model=d.get("model", ""),
            provider=d.get("provider", ""),
            raw=d.get("raw", {}),
            usage=d.get("usage", {}),
            latency_ms=d.get("latency_ms"),
        )


# ---- Assertion results ----------------------------------------------------


@dataclass(frozen=True)
class AssertionResult:
    type: str
    passed: bool
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


# ---- Run results ----------------------------------------------------------


@dataclass(frozen=True)
class CaseResult:
    case_name: str
    passed: bool
    completion: Completion
    assertion_results: list[AssertionResult] = field(default_factory=list)
    error: str | None = None
    duration_ms: float | None = None

    @property
    def failed_assertions(self) -> list[AssertionResult]:
        return [r for r in self.assertion_results if not r.passed]


@dataclass
class RunResult:
    suite_name: str
    provider: str
    model: str
    started_at: str
    case_results: list[CaseResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.case_results)

    @property
    def counts(self) -> dict[str, int]:
        passed = sum(1 for c in self.case_results if c.passed)
        errors = sum(1 for c in self.case_results if c.error)
        failed = sum(1 for c in self.case_results if not c.passed and not c.error)
        return {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total": len(self.case_results),
        }


@dataclass(frozen=True)
class Report:
    """Serializable summary of a RunResult, with a schema version for forward-compat."""

    run: RunResult
    schema_version: int = REPORT_SCHEMA_VERSION
    promptdiff_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "promptdiff_version": self.promptdiff_version,
            "run": {
                "suite_name": self.run.suite_name,
                "provider": self.run.provider,
                "model": self.run.model,
                "started_at": self.run.started_at,
                "counts": self.run.counts,
                "passed": self.run.passed,
                "case_results": [
                    {
                        "case_name": c.case_name,
                        "passed": c.passed,
                        "error": c.error,
                        "duration_ms": c.duration_ms,
                        "completion": c.completion.to_dict(),
                        "assertion_results": [asdict(a) for a in c.assertion_results],
                    }
                    for c in self.run.case_results
                ],
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> Report:
        d = json.loads(s)
        run_d = d["run"]
        case_results = [
            CaseResult(
                case_name=c["case_name"],
                passed=c["passed"],
                completion=Completion.from_dict(c["completion"]),
                assertion_results=[AssertionResult(**a) for a in c["assertion_results"]],
                error=c.get("error"),
                duration_ms=c.get("duration_ms"),
            )
            for c in run_d["case_results"]
        ]
        run = RunResult(
            suite_name=run_d["suite_name"],
            provider=run_d["provider"],
            model=run_d["model"],
            started_at=run_d["started_at"],
            case_results=case_results,
        )
        return cls(
            run=run,
            schema_version=d.get("schema_version", REPORT_SCHEMA_VERSION),
            promptdiff_version=d.get("promptdiff_version", ""),
        )


# ---- Cassette (record/replay fixture) -------------------------------------


def cassette_key(provider: str, model: str, prompt: str, params: dict[str, Any]) -> str:
    """Deterministic sha256 over the canonical (provider, model, prompt, allowed params).

    Volatile params are stripped via CANONICAL_PARAM_ALLOWLIST so a reordered or
    environment-specific param never invalidates a committed fixture.
    """
    canonical = {
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "params": {k: params[k] for k in sorted(params) if k in CANONICAL_PARAM_ALLOWLIST},
    }
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CassetteEntry:
    key: str
    provider: str
    model: str
    request: dict[str, Any]
    response: Completion
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "provider": self.provider,
            "model": self.model,
            "request": self.request,
            "response": self.response.to_dict(),
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CassetteEntry:
        return cls(
            key=d["key"],
            provider=d["provider"],
            model=d["model"],
            request=d["request"],
            response=Completion.from_dict(d["response"]),
            recorded_at=d["recorded_at"],
        )


@dataclass
class Cassette:
    path: str
    version: int = CASSETTE_VERSION
    entries: dict[str, CassetteEntry] = field(default_factory=dict)

    def get(self, key: str) -> CassetteEntry | None:
        return self.entries.get(key)

    def put(self, entry: CassetteEntry) -> None:
        self.entries[entry.key] = entry

    @classmethod
    def load(cls, path: str) -> Cassette:
        import os

        if not os.path.exists(path):
            return cls(path=path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        entries = {k: CassetteEntry.from_dict(v) for k, v in data.get("entries", {}).items()}
        return cls(path=path, version=data.get("version", CASSETTE_VERSION), entries=entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "entries": {k: e.to_dict() for k, e in sorted(self.entries.items())},
        }

    def serialize(self) -> str:
        """Stable, diffable JSON (sorted keys, pretty) so cassettes review cleanly in PRs."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def save(self) -> None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(self.serialize())
