"""Suite loading: turn YAML files into frozen :class:`Suite` / :class:`Case` objects.

A suite file looks like::

    name: greetings
    defaults:
      provider: replay
      model: gpt-4o-mini
      params: { temperature: 0 }
    cases:
      - name: says_hello
        prompt: "Say hello"
        assertions:
          - type: contains
            value: hello

Per-case fields override ``defaults``. ``params`` are shallow-merged (case wins).
"""
from __future__ import annotations

import glob
import os
from typing import Any

import yaml

from .models import AssertionSpec, Case, Suite


class LoaderError(Exception):
    """Raised when a suite file is malformed. Carries a human-readable message."""


def _parse_assertions(raw: Any, case_name: str) -> list[AssertionSpec]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise LoaderError(f"case '{case_name}': 'assertions' must be a list")
    specs: list[AssertionSpec] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise LoaderError(f"case '{case_name}': assertion #{i} must be a mapping")
        atype = item.get("type")
        if not atype:
            raise LoaderError(f"case '{case_name}': assertion #{i} missing 'type'")
        config = {k: v for k, v in item.items()
                  if k not in {"type", "negate", "weight", "description"}}
        specs.append(
            AssertionSpec(
                type=str(atype),
                config=config,
                negate=bool(item.get("negate", False)),
                weight=float(item.get("weight", 1.0)),
                description=item.get("description"),
            )
        )
    return specs


def _merge_params(defaults: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults.get("params", {}) or {})
    merged.update(case.get("params", {}) or {})
    return merged


def parse_suite(data: dict[str, Any], source_file: str = "") -> Suite:
    """Parse an already-loaded mapping into a :class:`Suite`."""
    if not isinstance(data, dict):
        raise LoaderError(f"{source_file or 'suite'}: top-level document must be a mapping")
    defaults = data.get("defaults", {}) or {}
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise LoaderError(f"{source_file or 'suite'}: 'cases' must be a non-empty list")

    cases: list[Case] = []
    seen: set[str] = set()
    for i, rc in enumerate(raw_cases):
        if not isinstance(rc, dict):
            raise LoaderError(f"{source_file or 'suite'}: case #{i} must be a mapping")
        name = rc.get("name")
        if not name:
            raise LoaderError(f"{source_file or 'suite'}: case #{i} missing 'name'")
        if name in seen:
            raise LoaderError(f"{source_file or 'suite'}: duplicate case name '{name}'")
        seen.add(name)

        provider = rc.get("provider", defaults.get("provider", "replay"))
        model = rc.get("model", defaults.get("model", ""))
        cases.append(
            Case(
                name=str(name),
                provider=str(provider),
                model=str(model),
                prompt=rc.get("prompt"),
                template=rc.get("template"),
                vars=rc.get("vars", {}) or {},
                params=_merge_params(defaults, rc),
                assertions=_parse_assertions(rc.get("assertions"), str(name)),
                tags=list(rc.get("tags", []) or []),
                source_file=source_file or None,
                line=None,
            )
        )

    suite_name = data.get("name") or (
        os.path.splitext(os.path.basename(source_file))[0] if source_file else "suite"
    )
    return Suite(name=str(suite_name), cases=cases, defaults=defaults, source_file=source_file)


def load_suite_file(path: str) -> Suite:
    """Load a single YAML suite file into a :class:`Suite`."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as e:
        raise LoaderError(f"suite file not found: {path}") from e
    except yaml.YAMLError as e:
        raise LoaderError(f"invalid YAML in {path}: {e}") from e
    return parse_suite(data, source_file=path)


def discover_suites(paths: list[str]) -> list[Suite]:
    """Expand paths (files, directories, globs) into loaded suites, sorted by path."""
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "**", "*.yaml"), recursive=True))
            files.extend(glob.glob(os.path.join(p, "**", "*.yml"), recursive=True))
        elif any(ch in p for ch in "*?["):
            files.extend(glob.glob(p, recursive=True))
        else:
            files.append(p)
    seen: set[str] = set()
    ordered = [f for f in sorted(files) if not (f in seen or seen.add(f))]
    return [load_suite_file(f) for f in ordered]
