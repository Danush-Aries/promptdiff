"""pytest plugin — collect promptdiff YAML suites as native pytest items.

Files matching ``*.promptdiff.yaml`` (or ``*.promptdiff.yml``) are collected; each
:class:`Case` becomes one test item with id ``<file>::<case_name>``. Items run the same
runner path with the offline :class:`ReplayProvider`, so ``pytest -k``, ``-x``, ``-q``,
xdist, and CI reporters all work with zero network.

Also exposes a ``@case`` decorator for defining cases inline in Python test files.
"""
from __future__ import annotations

import os
from typing import Any

import pytest

from .assertions import EvalContext, build
from .loader import LoaderError, load_suite_file
from .models import Cassette
from .runner import cassette_path_for, run_case
from .runner import resolve_provider as _resolve_provider

_SUFFIXES = (".promptdiff.yaml", ".promptdiff.yml")


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("promptdiff")
    group.addoption("--promptdiff-update-snapshots", action="store_true", default=False,
                    help="rewrite promptdiff snapshots instead of failing on drift")
    parser.addini("promptdiff_cassette_dir", "cassette dir for promptdiff", default="tests/cassettes")


class PromptAssertionError(AssertionError):
    """Raised when a promptdiff case fails inside pytest (clean repr_failure)."""


def pytest_collect_file(parent: Any, file_path: Any):  # pytest>=7 passes pathlib
    name = os.path.basename(str(file_path))
    if any(name.endswith(sfx) for sfx in _SUFFIXES):
        return PromptFile.from_parent(parent, path=file_path)
    return None


class PromptFile(pytest.File):
    def collect(self):
        try:
            suite = load_suite_file(str(self.path))
        except LoaderError as e:
            raise pytest.UsageError(f"promptdiff: {e}") from e
        cassette_dir = self.config.getini("promptdiff_cassette_dir")
        cassette = Cassette.load(cassette_path_for(suite, cassette_dir))
        update = self.config.getoption("--promptdiff-update-snapshots")
        for case in suite.cases:
            yield PromptItem.from_parent(
                self, name=case.name, case=case, suite=suite,
                cassette=cassette, update_snapshots=update,
            )


class PromptItem(pytest.Item):
    def __init__(self, *, case, suite, cassette, update_snapshots, **kwargs):
        super().__init__(**kwargs)
        self.case = case
        self.suite = suite
        self.cassette = cassette
        self.update_snapshots = update_snapshots

    def runtest(self) -> None:
        # Strictly hermetic in pytest: force replay, no recording.
        provider, eff = _resolve_provider(
            self.suite, "replay", "none", self.cassette, offline=True
        )
        snapshot_dir = os.path.join(os.path.dirname(str(self.path)) or ".", "__snapshots__")
        result = run_case(self.case, provider, eff, snapshot_dir=snapshot_dir,
                          update_snapshots=self.update_snapshots)
        if not result.passed:
            if result.error:
                raise PromptAssertionError(result.error)
            fails = "\n".join(f"  - {a.type}: {a.message}" for a in result.failed_assertions)
            raise PromptAssertionError(f"case '{self.case.name}' failed:\n{fails}")

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, PromptAssertionError):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, 0, f"promptdiff: {self.case.name}"


# ---- Python-defined cases -------------------------------------------------

_REGISTERED_CASES: list[Any] = []


def case(name: str, provider: str = "replay", model: str = "", **kwargs):
    """Decorator/registrar for defining a :class:`Case` in a Python test module.

    Usage::

        @promptdiff.case("says_hi", prompt="Say hi", assertions=[...])
        def _():
            ...
    """
    from .models import AssertionSpec, Case

    specs = []
    for a in kwargs.pop("assertions", []) or []:
        if isinstance(a, AssertionSpec):
            specs.append(a)
        else:
            specs.append(AssertionSpec(type=a["type"],
                                       config={k: v for k, v in a.items() if k != "type"}))
    the_case = Case(name=name, provider=provider, model=model,
                    prompt=kwargs.get("prompt"), template=kwargs.get("template"),
                    vars=kwargs.get("vars", {}), params=kwargs.get("params", {}),
                    assertions=specs)

    def _decorator(fn):
        _REGISTERED_CASES.append(the_case)
        return fn

    return _decorator


def run_case_offline(the_case, cassette: Cassette, snapshot_dir: str = "__snapshots__"):
    """Helper for Python tests: run one case against a cassette, return CaseResult."""
    from .providers.replay import ReplayProvider

    provider = ReplayProvider(cassette)
    return run_case(the_case, provider, "replay", snapshot_dir=snapshot_dir)


__all__ = ["case", "PromptFile", "PromptItem", "PromptAssertionError", "EvalContext", "build"]
