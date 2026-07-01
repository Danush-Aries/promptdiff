"""Assertion registry: map a spec ``type`` string to an :class:`Assertion` class.

``not_contains`` is registered as a convenience alias that flips ``negate`` on the
``contains`` implementation, matching the PLAN (not_contains == contains + negate).
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .base import Assertion, AssertionConfigError, EvalContext
from .contains import ContainsAssertion
from .exact import ExactAssertion
from .json_schema import JsonSchemaAssertion
from .regex import RegexAssertion
from .similarity import SimilarityAssertion
from .snapshot import SnapshotAssertion

if TYPE_CHECKING:
    from ..models import AssertionSpec

_REGISTRY: dict[str, type[Assertion]] = {
    ContainsAssertion.type_name: ContainsAssertion,
    RegexAssertion.type_name: RegexAssertion,
    ExactAssertion.type_name: ExactAssertion,
    JsonSchemaAssertion.type_name: JsonSchemaAssertion,
    SimilarityAssertion.type_name: SimilarityAssertion,
    SnapshotAssertion.type_name: SnapshotAssertion,
}

# aliases: name -> (target type, force_negate)
_ALIASES: dict[str, tuple[str, bool]] = {
    "not_contains": ("contains", True),
    "similarity": ("semantic_similarity", False),
}


def list_types() -> list[str]:
    """All registered assertion names (including aliases), sorted."""
    return sorted(set(_REGISTRY) | set(_ALIASES))


def get(type_name: str) -> type[Assertion]:
    """Look up the assertion class for a registry name (resolving aliases)."""
    if type_name in _ALIASES:
        target, _ = _ALIASES[type_name]
        return _REGISTRY[target]
    if type_name not in _REGISTRY:
        raise AssertionConfigError(
            f"unknown assertion type {type_name!r}; known: {', '.join(list_types())}"
        )
    return _REGISTRY[type_name]


def build(spec: AssertionSpec) -> Assertion:
    """Construct an :class:`Assertion` from a parsed spec, applying alias negation."""
    if spec.type in _ALIASES:
        target, force_negate = _ALIASES[spec.type]
        if force_negate:
            spec = replace(spec, type=target, negate=not spec.negate)
        else:
            spec = replace(spec, type=target)
    cls = get(spec.type)
    return cls.from_spec(spec)


__all__ = [
    "Assertion",
    "AssertionConfigError",
    "EvalContext",
    "ContainsAssertion",
    "RegexAssertion",
    "ExactAssertion",
    "JsonSchemaAssertion",
    "SimilarityAssertion",
    "SnapshotAssertion",
    "build",
    "get",
    "list_types",
]
