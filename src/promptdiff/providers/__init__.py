"""Provider registry and factory.

Live providers (openai/anthropic) are constructed lazily by name; their SDKs are only
imported inside ``generate`` so importing promptdiff pulls in zero optional deps.
"""

from __future__ import annotations

from .base import (
    CassetteMiss,
    MissingAPIKey,
    MissingProviderDep,
    Provider,
    ProviderError,
)

_FACTORIES = {}


def _register_live() -> None:
    from .anthropic import AnthropicProvider
    from .openai import OpenAIProvider

    _FACTORIES["openai"] = OpenAIProvider
    _FACTORIES["anthropic"] = AnthropicProvider


_register_live()


def list_providers() -> list[str]:
    return sorted(set(_FACTORIES) | {"replay"})


def get_provider(name: str, model: str = "") -> Provider:
    """Instantiate a live provider by name. ``replay`` is handled by the runner directly."""
    if name not in _FACTORIES:
        raise ProviderError(f"unknown provider {name!r}; known: {', '.join(list_providers())}")
    cls = _FACTORIES[name]
    return cls(model=model) if model else cls()


__all__ = [
    "Provider",
    "ProviderError",
    "MissingProviderDep",
    "MissingAPIKey",
    "CassetteMiss",
    "get_provider",
    "list_providers",
]
