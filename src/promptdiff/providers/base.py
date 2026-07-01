"""Provider ABC and provider-related errors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import Completion


class ProviderError(Exception):
    """Base class for provider failures."""


class MissingProviderDep(ProviderError):
    """The optional SDK for a live provider is not installed."""


class MissingAPIKey(ProviderError):
    """A live provider was used without its API key set in the environment."""


class CassetteMiss(ProviderError):
    """A replay request had no recorded entry — never falls back to the network."""


class Provider(ABC):
    """A text-completion provider. Concrete providers own their SDK (lazy-imported)."""

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, params: dict[str, Any]) -> Completion:
        """Return a :class:`Completion` for ``prompt``. May raise :class:`ProviderError`."""
