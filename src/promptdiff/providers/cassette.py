"""Cassette record/replay engine.

Provides :class:`RecordingProvider`, a VCR-style wrapper that calls a real provider,
stores the response into a :class:`Cassette`, and thereafter behaves like replay. The
record modes mirror VCR: ``none`` | ``once`` | ``new_episodes`` | ``all``.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any

from ..models import Cassette, CassetteEntry, Completion, cassette_key
from .base import CassetteMiss, Provider

RECORD_MODES = ("none", "once", "new_episodes", "all")

# Keys redacted from a stored ``raw`` blob so secrets never land in a committed cassette.
_REDACT_KEYS = frozenset({"api_key", "authorization", "auth", "token", "openai_api_key",
                          "anthropic_api_key"})


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def redact(raw: dict[str, Any]) -> dict[str, Any]:
    """Recursively blank out secret-looking keys in a provider ``raw`` blob."""
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k.lower() in _REDACT_KEYS:
            out[k] = "***REDACTED***"
        elif isinstance(v, dict):
            out[k] = redact(v)
        else:
            out[k] = v
    return out


class RecordingProvider(Provider):
    """Wrap a live provider; record its outputs into a cassette, then replay them.

    ``mode`` semantics:
      * ``none``          — never touch the cassette; always call the live provider.
      * ``once``          — replay if an entry exists, else record it.
      * ``new_episodes``  — replay existing entries, record misses.
      * ``all``           — always call live and (over)write the entry.
    """

    name = "recording"

    def __init__(self, inner: Provider, cassette: Cassette, mode: str = "once",
                 store_raw: bool = True) -> None:
        if mode not in RECORD_MODES:
            raise ValueError(f"record mode must be one of {RECORD_MODES}, got {mode!r}")
        self.inner = inner
        self.cassette = cassette
        self.mode = mode
        self.store_raw = store_raw

    def generate(self, prompt: str, params: dict[str, Any]) -> Completion:
        model = params.get("_model", getattr(self.inner, "model", ""))
        key = cassette_key(self.inner.name, model, prompt, params)
        existing = self.cassette.get(key)

        if self.mode == "none":
            return self.inner.generate(prompt, params)

        if existing is not None and self.mode in {"once", "new_episodes"}:
            return existing.response

        # Cache miss (or mode == "all"): call live and record.
        completion = self.inner.generate(prompt, params)
        stored = completion
        if not self.store_raw:
            stored = Completion(text=completion.text, model=completion.model,
                                provider=completion.provider, usage=completion.usage,
                                latency_ms=completion.latency_ms)
        elif completion.raw:
            stored = Completion(text=completion.text, model=completion.model,
                                provider=completion.provider, raw=redact(completion.raw),
                                usage=completion.usage, latency_ms=completion.latency_ms)
        canonical = {k: v for k, v in params.items() if not k.startswith("_")}
        entry = CassetteEntry(
            key=key,
            provider=self.inner.name,
            model=model,
            request={"prompt": prompt, "params": canonical},
            response=stored,
            recorded_at=_now(),
        )
        self.cassette.put(entry)
        return stored


def make_entry(provider: str, model: str, prompt: str, params: dict[str, Any],
               completion: Completion) -> CassetteEntry:
    """Build a :class:`CassetteEntry` (used by tests and manual recording)."""
    canonical = {k: v for k, v in params.items() if not k.startswith("_")}
    return CassetteEntry(
        key=cassette_key(provider, model, prompt, params),
        provider=provider,
        model=model,
        request={"prompt": prompt, "params": canonical},
        response=completion,
        recorded_at=_now(),
    )


__all__ = ["RecordingProvider", "RECORD_MODES", "redact", "make_entry", "CassetteMiss"]
