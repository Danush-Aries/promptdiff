"""ReplayProvider — serves recorded completions from a cassette, never the network."""
from __future__ import annotations

from typing import Any

from ..models import Cassette, Completion, cassette_key
from .base import CassetteMiss, Provider


class ReplayProvider(Provider):
    """Default CI provider. Hashes the request and returns the recorded completion.

    A cache miss raises :class:`CassetteMiss` — it never silently calls a live API.
    """

    name = "replay"

    def __init__(self, cassette: Cassette, model: str = "") -> None:
        self.cassette = cassette
        self.model = model

    def generate(self, prompt: str, params: dict[str, Any]) -> Completion:
        # The runner injects the *original* provider/model so the replay key matches
        # the key used at record time (see cassette_key). Both are stripped from the
        # canonical hash themselves, so they never perturb it.
        model = params.get("_model", self.model)
        provider = params.get("_provider", "replay")
        key = cassette_key(provider, model, prompt, params)
        entry = self.cassette.get(key)
        if entry is None:
            # Secondary safety: exact prompt match across any recorded entry.
            for e in self.cassette.entries.values():
                if e.request.get("prompt") == prompt:
                    entry = e
                    break
        if entry is None:
            raise CassetteMiss(
                f"no cassette entry for prompt (model={model!r}); "
                f"record it first with `promptdiff record`"
            )
        return entry.response
