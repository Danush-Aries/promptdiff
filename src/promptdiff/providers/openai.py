"""OpenAI provider — lazy SDK import so the package installs with zero heavy deps."""

from __future__ import annotations

import os
import time
from typing import Any

from ..models import Completion
from .base import MissingAPIKey, MissingProviderDep, Provider

_CHAT_PARAMS = {
    "temperature",
    "max_tokens",
    "top_p",
    "seed",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "n",
}


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def generate(self, prompt: str, params: dict[str, Any]) -> Completion:  # pragma: no cover
        try:
            from openai import OpenAI
        except ImportError as e:
            raise MissingProviderDep(
                "openai SDK not installed — `pip install promptdiff[llm]`"
            ) from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingAPIKey("OPENAI_API_KEY not set (only needed for `record`)")

        model = params.get("_model", self.model)
        call = {k: v for k, v in params.items() if k in _CHAT_PARAMS}
        system = params.get("system")
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        client = OpenAI()
        t0 = time.perf_counter()
        resp = client.chat.completions.create(model=model, messages=messages, **call)
        latency = (time.perf_counter() - t0) * 1000
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
        return Completion(
            text=resp.choices[0].message.content or "",
            model=model,
            provider=self.name,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
            usage=usage,
            latency_ms=latency,
        )
