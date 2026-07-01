"""Anthropic provider — lazy SDK import so the package installs with zero heavy deps."""

from __future__ import annotations

import os
import time
from typing import Any

from ..models import Completion
from .base import MissingAPIKey, MissingProviderDep, Provider


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-haiku-latest") -> None:
        self.model = model

    def generate(self, prompt: str, params: dict[str, Any]) -> Completion:  # pragma: no cover
        try:
            import anthropic
        except ImportError as e:
            raise MissingProviderDep(
                "anthropic SDK not installed — `pip install promptdiff[llm]`"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise MissingAPIKey("ANTHROPIC_API_KEY not set (only needed for `record`)")

        model = params.get("_model", self.model)
        client = anthropic.Anthropic()
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=model,
            max_tokens=params.get("max_tokens", 1024),
            temperature=params.get("temperature", 1.0),
            system=params.get("system", ""),
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.perf_counter() - t0) * 1000
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        usage = (
            {
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
            }
            if resp.usage
            else {}
        )
        return Completion(
            text=text,
            model=model,
            provider=self.name,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
            usage=usage,
            latency_ms=latency,
        )
