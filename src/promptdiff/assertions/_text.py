"""Shared text helpers for assertions: JSON extraction and token normalization."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON payload from possibly-chatty model output.

    Handles: raw JSON, ```json fenced blocks, and prose-prefixed JSON (finds the first
    balanced ``{...}`` or ``[...]``). Raises ``ValueError`` if nothing parses.
    """
    stripped = text.strip()
    # 1. Whole thing is JSON.
    try:
        return json.loads(stripped)
    except (ValueError, TypeError):
        pass
    # 2. Fenced code block.
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (ValueError, TypeError):
            pass
    # 3. First balanced object/array substring.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except (ValueError, TypeError):
                        break
    raise ValueError("no valid JSON payload found in output")


def normalize_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization used by the offline similarity backend."""
    return _TOKEN_RE.findall(text.lower())
