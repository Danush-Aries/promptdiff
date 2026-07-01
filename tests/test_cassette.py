"""Cassette / provider tests: determinism, replay, miss-not-network, roundtrip."""
from __future__ import annotations

from typing import Any

import pytest

from promptdiff.models import Cassette, Completion, cassette_key
from promptdiff.providers.base import CassetteMiss, Provider
from promptdiff.providers.cassette import RecordingProvider, make_entry
from promptdiff.providers.replay import ReplayProvider


class StubLiveProvider(Provider):
    """A fake 'live' provider — proves record/replay never touches a real network."""
    name = "openai"

    def __init__(self, text="stub-response", model="gpt-4o-mini"):
        self.text = text
        self.model = model
        self.calls = 0

    def generate(self, prompt: str, params: dict[str, Any]) -> Completion:
        self.calls += 1
        return Completion(text=self.text, model=self.model, provider="openai")


def test_cassette_key_is_deterministic():
    p = {"temperature": 0, "max_tokens": 50}
    k1 = cassette_key("openai", "gpt-4o-mini", "hello", p)
    k2 = cassette_key("openai", "gpt-4o-mini", "hello", dict(reversed(list(p.items()))))
    assert k1 == k2
    # volatile / non-allowlisted params do not change the key
    k3 = cassette_key("openai", "gpt-4o-mini", "hello", {**p, "_client_timeout": 999})
    assert k1 == k3
    assert len(k1) == 64


def test_replay_returns_recorded_completion(tmp_path):
    cas = Cassette(path=str(tmp_path / "c.json"))
    cas.put(make_entry("openai", "gpt-4o-mini", "hi", {"temperature": 0},
                       Completion(text="recorded!", model="gpt-4o-mini", provider="openai")))
    rp = ReplayProvider(cas)
    out = rp.generate("hi", {"temperature": 0, "_provider": "openai", "_model": "gpt-4o-mini"})
    assert out.text == "recorded!"


def test_replay_miss_raises_not_network(tmp_path):
    cas = Cassette(path=str(tmp_path / "empty.json"))
    rp = ReplayProvider(cas)
    with pytest.raises(CassetteMiss):
        rp.generate("never recorded", {"_provider": "openai", "_model": "gpt-4o-mini"})


def test_recording_roundtrip(tmp_path):
    path = str(tmp_path / "rec.json")
    cas = Cassette(path=path)
    live = StubLiveProvider(text="from-live")
    rec = RecordingProvider(live, cas, mode="once")

    params = {"temperature": 0, "_model": "gpt-4o-mini", "_provider": "openai"}
    first = rec.generate("q", params)
    assert first.text == "from-live"
    assert live.calls == 1

    # second call replays from cassette — no new live call
    second = rec.generate("q", params)
    assert second.text == "from-live"
    assert live.calls == 1

    cas.save()
    reloaded = Cassette.load(path)
    rp = ReplayProvider(reloaded)
    replayed = rp.generate("q", params)
    assert replayed.text == "from-live"


def test_cassette_is_stable_diffable(tmp_path):
    path = str(tmp_path / "s.json")
    cas = Cassette(path=path)
    cas.put(make_entry("openai", "m", "p", {"temperature": 0},
                       Completion(text="t", model="m", provider="openai")))
    a = cas.serialize()
    # saving twice is byte-identical (sorted keys, pretty JSON)
    cas.save()
    with open(path, encoding="utf-8") as fh:
        first = fh.read()
    cas.save()
    with open(path, encoding="utf-8") as fh:
        second = fh.read()
    assert first == second == a
    # and a reload -> reserialize is stable too
    assert Cassette.load(path).serialize() == a
