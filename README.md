# promptdiff

**pytest for LLM prompts.** A regression-testing harness that catches when a prompt edit or a model swap *silently* regresses your outputs — before it ships.

[![ci](https://github.com/Danush-Aries/promptdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/Danush-Aries/promptdiff/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## The problem

Application code has a compiler, a type checker, and a red build when you break something. An LLM prompt has **none of that**. A one-word change to a system prompt, or a swap from `gpt-4o` to `gpt-4o-mini`, can silently:

- stop producing valid JSON your parser depends on,
- drop a required legal disclaimer,
- change tone/format enough to break a downstream step,
- regress on the exact edge case you fixed last month.

Teams ship these changes blind. **promptdiff is the missing red build.**

## Why it's different

There are great eval tools already. promptdiff is the only one that combines the three things a normal Python team actually wants:

1. **Snapshot / baseline regression as the core primitive** — like Jest snapshots or `syrupy`. Record the expected output once, commit it, and fail the build the moment it drifts. Not "is my model good?" but "**did my change change something?**"
2. **Native pytest ergonomics** — cases are collected and reported by pytest itself. `pytest -k`, `-x`, `-q`, fixtures, xdist, and every CI reporter just work. No new mental model.
3. **Offline-by-default VCR cassettes** — `promptdiff record` captures live responses into deterministic JSON fixtures once; every later `run` is hermetic, free, and CI-safe. **All real, but hermetic** — a fresh clone runs green with no API key and zero spend.

| | promptfoo | OpenAI Evals / deepeval | LangSmith / Braintrust | **promptdiff** |
|---|---|---|---|---|
| Snapshot/baseline regression | partial | no | no | **yes, core** |
| Native pytest | no (Node/YAML) | no | no | **yes** |
| Offline & free in CI | live by default | needs API | SaaS + network | **yes, default** |

## Install

```bash
pip install promptdiff            # core: typer, rich, pyyaml, jsonschema
pip install promptdiff[llm]       # + openai, anthropic (only needed to record)
pip install promptdiff[dev]       # + pytest, ruff, coverage
```

## 60-second quickstart

**1. Write a suite** — `prompts/greetings.promptdiff.yaml`:

```yaml
name: greetings
defaults:
  provider: openai
  model: gpt-4o-mini
  params: { temperature: 0 }
cases:
  - name: capital_of_france
    prompt: "What is the capital of France? One word."
    assertions:
      - type: contains
        value: Paris
        case_sensitive: false
  - name: structured_user
    prompt: "Return a JSON object with name and age."
    assertions:
      - type: json_schema
        schema:
          type: object
          required: [name, age]
          properties:
            name: { type: string }
            age:  { type: integer, minimum: 0 }
```

**2. Record once** (the only step that touches the network / costs money):

```bash
export OPENAI_API_KEY=sk-...
promptdiff record prompts/          # writes tests/cassettes/greetings.json
git add tests/cassettes/            # commit the fixture
```

**3. Run forever, offline and free** — locally and in CI:

```bash
promptdiff run prompts/             # exit 0 all pass · 1 failure · 3 cassette miss
```

Now change a prompt or bump the model, re-run, and get an instant red build if anything regressed.

## Assertion reference

| Type | Config | Checks |
|---|---|---|
| `contains` | `value` (str \| list), `case_sensitive`, `all`/`any` | substring membership |
| `not_contains` | same as `contains` | forbidden text is absent |
| `regex` | `pattern`, `flags` (imsx), `full_match` | `re.search` / `re.fullmatch` |
| `exact` | `value`, `strip` | exact string equality |
| `json_schema` | `schema` (inline) or `schema_file` | output parses as JSON and validates (fenced/prose-tolerant) |
| `semantic_similarity` | `reference`, `threshold`, `metric` | embedding cosine, or offline Jaccard token-overlap fallback |
| `snapshot` | `id`, `update` | compares to a committed `__snapshots__/<suite>/<case>.txt`, unified diff on drift |

Any assertion accepts `negate: true`. `not_contains` is just `contains` + `negate`.

## Snapshot workflow

```yaml
- name: summary_format
  prompt: "Summarize the release notes in three bullets."
  assertions:
    - type: snapshot
```

First run records the output and passes (marked *new*). Every later run compares byte-for-byte and **fails with a unified diff** on drift. Accept an intended change with `promptdiff run --update-snapshots`.

## Cassettes & record modes

Cassettes are small, sorted, pretty JSON — they review cleanly in a PR. Record modes mirror VCR:

| mode | behaviour |
|---|---|
| `none` | live only, never touch cassette (default in pytest — strictly hermetic) |
| `once` | replay if present, else record (default for `run`) |
| `new_episodes` | replay existing, record misses |
| `all` | always re-record |

Secret-looking keys in a provider's raw response are redacted before they hit disk.

## pytest usage

promptdiff registers a pytest plugin. Any `*.promptdiff.yaml` file is collected automatically, one test item per case:

```bash
pytest -q                                   # runs your prompt suites, offline
pytest -k capital_of_france                 # filter by case name
pytest --promptdiff-update-snapshots        # accept snapshot changes
```

You can also define cases in Python:

```python
import promptdiff

@promptdiff.case("says_hi", prompt="Say hi", assertions=[{"type": "contains", "value": "hi"}])
def _(): ...
```

## CI regression gate with `diff`

Emit a JSON report per branch and diff them to fail the build only on *regressions* (a case that flipped pass -> fail):

```bash
promptdiff run prompts/ --report json --out current.json
promptdiff diff baseline.json current.json      # exit 1 if any regression
```

```yaml
# .github/workflows/prompts.yml
- run: pip install promptdiff
- run: promptdiff run prompts/          # green on a fresh fork, no secrets needed
```

## Config (`promptdiff.yaml` / `[tool.promptdiff]`)

```toml
[tool.promptdiff]
cassette_dir = "tests/cassettes"
record_mode = "none"        # hermetic by default under pytest
```

## Exit codes

`0` all pass · `1` assertion failure or regression · `2` config/load error · `3` cassette miss in offline mode.

## Contributing

```bash
pip install -e ".[dev]"
ruff check src && ruff format --check src
pytest -q
```

The whole suite runs offline with no API key. PRs welcome.

## License

MIT © 2026 Dhanush Shankar


---

<p align="center">
  <b>⭐ If this project helps you, star it</b> — stars are how open-source tools get found, and every one directly supports more development.
  <br/><sub>· Found a bug? Open an <a href="https://github.com/Danush-Aries/promptdiff/issues">issue</a> · Want to chat? <a href="https://github.com/Danush-Aries/promptdiff/discussions">Discussions</a> · Contribute? See <a href="https://github.com/Danush-Aries/promptdiff/blob/main/CONTRIBUTING.md">CONTRIBUTING.md</a> ·</sub>
</p>

