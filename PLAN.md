# promptdiff — Implementation PLAN

> "pytest for LLM prompts" — a regression-testing harness that catches when a prompt or model change silently regresses outputs. Python + Typer CLI, also usable as a pytest plugin. All tests run fully offline against recorded cassettes.

---

## 1. Context

### The real, unmet need
Teams ship prompt edits and model swaps (`gpt-4o` → `gpt-4o-mini`, a reworded system prompt, a new few-shot example) with **no safety net**. Unlike application code, an LLM prompt has no compiler, no type checker, and no obvious "this broke" signal. A one-word change to a system prompt can silently:
- stop producing valid JSON,
- drop a required disclaimer,
- change tone/format enough to break a downstream parser,
- regress on edge-case inputs the author never re-checked.

There is currently no ergonomic, git-committable, CI-runnable equivalent of `pytest` for prompts that a normal Python team can adopt in an afternoon.

### Honest differentiation vs existing tools

| Tool | What it is | Gap promptdiff fills |
|---|---|---|
| **promptfoo** | Node/YAML eval runner, matrix testing, web UI | Node-first (friction for Python teams), config-heavy, **no first-class snapshot/baseline regression model**, live calls by default (non-hermetic, costs money in CI) |
| **OpenAI Evals / deepeval / ragas** | Quality/accuracy benchmarking, LLM-graded metrics | Answer "is the model good?" not "**did my change regress vs last week?**"; heavy deps; assume live API access |
| **LangSmith / Braintrust** | Hosted eval + tracing platforms | SaaS, account + network required, not a local `pytest` you own in-repo |

**promptdiff's angle (three things none of the above combine):**
1. **Snapshot / baseline regression as the core primitive** — like Jest snapshots or `syrupy`. Record the expected output once, commit it, fail the build when it drifts. This is the workflow teams actually want: "tell me when my change *changed something*."
2. **pytest-native ergonomics** — cases are collected and reported by pytest itself. No new mental model; `pytest -k`, `-x`, fixtures, CI reporters all just work.
3. **Offline-by-default cassettes (VCR-style record/replay)** — `promptdiff record` captures live provider responses into deterministic fixtures; every subsequent `run` is hermetic, free, and CI-safe. "All real, but hermetic."

The design goal: **a Python team adds promptdiff, records cassettes once, commits them, and gets a red build the moment a prompt/model change silently regresses — with zero API spend in CI.**

---

## 2. Architecture — module breakdown for parallel builders

Package root: `promptdiff/`. Four **non-overlapping** modules, each ownable by one builder subagent. The only shared surface is `models.py` (the data contract), which is built first / frozen early so the other three can work against stable types.

```
promptdiff/
  __init__.py            # version, public re-exports
  models.py              # [SHARED CONTRACT] dataclasses — Builder A (freeze first)
  loader.py              # YAML/Python case discovery + parsing — Builder A
  assertions/            # [MODULE 1] — Builder B
    __init__.py          #   registry: name -> Assertion class
    base.py              #   Assertion ABC + AssertionResult
    contains.py          #   contains / not_contains
    regex.py             #   regex
    exact.py             #   exact
    json_schema.py       #   json-schema-valid
    similarity.py        #   semantic-similarity-threshold (+ offline fallback)
    snapshot.py          #   snapshot-vs-baseline
  providers/             # [MODULE 2] — Builder C
    __init__.py          #   registry + get_provider()
    base.py              #   Provider ABC (generate(prompt, params) -> Completion)
    openai.py            #   OpenAI provider (optional dep, lazy import)
    anthropic.py         #   Anthropic provider (optional dep, lazy import)
    replay.py            #   ReplayProvider — reads cassette, no network
    cassette.py          #   Cassette load/save, key hashing, record/replay logic
  runner.py              # [MODULE 3] — Builder D
    (runner.py)          #   orchestrates: load -> resolve provider -> run cases -> assert
  reporting.py           #   Report -> console (rich), JSON, JUnit XML — Builder D
  cli.py                 # [MODULE 4] — Builder D (shares w/ runner via clean API)
  pytest_plugin.py       #   pytest entry point (collect YAML as test items) — Builder D
  diff.py                # baseline-vs-run comparison logic — Builder D
```

### Builder assignment (4 parallel subagents)

- **Builder A — Contract & Loading** (`models.py`, `loader.py`)
  Owns the frozen dataclasses and turning YAML/Python into `Case` objects. *Must land first*; others import its types. Small, fast, unblocks everyone.

- **Builder B — Assertions** (`assertions/`)
  Owns every assertion type behind a common ABC + registry. Pure functions of `(output_text, case_context) -> AssertionResult`. No provider/network knowledge. Independently unit-testable against literal strings.

- **Builder C — Providers & Cassettes** (`providers/`)
  Owns live providers (lazy optional deps), the `ReplayProvider`, and the cassette record/replay engine (hashing, serialization, determinism). No assertion knowledge.

- **Builder D — Runner, Reporting, CLI, pytest plugin, diff** (`runner.py`, `reporting.py`, `cli.py`, `pytest_plugin.py`, `diff.py`)
  Wires A+B+C together, produces reports, exposes Typer CLI + pytest plugin + `diff`. Depends on the public APIs of A/B/C but not their internals.

**Interface contracts that let them work in parallel (agree up front):**
- `Provider.generate(prompt: str, params: dict) -> Completion` (Builder C).
- `Assertion.evaluate(output: str, ctx: EvalContext) -> AssertionResult` (Builder B).
- `Assertion.from_spec(spec: dict) -> Assertion` classmethod for construction from parsed YAML.
- Registry lookups: `assertions.get(type_name)`, `providers.get_provider(name)`.
- Everything speaks `models.py` dataclasses only.

---

## 3. Shared data contract (`models.py`)

All frozen `@dataclass` (immutable where practical). This is the single source of truth every builder imports.

```python
# ---- Inputs (parsed from YAML/Python) ----

@dataclass(frozen=True)
class AssertionSpec:
    type: str                     # "contains" | "regex" | "json_schema" | ...
    config: dict[str, Any]        # type-specific params (value, pattern, schema, threshold, ...)
    negate: bool = False          # generic negation wrapper (powers not_contains too)
    weight: float = 1.0           # for future weighted scoring; default equal
    description: str | None = None

@dataclass(frozen=True)
class Case:
    name: str                     # unique within a suite; used as test id + cassette key part
    provider: str                 # "openai" | "anthropic" | "replay" (resolved at run time)
    model: str                    # e.g. "gpt-4o-mini"
    prompt: str | None            # literal prompt OR...
    template: str | None          # ...a template string with {var} placeholders
    vars: dict[str, Any]          # values substituted into template
    params: dict[str, Any]        # temperature, max_tokens, seed, etc.
    assertions: list[AssertionSpec]
    tags: list[str] = field(default_factory=list)
    source_file: str | None = None
    line: int | None = None       # for pytest/report locating

    @property
    def rendered_prompt(self) -> str: ...   # template.format(**vars) or prompt

@dataclass(frozen=True)
class Suite:
    name: str
    cases: list[Case]
    defaults: dict[str, Any]      # provider/model/params merged into each case
    source_file: str

# ---- Provider I/O ----

@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    provider: str
    raw: dict[str, Any] = field(default_factory=dict)   # full provider response (redactable)
    usage: dict[str, int] = field(default_factory=dict) # prompt/compl tokens if available
    latency_ms: float | None = None

# ---- Assertion results ----

@dataclass(frozen=True)
class AssertionResult:
    type: str
    passed: bool
    message: str                  # human-readable pass/fail explanation
    detail: dict[str, Any] = field(default_factory=dict)  # e.g. similarity score, diff

# ---- Run results ----

@dataclass(frozen=True)
class CaseResult:
    case_name: str
    passed: bool                  # AND of all assertion results
    completion: Completion
    assertion_results: list[AssertionResult]
    error: str | None = None      # provider/render error (counts as fail)
    duration_ms: float | None = None

    @property
    def failed_assertions(self) -> list[AssertionResult]: ...

@dataclass
class RunResult:                  # mutable while accumulating
    suite_name: str
    provider: str
    model: str
    started_at: str               # ISO8601
    case_results: list[CaseResult] = field(default_factory=list)
    @property
    def passed(self) -> bool: ...
    @property
    def counts(self) -> dict[str, int]: ...   # {"passed": n, "failed": m, "errors": k}

@dataclass(frozen=True)
class Report:                     # serializable summary of a RunResult
    schema_version: int           # for forward-compat of on-disk reports
    promptdiff_version: str
    run: RunResult
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, s: str) -> "Report": ...

# ---- Cassette (record/replay fixture) ----

@dataclass(frozen=True)
class CassetteEntry:
    key: str                      # sha256 of canonical(provider, model, prompt, params)
    provider: str
    model: str
    request: dict[str, Any]       # {prompt, params} — stored for human inspection
    response: Completion
    recorded_at: str

@dataclass
class Cassette:
    version: int
    entries: dict[str, CassetteEntry]   # key -> entry
    path: str
    def get(self, key: str) -> CassetteEntry | None: ...
    def put(self, entry: CassetteEntry) -> None: ...
    @classmethod
    def load(cls, path: str) -> "Cassette": ...
    def save(self) -> None: ...
```

**Cassette key hashing (deterministic, Builder C):**
`key = sha256(json.dumps({"provider":..,"model":..,"prompt": rendered_prompt, "params": sorted params}, sort_keys=True, ensure_ascii=False))`.
Volatile params that must NOT affect the key (e.g. streaming flags, client timeouts) are stripped by a `CANONICAL_PARAM_ALLOWLIST` before hashing. Documented so record/replay stays stable.

---

## 4. Assertion catalog (`assertions/`)

Common shape: each assertion subclasses `Assertion(ABC)`, implements `evaluate(output, ctx) -> AssertionResult` and `from_spec(spec) -> Assertion`. Generic `negate` is applied by the base class after `evaluate` (so `not_contains` == `contains` + `negate`). Registered by string name in `assertions/__init__.py`.

`EvalContext` carries `{case, cassette_dir, baseline_dir}` so snapshot/similarity can read/write side files.

| Type (`AssertionSpec.type`) | Config keys | How it's evaluated | How it's unit-tested |
|---|---|---|---|
| `contains` | `value` (str \| list[str]), `case_sensitive` (bool), `all`/`any` | substring membership in output | `evaluate("hello world", value="world") → passed`; `value="xyz" → fail`; list + `all`/`any` combos; case-insensitive match |
| `not_contains` | same as contains | `contains` with `negate=True` | asserts forbidden string absent passes, present fails |
| `regex` | `pattern`, `flags` (i/m/s), `full_match` (bool) | `re.search`/`re.fullmatch` | known-good pattern matches; anchored pattern fails on partial; invalid regex → construction error surfaced clearly |
| `exact` | `value`, `strip` (bool, default True) | string equality after optional strip | exact equal passes; whitespace-diff with strip on/off; trailing-newline case |
| `json_schema` | `schema` (inline dict or `schema_file`) | parse output as JSON (tolerate ```json fences via extractor), validate with `jsonschema` | valid JSON+schema passes; wrong type fails with jsonpath in message; non-JSON output fails gracefully; **real schema fixture** (see QG) |
| `semantic_similarity` | `reference` (str), `threshold` (0–1), `metric` (`cosine`/`token_overlap`) | if `sentence-transformers` installed → embed both, cosine sim; else **offline fallback** = Jaccard token-overlap over normalized tokens. `passed = score >= threshold` | with fallback forced: identical strings → score 1.0 pass; disjoint strings → ~0 fail; paraphrase near-threshold boundary; both code paths behind a `_backend` seam so the embedding path is mockable |
| `snapshot` | `update` (bool, from CLI), snapshot stored under `__snapshots__/<suite>/<case>.txt` | first run with no snapshot → **record** (pass, marked "new"); subsequent runs → compare output to stored snapshot; mismatch → fail with unified diff in `detail` | no snapshot → creates file, passes; identical output → pass; changed output → fail + diff present; `update=True` rewrites snapshot and passes |

**Extractor helper** (shared, in `assertions/json_schema.py` or a small `_text.py`): strips markdown code fences and leading prose to find the JSON payload, so schema/exact assertions are robust to chatty models. Unit-tested on fenced, prose-prefixed, and raw JSON.

**Registry test:** every name in the registry round-trips through `from_spec` and appears in CLI `--list-assertions`.

---

## 5. Providers & cassettes (`providers/`)

- `base.Provider` ABC: `generate(prompt, params) -> Completion`; `name` classvar.
- `openai.OpenAIProvider` / `anthropic.AnthropicProvider`: lazy-import the SDK inside `generate` so the package imports with **zero** heavy/optional deps. Read key from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. Raise a clear `MissingProviderDep` / `MissingAPIKey` if used without setup.
- `replay.ReplayProvider`: wraps a `Cassette`. `generate` hashes the request key and returns the recorded `Completion`; **cache miss → `CassetteMiss` error** (never silently calls network). This is the default provider in CI.
- `cassette.py`: load/save JSON cassettes (`tests/cassettes/<suite>.json` by default), key hashing, redaction of secrets from `raw`, and a `RecordingProvider` wrapper: calls a real provider, stores the result, then behaves like replay. Records are **stable and diffable** (sorted keys, pretty JSON) so they review cleanly in PRs.

**Record/replay modes** (mirrors VCR): `none` (live only), `once` (replay if present else record), `new_episodes` (replay existing, record misses), `all` (always re-record). CLI `--record-mode`.

---

## 6. Runner & reporting (`runner.py`, `reporting.py`, `diff.py`)

- `runner.run_suite(suite, provider, record_mode) -> RunResult`: for each case → render prompt → `provider.generate` → run each assertion → build `CaseResult`. Catches render/provider errors as failing results (never crashes the whole run). Deterministic ordering.
- `reporting.py`: `render_console(run)` via `rich` (pass/fail table, per-assertion detail, diffs), `to_junit_xml(run)` for CI, `Report.to_json` for `diff`.
- `diff.py`: `diff_reports(baseline: Report, current: Report) -> DiffReport`: aligns by `case_name`, classifies each as `passed→failed` (**regression**), `failed→passed` (fixed), `new`, `removed`, `unchanged`. Exit non-zero if any regressions. This is what `promptdiff diff` renders.

---

## 7. CLI + pytest plugin surface

### Typer CLI (`cli.py`) — `promptdiff <cmd>`

| Command | Purpose | Key flags |
|---|---|---|
| `run [PATHS...]` | Load suites, run assertions, report | `--provider`, `--record-mode {none,once,new_episodes,all}` (default `once`), `--cassette-dir`, `--update-snapshots`, `--report json\|junit\|console`, `--out FILE`, `-k EXPR` (filter by name/tag), `--fail-fast`, `--offline` (force replay) |
| `record [PATHS...]` | Capture live outputs into cassettes | `--provider`, `--overwrite`, `--cassette-dir` (implies record-mode `all` on selected cases) |
| `diff BASELINE.json [CURRENT.json]` | Compare two reports; show regressions | `--fail-on-regression` (default on), `--format console\|json` |
| `list` | List discovered cases / assertion types | `--assertions`, `--cases`, `--tags` |
| `init` | Scaffold `promptdiff.yaml` + example suite + `.github` snippet | `--dir` |
| `version` | Print version | — |

Global: `--config promptdiff.yaml` for defaults (provider, model, cassette dir, paths).

Exit codes: `0` all pass, `1` assertion failure/regression, `2` config/load error, `3` cassette miss in offline mode.

### pytest plugin (`pytest_plugin.py`)

- Registered via `entry_points` group `pytest11` = `promptdiff = promptdiff.pytest_plugin`.
- `pytest_collect_file`: collect `*.promptdiff.yaml` / `prompts/**.yaml` as a `PromptFile` → yields one `PromptItem` per `Case` (test id = `file::case_name`).
- Each item runs the same `runner` path with `ReplayProvider` (offline by default) and raises `PromptAssertionError` (custom, with clean `repr_failure`) on failure — so `pytest`, `-k`, `-x`, `-q`, xdist, and CI reporters all work natively.
- Config via `pytest.ini`/`pyproject`: `[tool.promptdiff]` cassette dir, record mode (default `none` in pytest = strictly hermetic), snapshot update via `--promptdiff-update-snapshots`.
- Also expose a Python API: `promptdiff.run_suite(...)` and a `@promptdiff.case(...)` decorator for defining cases in Python test files.

---

## 8. Quality gate (all real, no fakes) — 15+ enumerated tests

All tests run **offline, no API key**, via cassettes + forced token-overlap similarity backend. Test layout: `tests/` mirrors package; fixtures in `tests/fixtures/`, cassettes in `tests/cassettes/`.

**Assertion tests (against known literal strings):**
1. `test_contains_pass_and_fail` — "the sky is blue" contains "sky" passes, contains "green" fails.
2. `test_contains_case_insensitive_and_list_all_any` — list with `all`/`any` + case folding.
3. `test_not_contains` — forbidden token absent passes / present fails (negate path).
4. `test_regex_search_vs_fullmatch` — anchored fullmatch fails on partial, search passes.
5. `test_regex_invalid_pattern_errors_clearly` — bad pattern raises at construction with useful message.
6. `test_exact_strip_behavior` — trailing-newline/whitespace with `strip` on and off.
7. `test_json_schema_valid_case` — **real** JSON output validated against a real inline schema (object with required `name:str`, `age:int≥0`) → passes.
8. `test_json_schema_type_violation` — age as string → fails, message names the offending path.
9. `test_json_schema_non_json_output` — plain prose → graceful fail, not a crash.
10. `test_json_extractor_handles_fences_and_prose` — ```json fenced and prose-prefixed payloads extracted correctly.
11. `test_similarity_identical_is_one` (offline backend) — identical strings score 1.0, pass at threshold 0.9.
12. `test_similarity_disjoint_is_low` — disjoint strings score ~0, fail at threshold 0.5.
13. `test_similarity_threshold_boundary` — near-threshold paraphrase behaves correctly at `>=`.

**Snapshot tests:**
14. `test_snapshot_first_run_records` — no snapshot present → creates file, passes as "new".
15. `test_snapshot_match_passes` — identical output vs stored snapshot passes.
16. `test_snapshot_drift_fails_with_diff` — changed output fails and `detail` contains a unified diff.
17. `test_snapshot_update_flag_rewrites` — `update=True` rewrites and passes.

**Cassette / provider tests:**
18. `test_cassette_key_is_deterministic` — same request → identical hash across runs/process restarts (canonicalization).
19. `test_replay_returns_recorded_completion` — ReplayProvider serves recorded text without network.
20. `test_replay_miss_raises_not_network` — unknown request → `CassetteMiss`, never a live call (assert no SDK import attempted).
21. `test_recording_roundtrip` — RecordingProvider (with a stub live provider) records, save→load→replay yields identical `Completion` (record/replay determinism).
22. `test_cassette_is_stable_diffable` — saving same content twice byte-identical (sorted keys).

**Runner / loader / diff / plugin tests:**
23. `test_loader_yaml_to_cases` — YAML fixture parses into expected `Case`/`AssertionSpec` objects, defaults merged.
24. `test_template_rendering_with_vars` — `{name}` substitution and missing-var error.
25. `test_runner_aggregates_pass_fail` — mixed suite → correct `RunResult.counts`.
26. `test_runner_provider_error_is_failure_not_crash`.
27. `test_diff_detects_regression` — baseline pass → current fail classified as regression, exit non-zero.
28. `test_diff_classifies_new_removed_fixed`.
29. `test_pytest_plugin_collects_yaml_cases` — via `pytester`, a YAML file yields N test items with correct ids.
30. `test_cli_run_offline_exit_codes` — Typer `CliRunner`: all-pass→0, failure→1, cassette miss→3.

(30 real tests; well above the 15 floor. Every assertion type is exercised against known strings; JSON-schema uses a real schema; cassette replay determinism is proven; no mocked "always pass" shims.)

**Tooling:**
- **ruff** (lint + format) — config in `pyproject.toml`; `ruff check` + `ruff format --check` in CI.
- **mypy** (optional, non-blocking gate initially) on `models.py` + public API.
- **pytest** with `--cov=promptdiff`, coverage floor ~85% enforced in CI.
- **GitHub Actions CI** (`.github/workflows/ci.yml`): matrix `python-version: [3.10, 3.11, 3.12]`, steps = install (`pip install -e .[dev]`), `ruff check`, `ruff format --check`, `pytest`. **No secrets required** — suite is fully offline, so CI is green on a fresh fork with zero config.
- Second workflow `release.yml`: on tag `v*`, build sdist+wheel, publish to PyPI via trusted publishing (manual approval), attach GitHub release notes.

**README outline:**
1. What/why (the "silent prompt regression" problem, one screenshot of a red diff).
2. Install (`pip install promptdiff`).
3. 60-second quickstart: write a YAML suite → `promptdiff record` → commit cassette → `promptdiff run`.
4. Assertion reference table.
5. Snapshot workflow.
6. Cassettes / offline / record modes.
7. pytest usage.
8. `diff` for CI regression gates + example GH Action.
9. Config reference (`promptdiff.yaml`).
10. Contributing, license.

**License:** MIT (`LICENSE` file, SPDX headers optional).
**Release:** tag **v0.1.0** once P0–P3 land, CI green on 3.10–3.12, README complete; publish to PyPI.

---

## 9. Offline / free-tier guarantee

- Package imports with **no** optional deps: `openai`, `anthropic`, `sentence-transformers` are all `[project.optional-dependencies]` extras, lazy-imported only when actually used.
- Default provider in both pytest plugin and `promptdiff run --offline`/CI is `ReplayProvider` → **zero network, zero cost**.
- Similarity assertion falls back to pure-Python token-overlap when `sentence-transformers` is absent → semantic tests still run offline.
- The **entire committed test suite passes with no API key set** (asserted by test 20 that no live SDK import is attempted in replay). Live providers are strictly opt-in for `record`.
- Cassettes are small JSON, committed to the repo → fresh clone + `pip install -e .[dev]` + `pytest` is green offline.

---

## 10. Phased roadmap

- **P0 — Scaffold** (Builder A leads)
  Repo layout, `pyproject.toml` (Typer, rich, jsonschema, pyyaml core deps; optional extras), `models.py` frozen + committed, `loader.py` YAML→Case, ruff config, empty CI, LICENSE, README skeleton. *Gate: `pip install -e .` + `promptdiff version` works; loader test green.*

- **P1 — Assertions + runner (offline)** (Builders B + D)
  All assertion types + registry (Builder B); `runner.py` + console reporting (Builder D) using a trivial in-memory stub provider. *Gate: assertion tests 1–17, runner tests 23–26 green; `promptdiff run` on a YAML suite with a stub provider works fully offline.*

- **P2 — Cassettes record/replay** (Builder C)
  Provider ABC, OpenAI/Anthropic (lazy), `ReplayProvider`, cassette engine, `RecordingProvider`, `promptdiff record`. *Gate: cassette tests 18–22 green; record against a stub live provider then replay deterministically.*

- **P3 — pytest plugin + baseline diff** (Builder D)
  `pytest_plugin.py` (entry point, collection), `diff.py` + `promptdiff diff`, `Report` JSON serialization, JUnit output, `promptdiff init`. *Gate: plugin/diff/CLI tests 27–30 green; example project runs under both `promptdiff run` and `pytest`.*

- **Ship — v0.1.0**
  Full README, CI matrix green on 3.10–3.12, coverage ≥85%, tag + PyPI publish.

**Parallelization:** after P0 freezes `models.py`, Builders B, C, and the reporting half of D run **in parallel** against the contract; D's wiring (runner→cli→plugin) integrates as B/C land. Recommended: A → (B ∥ C ∥ D-reporting) → D-integration.

---

## 11. Risks & differentiation

**Risks & mitigations:**
- *Non-determinism of LLMs breaks snapshots/replay.* → Cassettes make replay 100% deterministic; snapshots are compared only against recorded/committed output, never a fresh live call. Record modes are explicit.
- *Embedding dep is heavy / platform-fragile.* → It's an optional extra; token-overlap fallback keeps similarity offline and dependency-free.
- *Cassette key brittleness* (a param reorder invalidates every fixture). → Canonicalized, allowlisted, sorted hashing + a documented, tested key function; `new_episodes` mode to backfill.
- *Prompt/output PII or secrets in committed cassettes.* → Redaction hook on `Completion.raw`; docs warn; `--no-raw` to store only text.
- *Scope creep toward "eval platform."* → Stay a **regression harness**: snapshot + assert + diff. LLM-graded quality metrics are explicitly out of v0.1.
- *pytest collection edge cases (xdist, ids).* → Test via `pytester` early (P3), keep item ids stable and file-scoped.

**Differentiation restated:** promptdiff is the only tool that combines (1) **snapshot/baseline regression** as the primary workflow, (2) **native pytest ergonomics** a Python team already knows, and (3) **offline-by-default VCR cassettes** so CI is real-but-hermetic and **free**. Not "is my model good" (evals) but "**did my change break something**" — the question teams actually ship blind on today.
