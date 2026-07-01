"""CLI exit-code tests and pytest-plugin collection test (all offline)."""
from __future__ import annotations

import os

from typer.testing import CliRunner

from promptdiff import assertions
from promptdiff.cli import app

runner = CliRunner()
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CASSETTES = os.path.join(os.path.dirname(__file__), "cassettes")
GREETINGS = os.path.join(FIXTURES, "greetings.promptdiff.yaml")


def test_cli_version():
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "promptdiff" in res.stdout


def test_cli_list_assertions_matches_registry():
    res = runner.invoke(app, ["list", "--assertions"])
    assert res.exit_code == 0
    for name in assertions.list_types():
        assert name in res.stdout


def test_cli_run_all_pass_exit_0():
    res = runner.invoke(app, ["run", GREETINGS, "--cassette-dir", CASSETTES])
    assert res.exit_code == 0
    assert "passed" in res.stdout


def test_cli_run_failure_exit_1(tmp_path):
    suite = tmp_path / "fail.promptdiff.yaml"
    suite.write_text(
        "name: fail\n"
        "defaults: { provider: openai, model: gpt-4o-mini, params: { temperature: 0 } }\n"
        "cases:\n"
        "  - name: capital_of_france\n"
        "    prompt: \"What is the capital of France? Answer in one word.\"\n"
        "    assertions:\n"
        "      - type: contains\n"
        "        value: Berlin\n"
    )
    res = runner.invoke(app, ["run", str(suite), "--cassette-dir", CASSETTES])
    assert res.exit_code == 1


def test_cli_run_cassette_miss_exit_3(tmp_path):
    suite = tmp_path / "miss.promptdiff.yaml"
    suite.write_text(
        "name: miss\n"
        "defaults: { provider: openai, model: gpt-4o-mini }\n"
        "cases:\n"
        "  - name: unknown\n"
        "    prompt: \"never recorded prompt\"\n"
        "    assertions:\n"
        "      - type: contains\n"
        "        value: x\n"
    )
    res = runner.invoke(app, ["run", str(suite), "--cassette-dir", str(tmp_path)])
    assert res.exit_code == 3


def test_cli_diff_regression_exit_1(tmp_path):
    from promptdiff.models import CaseResult, Completion, Report, RunResult

    def rep(ok):
        run = RunResult(suite_name="s", provider="replay", model="m",
                        started_at="t", case_results=[
                            CaseResult("c", ok, Completion("", "m", "replay"))])
        return Report(run=run, promptdiff_version="0.1.0")

    base = tmp_path / "base.json"
    curr = tmp_path / "curr.json"
    base.write_text(rep(True).to_json())
    curr.write_text(rep(False).to_json())
    res = runner.invoke(app, ["diff", str(base), str(curr)])
    assert res.exit_code == 1


def test_cli_init_scaffolds(tmp_path):
    res = runner.invoke(app, ["init", "--dir", str(tmp_path)])
    assert res.exit_code == 0
    assert (tmp_path / "prompts" / "example.promptdiff.yaml").exists()


# ---- pytest plugin collection (via pytester) ------------------------------

pytest_plugins = ["pytester"]


def test_pytest_plugin_collects_yaml_cases(pytester):
    pytester.makefile(
        ".json",
        **{"tests/cassettes/mini": (
            '{"version": 1, "entries": {}}'
        )},
    )
    # a suite whose single case will pass against an inline cassette entry
    from promptdiff.models import Cassette, Completion, cassette_key  # noqa: F401
    from promptdiff.providers.cassette import make_entry

    cas = Cassette(path=str(pytester.path / "cassettes" / "mini.json"))
    cas.put(make_entry("openai", "m", "say hi", {},
                       Completion("hi there", "m", "openai")))
    os.makedirs(os.path.dirname(cas.path), exist_ok=True)
    cas.save()

    pytester.makefile(
        ".yaml",
        **{"mini.promptdiff": (
            "name: mini\n"
            "defaults: { provider: openai, model: m }\n"
            "cases:\n"
            "  - name: greet\n"
            "    prompt: \"say hi\"\n"
            "    assertions:\n"
            "      - type: contains\n"
            "        value: hi\n"
        )},
    )
    result = pytester.runpytest("-p", "promptdiff.pytest_plugin",
                                "-o", "promptdiff_cassette_dir=cassettes", "-v")
    result.assert_outcomes(passed=1)
    assert "greet" in result.stdout.str()
