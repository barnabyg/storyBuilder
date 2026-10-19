import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from nde.runs import prompts
from nde.runs.runner import prepare
from tests.conftest import REPO, base_config, write_config
from tests.fake_openai import FakeOpenAI, completion

SMOKE = REPO / "configs" / "smoke.yaml"


def _only_run_dir(runs: Path) -> Path:
    [run_dir] = list(runs.iterdir())
    return run_dir


def _manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def smoke_env(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    monkeypatch.setenv("NDE_DEEPSEEK_BASE_URL", fake_llm.base_url)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    return fake_llm


def test_smoke_run_records_manifest_output_and_report(
    smoke_env: FakeOpenAI, tmp_path: Path
) -> None:
    smoke_env.queue(completion("The lamp went out at midnight.", model="deepseek-flash"))
    runs = tmp_path / "runs"

    code = main(["run", str(SMOKE), "--runs-dir", str(runs)])

    assert code == 0
    run_dir = _only_run_dir(runs)
    manifest = _manifest(run_dir)
    assert manifest["status"] == "succeeded"
    assert manifest["seed"] == 20261018
    assert re.fullmatch(r"[0-9a-f]{40}", str(manifest["git_sha"]))
    assert re.fullmatch(r"[0-9a-f]{64}", str(manifest["config_hash"]))
    assert manifest["models"] == [
        {"role": "prose", "provider": "deepseek", "model": "deepseek-flash", "family": None}
    ]
    assert manifest["prompt_templates"] == [
        {"template": "smoke", "version": 1, "sha256": manifest["prompt_templates"][0]["sha256"]}
    ]
    assert manifest["started_at"] <= manifest["finished_at"]

    [output] = [
        json.loads(line)
        for line in (run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert output["text"] == "The lamp went out at midnight."

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "The lamp went out at midnight." in report
    assert "deepseek-flash" in report
    assert "smoke@1" in report

    [request] = smoke_env.requests
    assert request.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.body["model"] == "deepseek-flash"
    assert request.body["seed"] == 20261018
    assert request.body["temperature"] == 0.7
    assert request.body["thinking"] == {"type": "disabled"}
    assert request.body["messages"] == [
        {"role": "system", "content": "You are a concise fiction writer."},
        {
            "role": "user",
            "content": (
                "Write one opening sentence for a short story set in a lighthouse in a storm."
            ),
        },
    ]


@dataclass(frozen=True)
class InvalidCase:
    path: str
    value: object
    message: str


DELETE = object()

INVALID_CASES = {
    "unknown-role": InvalidCase(
        "roles.narrator", {"provider": "fake", "model": "m"}, "roles.narrator"
    ),
    "missing-template-version": InvalidCase(
        "calls.0.prompt.version", DELETE, "calls.0.prompt.version"
    ),
    "missing-model": InvalidCase("roles.prose.model", DELETE, "roles.prose.model"),
    "role-without-model": InvalidCase("calls.0.role", "judge", "role 'judge'"),
    "unknown-template-version": InvalidCase("calls.0.prompt.version", 99, "smoke@99 not found"),
    "missing-variable": InvalidCase("calls.0.variables", {}, "needs variable 'setting'"),
    "missing-api-key": InvalidCase("providers.fake.api_key_env", "NDE_UNSET_KEY", "NDE_UNSET_KEY"),
}


def _patch(config: dict[str, Any], path: str, value: object) -> None:
    *parents, leaf = path.split(".")
    node: Any = config
    for part in parents:
        node = node[int(part)] if part.isdigit() else node[part]
    if value is DELETE:
        del node[leaf]
    else:
        node[leaf] = value


@pytest.fixture
def keyed_llm(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    monkeypatch.delenv("NDE_UNSET_KEY", raising=False)
    return fake_llm


@pytest.mark.parametrize("case", INVALID_CASES.values(), ids=INVALID_CASES.keys())
def test_invalid_configs_fail_clearly_before_any_network_call(
    keyed_llm: FakeOpenAI, tmp_path: Path, capsys: pytest.CaptureFixture[str], case: InvalidCase
) -> None:
    config = base_config(keyed_llm.base_url)
    _patch(config, case.path, case.value)
    runs = tmp_path / "runs"

    code = main(["run", str(write_config(tmp_path, config)), "--runs-dir", str(runs)])

    assert code == 2
    assert case.message in capsys.readouterr().err
    assert keyed_llm.call_count == 0
    assert not runs.exists()


def test_changed_template_text_without_a_new_version_is_rejected(
    fake_llm: FakeOpenAI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    prompts = tmp_path / "prompts"
    (prompts / "smoke").mkdir(parents=True)
    (prompts / "smoke" / "v1.yaml").write_text("user: Hello $setting.\n", encoding="utf-8")
    assert main(["prompts", "lock", "--dir", str(prompts)]) == 0
    (prompts / "smoke" / "v1.yaml").write_text("user: Goodbye $setting.\n", encoding="utf-8")
    config = base_config(fake_llm.base_url) | {"prompts_dir": str(prompts)}

    code = main(["run", str(write_config(tmp_path, config)), "--runs-dir", str(tmp_path / "runs")])

    assert code == 2
    assert "smoke@1 changed after release" in capsys.readouterr().err
    assert fake_llm.call_count == 0
    assert main(["prompts", "lock", "--dir", str(prompts)]) == 2


def test_committed_prompt_templates_match_their_checksums() -> None:
    assert prompts.check(REPO / "prompts") == ([], [])


def test_api_key_can_come_from_a_dotenv_file(fake_llm: FakeOpenAI, tmp_path: Path) -> None:
    # pragma: allowlist nextline secret - a fake key for the fake server
    (tmp_path / ".env").write_text('FAKE_API_KEY="from-dotenv"\n', encoding="utf-8")
    config = write_config(tmp_path, base_config(fake_llm.base_url))

    assert main(["run", str(config), "--runs-dir", str(tmp_path / "runs")]) == 0
    assert fake_llm.requests[0].headers["authorization"] == "Bearer from-dotenv"


@pytest.mark.parametrize("config", sorted((REPO / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_committed_configs_are_valid(config: Path) -> None:
    keys = {
        "DEEPSEEK_API_KEY": "k",
        "OPENAI_API_KEY": "k",
    }  # pragma: allowlist secret - fake values

    prepared = prepare(config, keys)

    assert prepared.loaded.config.calls or prepared.stories
