"""Cycle behavior at the ticket-source/runner/publication boundary; no live issues."""

import copy
import json
import multiprocessing
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from tools import hygiene

REPO = Path(__file__).resolve().parents[1]


def config() -> dict[str, Any]:
    result = hygiene.read_json(REPO / ".codex" / "hygiene.json")
    result["enabled"] = True
    return result


def ticket(number: int, **changes: Any) -> dict[str, Any]:
    return {
        "node_id": f"issue-{number}",
        "number": number,
        "state": "closed",
        "state_reason": "completed",
        "labels": [{"name": "ready-for-agent"}],
        "title": f"Implement behavior {number}",
        "body": "",
        "html_url": f"https://example/{number}",
        **changes,
    }


class FakeGitHub(hygiene.GitHub):
    def __init__(self, issues: list[dict[str, Any]]):
        self.items = issues
        self.created: list[dict[str, Any]] = []
        self.offline = False
        self.ambiguous = False

    def issues(self) -> list[dict[str, Any]]:
        if self.offline:
            raise OSError("GitHub offline")
        return copy.deepcopy(self.items)

    def target(self) -> str:
        return "default-branch-commit"

    def create(self, title: str, body: str, labels: list[str]) -> dict[str, Any]:
        issue = ticket(
            100 + len(self.created),
            title=title,
            body=body,
            state="open",
            labels=[{"name": label} for label in labels],
        )
        self.items.append(issue)
        self.created.append(issue)
        if self.ambiguous:
            self.ambiguous = False
            raise TimeoutError("POST response lost after write")
        return copy.deepcopy(issue)


def result(*findings: dict[str, Any]) -> dict[str, Any]:
    return {
        "completed": True,
        "report": "Inspected snapshot. Measurements and limits.",
        "findings": list(findings),
    }


def finding(problem: str = "stale-command", **changes: Any) -> dict[str, Any]:
    return {
        "category": "documentation",
        "area": "setup",
        "problem": problem,
        "title": "Fix the stale setup command",
        "body": "Evidence and acceptance criteria.",
        "existing_issue": 0,
        **changes,
    }


@pytest.fixture
def cycle(tmp_path: Path) -> hygiene.Cycle:
    instance = hygiene.Cycle(tmp_path, config(), tmp_path / "common" / "project-hygiene")
    instance.install([ticket(0)], "installation-commit")
    return instance


def test_four_do_nothing_five_coalesce_and_success_consumes_only_snapshot(
    cycle: hygiene.Cycle,
) -> None:
    github = FakeGitHub([ticket(i) for i in range(5)])
    calls: list[dict[str, Any]] = []

    def audit(
        run: dict[str, Any], _directory: Path, _issues: list[dict[str, Any]]
    ) -> dict[str, Any]:
        calls.append(copy.deepcopy(run))
        github.items.append(ticket(6))
        return result()

    assert cycle.dispatch(github, audit) == "pending: 4/5"
    assert calls == []
    github.items.append(ticket(5))
    assert "succeeded" in cycle.dispatch(github, audit)
    assert len(calls) == 1
    assert calls[0]["commit"] == github.target()
    state = cycle.load()
    assert state["consumed_ids"] == [f"issue-{i}" for i in range(1, 6)]
    assert cycle.pending(state, github.items) == ["issue-6"]
    assert state["last_success"]["report"].endswith("report.md")
    assert cycle.dispatch(github, audit) == "pending: 1/5"


def test_eligibility_excludes_control_hygiene_cancelled_and_processed_reclosures(
    cycle: hygiene.Cycle,
) -> None:
    issues = [
        ticket(0),
        ticket(1, labels=[{"name": "ready-for-human"}]),
        ticket(2, state_reason="not_planned"),
        ticket(3, pull_request={}),
        ticket(4, labels=[{"name": "ready-for-agent"}, {"name": "project-hygiene"}]),
        ticket(5, body="<!-- project-hygiene:audit=old -->"),
        ticket(6, title="Set up project-hygiene"),
        ticket(7, state="open"),
        ticket(8, labels=[{"name": "ready-for-agent"}, {"name": "wayfinder:map"}]),
        ticket(9),
    ]
    assert cycle.pending(cycle.load(), issues) == ["issue-9"]
    github = FakeGitHub(issues)
    cycle.dispatch(github, lambda _run, _directory, _issues: result(), manual=True)
    github.items[-1]["state"] = "open"
    assert cycle.pending(cycle.load(), github.items) == []
    github.items[-1]["state"] = "closed"
    assert cycle.pending(cycle.load(), github.items) == []


def test_rerun_and_dry_run_preserve_baseline_checkpoint_and_pending_run(
    cycle: hygiene.Cycle,
) -> None:
    github = FakeGitHub([ticket(i) for i in range(9)])
    initial = cycle.path.read_bytes()
    assert "would dispatch: 8" in cycle.dispatch(
        github, lambda _run, _directory, _issues: result(), dry_run=True
    )
    assert cycle.path.read_bytes() == initial

    def broken(*_args: Any) -> dict[str, Any]:
        raise RuntimeError("runner unavailable")

    with pytest.raises(RuntimeError, match="unavailable"):
        cycle.dispatch(github, broken)
    before = cycle.path.read_bytes()
    assert cycle.install(github.items, "later-commit")["last_success"] is None
    assert cycle.path.read_bytes() == before
    assert cycle.load()["installation"]["commit"] == "installation-commit"
    assert cycle.load()["run"]["status"] == "failed"
    cycle.config["eligibility"]["excluded_labels"].append("new-rule")
    with pytest.raises(ValueError, match="migration"):
        cycle.load()


def test_api_outage_preserves_state_and_worker_recursion_is_suppressed(
    cycle: hygiene.Cycle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    github = FakeGitHub([ticket(i) for i in range(6)])
    initial = cycle.path.read_bytes()
    github.offline = True
    with pytest.raises(OSError, match="offline"):
        cycle.dispatch(github, lambda _run, _directory, _issues: result())
    assert cycle.path.read_bytes() == initial
    monkeypatch.setenv(hygiene.WORKER_ENV, "1")
    assert (
        cycle.dispatch(github, lambda _run, _directory, _issues: result())
        == "audit-worker: suppressed"
    )
    assert cycle.path.read_bytes() == initial


def test_ambiguous_publication_retries_same_snapshot_without_duplicate_issues(
    cycle: hygiene.Cycle,
) -> None:
    github = FakeGitHub([ticket(i) for i in range(6)])
    github.ambiguous = True
    attempts: list[str] = []

    def audit(
        run: dict[str, Any], _directory: Path, _issues: list[dict[str, Any]]
    ) -> dict[str, Any]:
        attempts.append(run["id"])
        return result(finding(), finding("second-problem"))

    with pytest.raises(TimeoutError, match="response lost"):
        cycle.dispatch(github, audit)
    failed = cycle.load()
    assert failed["consumed_ids"] == []
    assert failed["last_success"] is None
    github.items.append(ticket(9))
    assert "succeeded" in cycle.dispatch(github, audit)
    state = cycle.load()
    assert len(attempts) == 1
    assert len(github.created) == 2
    assert len(state["publications"]) == 2
    assert state["last_success"]["id"] == attempts[0]
    assert cycle.pending(state, github.items) == ["issue-9"]


def test_manual_equivalent_and_dismissed_marker_are_reused_without_edits(
    cycle: hygiene.Cycle,
) -> None:
    manual = ticket(70, title="Old setup issue", state="open")
    dismissed = ticket(
        71,
        body=hygiene.finding_marker(cycle.config["repository"], finding("dismissed")),
        state_reason="not_planned",
    )
    github = FakeGitHub([ticket(i) for i in range(6)] + [manual, dismissed])
    cycle.dispatch(
        github,
        lambda _run, _directory, _issues: result(finding(existing_issue=70), finding("dismissed")),
    )
    assert github.created == []
    assert github.items[-2:] == [manual, dismissed]
    assert len(cycle.load()["publications"]) == 2


def _locked_attempt(directory: str, queue: Any) -> None:
    try:
        with hygiene.exclusive(Path(directory)):
            queue.put("acquired")
    except OSError:
        queue.put("busy")


def test_kernel_lock_serializes_worktrees_and_recovers_after_process_exit(tmp_path: Path) -> None:
    shared = tmp_path / "git-common" / "project-hygiene"
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    with hygiene.exclusive(shared):
        worker = ctx.Process(target=_locked_attempt, args=(str(shared), queue))
        worker.start()
        worker.join(timeout=20)
        assert worker.exitcode == 0
        assert queue.get(timeout=2) == "busy"
    worker = ctx.Process(target=_locked_attempt, args=(str(shared), queue))
    worker.start()
    worker.join(timeout=20)
    assert worker.exitcode == 0
    assert queue.get(timeout=2) == "acquired"
    queue.close()


def test_stale_owner_cannot_finalize_or_publish(cycle: hygiene.Cycle) -> None:
    state = cycle.load()
    old = copy.deepcopy(cycle.claim(state, ["issue-1"], "commit"))
    cycle.claim(cycle.load(), [], "commit")
    with pytest.raises(RuntimeError, match="ownership lost"):
        cycle.publish(old, result(finding()), FakeGitHub([]))
    with pytest.raises(RuntimeError, match="ownership lost"):
        cycle.failure(old, "stale")


def test_incomplete_audit_never_consumes_tickets(cycle: hygiene.Cycle) -> None:
    github = FakeGitHub([ticket(i) for i in range(6)])
    with pytest.raises(ValueError, match="incomplete"):
        cycle.dispatch(github, lambda _run, _directory, _issues: {**result(), "completed": False})
    assert cycle.load()["consumed_ids"] == []
    assert cycle.load()["last_success"] is None
    github.items.append(ticket(7))
    assert "succeeded" in cycle.dispatch(github, lambda _run, _directory, _issues: result())
    assert "issue-7" not in cycle.load()["consumed_ids"]


def test_read_only_status_and_hook_guard_from_a_subdirectory(
    tmp_path: Path,
    cycle: hygiene.Cycle,
) -> None:
    blank = hygiene.Cycle(tmp_path, config(), tmp_path / "absent")
    assert hygiene.status(blank, FakeGitHub([]))["installation"] is None
    assert not blank.directory.exists()
    subdirectory = tmp_path / "subdirectory"
    subdirectory.mkdir()
    completed = subprocess.run(  # noqa: S603 - fixed helper and test environment.
        [sys.executable, str(REPO / "tools/hygiene.py"), "hook"],
        cwd=subdirectory,
        env={**os.environ, hygiene.WORKER_ENV: "1"},
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert not completed.stdout
    assert not completed.stderr
    assert hygiene.status(cycle, FakeGitHub([ticket(1)]))["pending_ids"] == ["issue-1"]


def test_exact_hook_bootstrap_resolves_git_root_from_docs() -> None:
    definition = hygiene.hook_definition()
    windows_command = definition["hooks"][0]["commandWindows"]
    args = shlex.split(windows_command)
    assert args[:2] == ["python", "-c"]
    completed = subprocess.run(  # noqa: S603 - reviewed hook bootstrap, recursion guard prevents work.
        [sys.executable, *args[1:]],
        cwd=REPO / "docs",
        env={**os.environ, hygiene.WORKER_ENV: "1"},
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout == b""
    assert completed.stderr == b""


def test_github_listing_paginates_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    github = hygiene.GitHub(REPO, "barnabyg/storyBuilder")
    calls: list[str] = []

    def api(suffix: str, **_kwargs: Any) -> Any:
        calls.append(suffix)
        return [ticket(i) for i in range(100)] if suffix.endswith("page=1") else [ticket(100)]

    monkeypatch.setattr(github, "api", api)
    assert len(github.issues()) == 101
    assert len(calls) == 2

    def incomplete(_suffix: str) -> dict[str, str]:
        return {"error": "outage"}

    monkeypatch.setattr(github, "api", incomplete)
    with pytest.raises(ValueError, match="incomplete"):
        github.issues()


def test_disabled_dispatch_and_configuration_schema(cycle: hygiene.Cycle) -> None:
    cycle.config["enabled"] = False
    github = FakeGitHub([])
    github.offline = True
    assert cycle.dispatch(github, lambda _run, _directory, _issues: result()) == "disabled"
    schema = json.loads((REPO / "tools/hygiene/schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["findings"]["maxItems"] == cycle.config["limits"]["new_issue_cap"]


def test_setup_hook_rerun_preserves_unrelated_hooks_and_state(cycle: hygiene.Cycle) -> None:
    path = cycle.root / ".codex" / "hooks.json"
    unrelated = {"matcher": "compact", "hooks": [{"type": "command", "command": "existing"}]}
    initial = {"description": "Keep this", "hooks": {"SessionStart": [unrelated], "Stop": []}}
    hygiene.write_json(path, initial)
    baseline = cycle.path.read_bytes()
    hygiene.install_hook(cycle.root)
    first = path.read_bytes()
    hygiene.install_hook(cycle.root)
    assert path.read_bytes() == first
    assert cycle.path.read_bytes() == baseline
    hooks = hygiene.read_json(path)
    assert hooks["description"] == "Keep this"
    assert hooks["hooks"]["SessionStart"] == [unrelated, hygiene.hook_definition()]
    assert hooks["hooks"]["Stop"] == []


def test_hook_installer_preserves_existing_toml_representation(tmp_path: Path) -> None:
    folder = tmp_path / ".codex"
    folder.mkdir()
    path = folder / "config.toml"
    original = '[[hooks.Stop]]\n[[hooks.Stop.hooks]]\ntype="command"\ncommand="existing"\n'
    path.write_text(original, encoding="utf-8")
    hygiene.install_hook(tmp_path)
    first = path.read_bytes()
    hygiene.install_hook(tmp_path)
    assert path.read_bytes() == first
    assert path.read_text(encoding="utf-8").startswith(original)
    assert not (folder / "hooks.json").exists()


def test_concurrent_dispatch_checks_across_worktrees_launch_once(
    cycle: hygiene.Cycle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Model the second checkout resolving exactly the same Git common directory.
    second = hygiene.Cycle(cycle.root / "other-checkout", cycle.config, cycle.directory)
    github = FakeGitHub([ticket(i) for i in range(6)])
    calls: list[str] = []

    def audit(run: dict[str, Any], _path: Path, _issues: list[dict[str, Any]]) -> dict[str, Any]:
        calls.append(run["id"])
        with pytest.raises(OSError, match=r".*"):
            second.dispatch(github, audit)
        return result()

    # On Windows the kernel enforces locks even between independent handles in
    # one process; POSIX flock behaves the same for separately opened files.
    monkeypatch.delenv(hygiene.WORKER_ENV, raising=False)
    assert "succeeded" in cycle.dispatch(github, audit)
    assert second.dispatch(github, audit) == "pending: 0/5"
    assert len(calls) == 1


@pytest.mark.parametrize("keep_outputs", [False, True])
def test_runner_isolates_snapshot_and_cleans_only_empty_worktrees(
    cycle: hygiene.Cycle,
    monkeypatch: pytest.MonkeyPatch,
    keep_outputs: bool,
) -> None:
    hygiene.git(cycle.root, "init", "--initial-branch=main")
    (cycle.root / "README.md").write_text("committed input", encoding="utf-8")
    (cycle.root / ".gitignore").write_text("/.audit-cache\n/common/\n", encoding="utf-8")
    hygiene.git(cycle.root, "add", "README.md", ".gitignore")
    hygiene.git(
        cycle.root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "fixture",
    )
    commit = hygiene.git(cycle.root, "rev-parse", "HEAD")
    hygiene.git(cycle.root, "remote", "add", "origin", str(cycle.root))
    (cycle.root / "README.md").write_text("ongoing owner work", encoding="utf-8")
    run = cycle.claim(cycle.load(), ["issue-1"], commit)
    directory = cycle.directory / "runs" / run["id"]
    directory.mkdir(parents=True)
    observed: list[Path] = []

    def bundle(_cycle: hygiene.Cycle, target: Path) -> None:
        (target / "prompt.md").write_text("fixture audit", encoding="utf-8")
        hygiene.write_json(target / "tools.json", {"python": sys.executable, "uv": sys.executable})

    def fake_agent(
        root: Path,
        _bundle: Path,
        _prompt: str,
        *,
        seconds: int,
        sandbox: str,
    ) -> dict[str, Any]:
        assert seconds == 1200
        assert sandbox == "workspace-write"
        assert hygiene.git(root, "rev-parse", "HEAD") == commit
        assert hygiene.git(root, "branch", "--show-current").startswith("codex/hygiene-")
        assert (root / "README.md").read_text(encoding="utf-8") == "committed input"
        observed.append(root)
        if keep_outputs:
            (root / ".audit-cache").write_text("diagnostic evidence", encoding="utf-8")
        return result()

    monkeypatch.setattr(hygiene, "prepare_bundle", bundle)
    monkeypatch.setattr(hygiene, "agent", fake_agent)
    assert hygiene.isolated_audit(cycle, run, directory, []) == result()
    assert (cycle.root / "README.md").read_text(encoding="utf-8") == "ongoing owner work"
    assert observed[0].exists() is keep_outputs
    assert (directory / "retained-checkout.txt").exists() is keep_outputs
    if keep_outputs:
        (observed[0] / ".audit-cache").unlink()
        hygiene.git(cycle.root, "worktree", "remove", str(observed[0]))
        hygiene.git(cycle.root, "branch", "-d", "codex/hygiene-" + observed[0].name)
