"""Single-machine hygiene cycle. Run with Python 3.12; no project install needed."""

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import tomllib
import uuid
from collections.abc import Callable, Generator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

if os.name == "nt":
    import msvcrt
else:
    import fcntl

Json = dict[str, Any]
WORKER_ENV = "STORYBUILDER_HYGIENE_WORKER"
MAX_REPORT_BYTES = 128_000
PAGE_SIZE = 100
THRESHOLD = 5


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Json:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(Json, value)


def write_json(path: Path, value: Any) -> None:
    """Atomic replacement, including when two worktrees share the state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as f:
        temporary = Path(f.name)
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def executable(name: str) -> str:
    # Prefer native binaries on Windows: avoid executing .ps1 or interpolating
    # command arguments through cmd.exe. The desktop installs codex.exe on PATH.
    found = shutil.which(name + ".exe") if os.name == "nt" else shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} executable is unavailable on PATH")
    return found


def command(args: Sequence[str], cwd: Path, *, timeout: int = 30) -> str:
    # S603: argument arrays only; callers constrain executables and repository paths.
    result = subprocess.run(  # noqa: S603
        args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=timeout, check=True
    )
    return result.stdout.strip()


def git(root: Path, *args: str) -> str:
    return command([executable("git"), *args], root)


@contextlib.contextmanager
def exclusive(directory: Path) -> Generator[None]:
    """Kernel lock held through publication; process death releases ownership.

    A contender fails promptly. No lease expiry can displace a live worker.
    The saved run survives a dead worker and is reclaimed only under this lock.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class GitHub:
    """Host-keyring access; all API operations stay in the configured repository."""

    def __init__(self, root: Path, repository: str):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
            raise ValueError("invalid repository identity")
        self.root = root
        self.repository = repository

    def api(self, suffix: str, *, payload: Json | None = None) -> Any:
        args = [executable("gh"), "api", "--hostname", "github.com"]
        args.append(f"repos/{self.repository}{suffix}")
        if payload is None:
            return json.loads(command(args, self.root))
        # JSON on stdin preserves multiline issue bodies and never exposes credentials.
        args += ["--method", "POST", "--input", "-"]
        result = subprocess.run(  # noqa: S603 - executable and repository are constrained above.
            args,
            cwd=self.root,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=True,
        )
        return json.loads(result.stdout)

    def issues(self) -> list[Json]:
        """Fetch every page, including closed issues, before deciding eligibility."""
        result: list[Json] = []
        page = 1
        while True:
            batch = self.api(f"/issues?state=all&per_page=100&page={page}")
            if not isinstance(batch, list):
                raise ValueError("incomplete GitHub issue listing")
            items = cast(list[Json], batch)
            result.extend(item for item in items if "pull_request" not in item)
            if len(items) < PAGE_SIZE:
                return result
            page += 1

    def target(self) -> str:
        branch = self.api("")["default_branch"]
        return self.api(f"/commits/{branch}")["sha"]

    def create(self, title: str, body: str, labels: list[str]) -> Json:
        return self.api("/issues", payload={"title": title, "body": body, "labels": labels})


def eligible(issues: list[Json], rule: Json) -> set[str]:
    result: set[str] = set()
    for issue in issues:
        labels = {label["name"] for label in issue["labels"]}
        if (
            "pull_request" not in issue
            and issue["state"] == "closed"
            and issue.get("state_reason") == rule["closed_reason"]
            and labels.intersection(rule["implementation_labels"])
            and not labels.intersection(rule["excluded_labels"])
            and not re.search(rule["excluded_title_pattern"], issue["title"], re.IGNORECASE)
            and "<!-- project-hygiene:" not in (issue.get("body") or "")
        ):
            result.add(issue["node_id"])
    return result


def rule_identity(config: Json) -> str:
    value = {key: config[key] for key in ("repository", "eligibility")}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def finding_marker(repository: str, finding: Json) -> str:
    keys = [repository, *[finding[key].strip().lower() for key in ("category", "area", "problem")]]
    fingerprint = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    return f"<!-- project-hygiene:finding={fingerprint} -->"


def validate_result(result: Json, config: Json) -> None:
    if result.get("completed") is not True:
        raise ValueError("audit incomplete")
    report = result.get("report")
    if not isinstance(report, str) or not report.strip() or len(report.encode()) > MAX_REPORT_BYTES:
        raise ValueError("missing or oversized audit report")
    findings = result.get("findings")
    if (
        not isinstance(findings, list)
        or len(cast(list[Any], findings)) > config["limits"]["new_issue_cap"]
    ):
        raise ValueError("invalid findings or issue cap exceeded")
    for finding in cast(list[Json], findings):
        if finding["category"] not in ("documentation", "tests", "onboarding"):
            raise ValueError("invalid finding category")
        for key in ("area", "problem", "title", "body"):
            if not isinstance(finding[key], str) or not finding[key].strip():
                raise ValueError(f"missing finding {key}")
        if type(finding["existing_issue"]) is not int or finding["existing_issue"] < 0:
            raise ValueError("invalid existing issue number")


class Cycle:
    def __init__(self, root: Path, config: Json, directory: Path):
        self.root = root
        self.config = config
        self.directory = directory
        self.path = directory / "state.json"

    def load(self) -> Json:
        state = read_json(self.path)
        if state["version"] != 1 or state["rule_identity"] != rule_identity(self.config):
            raise ValueError("eligibility changed: explicit baseline migration required")
        return state

    def install(self, issues: list[Json], target: str) -> Json:
        """Reruns preserve the existing baseline, checkpoint, run, and ledger."""
        if self.path.exists():
            return self.load()
        state: Json = {
            "version": 1,
            "rule_identity": rule_identity(self.config),
            "installation": {
                "at": now(),
                "commit": target,
                "eligible_ids": sorted(eligible(issues, self.config["eligibility"])),
            },
            "consumed_ids": [],
            "last_success": None,
            "run": None,
            "publications": {},
            "diagnostic": None,
        }
        write_json(self.path, state)
        return state

    def pending(self, state: Json, issues: list[Json]) -> list[str]:
        processed = set(state["installation"]["eligible_ids"]) | set(state["consumed_ids"])
        return sorted(eligible(issues, self.config["eligibility"]) - processed)

    def claim(self, state: Json, ids: list[str], target: str) -> Json:
        # The caller holds the kernel lock for the ENTIRE claim/audit/publication.
        run: Json | None = state.get("run")
        if run is None:
            run = {"id": uuid.uuid4().hex, "ids": ids, "commit": target, "cutoff": now()}
        run.update(owner=uuid.uuid4().hex, pid=os.getpid(), status="running", claimed_at=now())
        state["run"] = run
        write_json(self.path, state)
        return run

    def owned(self, run: Json) -> Json:
        state = self.load()
        active: Json = state.get("run") or {}
        if active.get("owner") != run["owner"]:
            raise RuntimeError("run ownership lost; publication stopped")
        return state

    def failure(self, run: Json, error: str) -> None:
        state = self.owned(run)
        state["run"]["status"] = "failed"
        state["diagnostic"] = {"at": now(), "error": error}
        write_json(self.path, state)

    def success(self, run: Json, report: Path) -> None:
        state = self.owned(run)
        if not report.is_file():
            raise ValueError("report is missing")
        state["consumed_ids"] = sorted(set(state["consumed_ids"]) | set(run["ids"]))
        state["last_success"] = {
            **run,
            "status": "succeeded",
            "report": str(report),
            "finished_at": now(),
        }
        state["run"] = None
        state["diagnostic"] = None
        write_json(self.path, state)

    def publish(self, run: Json, result: Json, github: GitHub) -> list[str]:
        """Reconcile every intended write with all live markers before retrying."""
        links: list[str] = []
        validate_result(result, self.config)
        findings = result["findings"]
        for finding in findings:
            state = self.owned(run)
            marker = finding_marker(self.config["repository"], finding)
            issues = github.issues()  # Includes earlier ambiguous POSTs and manual equivalents.
            matches = [issue for issue in issues if marker in (issue.get("body") or "")]
            existing = finding["existing_issue"]
            matches += [issue for issue in issues if existing and issue["number"] == existing]
            if existing and not matches:
                raise ValueError(f"unverified existing issue #{existing}")
            old = state["publications"].get(marker)
            if matches:
                issue = next((item for item in matches if item["state"] == "open"), matches[0])
                # Dismissed findings stay suppressed. Resolved findings need explicit
                # recurrence evidence and a new marker generation reviewed by the owner.
                # Conservatively reuse/suppress all matched findings; retain new evidence in report.
            elif old:
                raise ValueError(
                    "ledger issue missing from complete GitHub listing; reconcile first"
                )
            else:
                body = (
                    f"{finding['body']}\n\n{marker}\n<!-- project-hygiene:audit={run['id']} -->\n"
                )
                issue = github.create(finding["title"], body, self.config["issue_labels"])
            self.owned(run)  # Fence before saving the confirmed write.
            state["publications"][marker] = {
                "node_id": issue["node_id"],
                "url": issue["html_url"],
                "number": issue["number"],
                "run": run["id"],
                "state": issue["state"],
            }
            write_json(self.path, state)
            links.append(issue["html_url"])
        return links

    def dispatch(
        self,
        github: GitHub,
        audit: Callable[[Json, Path, list[Json]], Json],
        *,
        manual: bool = False,
        dry_run: bool = False,
    ) -> str:
        if os.environ.get(WORKER_ENV):
            return "audit-worker: suppressed"
        if not manual and not self.config["enabled"]:
            return "disabled"
        # The same lock protects checks and the worker for all of this clone's worktrees.
        with exclusive(self.directory):
            state = self.load()
            issues = github.issues()  # Failure leaves all checkpoint data intact.
            ids = self.pending(state, issues)
            if not manual and not state["run"] and len(ids) < self.config["threshold"]:
                return f"pending: {len(ids)}/{self.config['threshold']}"
            if dry_run:
                return f"would dispatch: {len(ids)} pending; retry={bool(state['run'])}"
            run = self.claim(
                state, ids, state["run"]["commit"] if state["run"] else github.target()
            )
            report_dir = self.directory / "runs" / run["id"]
            report_dir.mkdir(parents=True, exist_ok=True)
            try:
                result_path = report_dir / "result.json"
                # Persist intended publications BEFORE the first POST; failed publication
                # retries the exact report and marker set, not a different model response.
                if result_path.exists():
                    result = read_json(result_path)
                else:
                    result = audit(run, report_dir, issues)
                    validate_result(result, self.config)
                    write_json(result_path, result)
                links = self.publish(run, result, github)
                report = result["report"] + "\n\nIssues:\n" + "\n".join(links) + "\n"
                if len(report.encode()) > MAX_REPORT_BYTES:
                    raise ValueError("report exceeds bounded size")
                path = report_dir / "report.md"
                path.write_text(report, encoding="utf-8")
                self.success(run, path)
                return f"audit succeeded: {path}"
            except Exception as error:
                self.failure(run, str(error))
                raise


def context(root: Path) -> Cycle:
    config = read_json(root / ".codex" / "hygiene.json")
    if config["host"] != "github.com" or config["threshold"] != THRESHOLD:
        raise ValueError("this installation supports github.com and a five-ticket threshold")
    remote = git(root, "remote", "get-url", "origin")
    if remote.removesuffix(".git") not in (
        f"https://github.com/{config['repository']}",
        f"git@github.com:{config['repository']}",
    ):
        raise ValueError("origin does not match configured repository")
    common = Path(git(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    return Cycle(root, config, common / config["state_directory"])


def skill_files() -> dict[str, Path]:
    home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    skill = home / "skills" / "project-hygiene"
    return {
        "SKILL.md": skill / "SKILL.md",
        "cycle-contract.md": skill / "references/cycle-contract.md",
    }


def stop_process(process: subprocess.Popen[str], root: Path) -> None:
    if os.name == "nt":
        command([executable("taskkill"), "/PID", str(process.pid), "/T", "/F"], root)
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=30)


def agent(root: Path, bundle: Path, prompt: str, *, seconds: int, sandbox: str) -> Json:
    env = {**os.environ, WORKER_ENV: "1", "NDE_DASHBOARD": "0"}
    args = [
        executable("codex"),
        "exec",
        "--ephemeral",
        "--sandbox",
        sandbox,
        "--config",
        'approval_policy="never"',
        "--color",
        "never",
        "--json",
        "--cd",
        str(root),
        "--output-schema",
        str(bundle / "schema.json"),
        "--output-last-message",
        str(bundle / "response.json"),
        "-",
    ]
    with (bundle / "agent.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(  # noqa: S603 - native executable and argument array.
            args,
            cwd=root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
            text=True,
            encoding="utf-8",
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        try:
            process.communicate(prompt, timeout=seconds)
        except subprocess.TimeoutExpired:
            stop_process(process, root)
            raise RuntimeError(f"audit exceeded {seconds}s; see {bundle / 'agent.log'}") from None
    if process.returncode:
        raise RuntimeError(f"Codex exited {process.returncode}; see {bundle / 'agent.log'}")
    return read_json(bundle / "response.json")


def prepare_bundle(
    cycle: Cycle,
    directory: Path,
    *,
    python: Path | None = None,
    uv: Path | None = None,
) -> None:
    for name, source in skill_files().items():
        shutil.copyfile(source, directory / name)
    for name in ("prompt.md", "schema.json"):
        shutil.copyfile(cycle.root / "tools" / "hygiene" / name, directory / name)
    write_json(directory / "config.json", cycle.config)
    settings = cycle.directory / "tools.json"
    tools = (
        read_json(settings)
        if settings.exists()
        else {
            "python": executable("python"),
            "uv": executable("uv"),
        }
    )
    if python is not None:
        tools["python"] = str(python.resolve(strict=True))
    if uv is not None:
        tools["uv"] = str(uv.resolve(strict=True))
    write_json(directory / "tools.json", tools)
    write_json(
        directory / "skill-version.json",
        {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in skill_files()
        },
    )


@contextlib.contextmanager
def staged_uv(checkout: Path, directory: Path) -> Generator[None]:
    """Grant executable access through the worker workspace, without changing host ACLs."""
    settings = directory / "tools.json"
    tools = read_json(settings)
    source = Path(tools["uv"])
    target_dir = checkout / ".nde" / "hygiene-tools"
    target_dir.mkdir(parents=True)
    target = target_dir / source.name
    shutil.copyfile(source, target)
    with source.open("rb") as handle:
        original = hashlib.file_digest(handle, "sha256").hexdigest()
    write_json(settings, {**tools, "uv": str(target)})
    try:
        yield
    finally:
        # Remove only the exact staged executable if it has not changed. Preserve
        # agent outputs or a changed binary for diagnosis rather than force-cleaning.
        if target.is_file():
            with target.open("rb") as handle:
                unchanged = hashlib.file_digest(handle, "sha256").hexdigest() == original
            if unchanged:
                target.unlink()
                if not any(target_dir.iterdir()):
                    target_dir.rmdir()
                    if not any(target_dir.parent.iterdir()):
                        target_dir.parent.rmdir()


def isolated_audit(cycle: Cycle, run: Json, directory: Path, issues: list[Json]) -> Json:
    # A checkout inside .git inherits protected-directory restrictions in the
    # Windows sandbox. Keep working files in an ignored, run-specific workspace.
    checkout = cycle.root / ".nde" / "hygiene-checkouts" / run["id"]
    prepare_bundle(cycle, directory)
    write_json(directory / "snapshot.json", {**run, "last_success": cycle.load()["last_success"]})
    write_json(directory / "issues.json", issues)
    # Fetch the explicit default-branch snapshot; never change the active checkout.
    git(cycle.root, "fetch", "--no-tags", "origin", run["commit"])
    if checkout.exists():
        # An interrupted audit's checkout may contain useful diagnostics. Retain it
        # and use a new checkout, rather than resetting or deleting unknown files.
        checkout = checkout.with_name(checkout.name + "-" + uuid.uuid4().hex[:8])
    branch = "codex/hygiene-" + checkout.name
    git(cycle.root, "worktree", "add", "-b", branch, str(checkout), run["commit"])
    try:
        prompt = (directory / "prompt.md").read_text(encoding="utf-8")
        prompt += f"\nAudit inputs and bundled skills are in {directory}.\n"
        with staged_uv(checkout, directory):
            return agent(
                checkout,
                directory,
                prompt,
                seconds=cycle.config["limits"]["seconds"],
                sandbox="workspace-write",
            )
    finally:
        # Remove a clean checkout only. Preserve modifications, untracked AND ignored
        # outputs with the report; never force-delete the worker's unresolved evidence.
        dirty = git(checkout, "status", "--porcelain", "--ignored")
        if dirty:
            (directory / "retained-checkout.txt").write_text(
                f"Retained {checkout}\n{dirty}\n", encoding="utf-8"
            )
        else:
            git(cycle.root, "worktree", "remove", str(checkout))
            try:
                git(cycle.root, "branch", "-d", branch)
            except subprocess.CalledProcessError:
                (directory / "retained-branch.txt").write_text(
                    f"Retained {branch}: normal safe branch deletion refused; "
                    "verify integration before deleting.\n",
                    encoding="utf-8",
                )


def status(cycle: Cycle, github: GitHub) -> Json:
    # Intentionally read-only, including before initial installation.
    state = cycle.load() if cycle.path.exists() else None
    return {
        "enabled": cycle.config["enabled"],
        "state": str(cycle.path),
        "installation": state["installation"] if state else None,
        "last_success": state["last_success"] if state else None,
        "run": state["run"] if state else None,
        "pending_ids": cycle.pending(state, github.issues()) if state else None,
        "diagnostic": state["diagnostic"] if state else None,
    }


def preflight(
    cycle: Cycle,
    github: GitHub,
    *,
    python: Path | None = None,
    uv: Path | None = None,
) -> Json:
    """Real isolated agent/tool/authentication probe; never creates an issue."""
    write_json(cycle.directory / "preflight.json", {"completed": False, "at": now()})
    repo = github.api("")
    if not repo["permissions"].get("push"):
        raise RuntimeError("host GitHub credential lacks repository write permission")
    github.issues()
    directory = cycle.directory / "preflight" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    prepare_bundle(cycle, directory, python=python, uv=uv)
    commit = github.target()
    git(cycle.root, "fetch", "--no-tags", "origin", commit)
    checkout = cycle.root / ".nde" / "hygiene-checkouts" / ("preflight-" + directory.name)
    git(cycle.root, "worktree", "add", "--detach", str(checkout), commit)
    try:
        tools = read_json(directory / "tools.json")
        python_version = command([tools["python"], "--version"], cycle.root)
        uv_version = command([tools["uv"], "--version"], cycle.root)
        project = tomllib.loads((cycle.root / "pyproject.toml").read_text(encoding="utf-8"))
        required = project["tool"]["uv"]["required-version"].removeprefix("==")
        if uv_version.split()[1] != required:
            raise RuntimeError(f"uv must match project pin {required}; pass preflight --uv <path>")
        with staged_uv(checkout, directory):
            result = preflight_agent(directory, checkout)
        # Verify actual successful tool output, not the model's formatting of its summary.
        output = successful_tool_output(directory / "agent.log")
        confirmed = all(value in output for value in (commit, python_version, uv_version))
        if not result["completed"] or result["findings"] or not confirmed:
            raise RuntimeError("runner preflight response did not confirm the snapshot")
        result.update(at=now(), commit=commit, github_write_permission=True)
        write_json(cycle.directory / "tools.json", tools)
        write_json(cycle.directory / "preflight.json", result)
        return result
    finally:
        git(cycle.root, "worktree", "remove", str(checkout))


def successful_tool_output(log: Path) -> str:
    outputs: list[str] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            event: Json = json.loads(line)
        except json.JSONDecodeError:
            continue  # CLI startup diagnostics are interleaved with JSON events.
        item = event.get("item", {})
        if (
            event.get("type") == "item.completed"
            and item.get("type") == "command_execution"
            and item.get("exit_code") == 0
        ):
            outputs.append(item.get("aggregated_output", ""))
    return "\n".join(outputs)


def preflight_agent(directory: Path, checkout: Path) -> Json:
    return agent(
        checkout,
        directory,
        f"This is ONLY an installation preflight, not an audit. Read README.md, "
        f"{directory / 'SKILL.md'} and {directory / 'cycle-contract.md'}. "
        f"Run git rev-parse HEAD, and run --version on the exact Python and uv paths "
        f"in {directory / 'tools.json'} (use PowerShell's & with quoted executable paths). "
        "Return completed=true only if ALL commands succeed, "
        "findings=[] and report containing the observed commit, Python version, "
        "uv version and the three audit areas named in the skill. "
        "Make no file or remote writes.",
        seconds=120,
        sandbox="workspace-write",
    )


def spawn_check(cycle: Cycle) -> None:
    if os.environ.get(WORKER_ENV) or not cycle.config["enabled"]:
        return
    cycle.directory.mkdir(parents=True, exist_ok=True)
    with (cycle.directory / "dispatch.log").open("a", encoding="utf-8") as log:
        subprocess.Popen(  # noqa: S603 - fixed helper path; independent lifecycle, captured output.
            [sys.executable, str(Path(__file__).resolve()), "check", "--root", str(cycle.root)],
            cwd=cycle.root,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=os.name != "nt",
            creationflags=(subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
            if os.name == "nt"
            else 0,
        )


def hook_definition() -> Json:
    # Root resolution works when Codex starts in a repository subdirectory.
    bootstrap = (
        "import pathlib,runpy,subprocess,sys; "
        "p=pathlib.Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],"
        "text=True).strip())/'tools'/'hygiene.py'; "
        "sys.argv=[str(p),'hook']; runpy.run_path(str(p),run_name='__main__')"
    )
    return {
        "matcher": "^(startup|resume)$",
        "hooks": [
            {
                "type": "command",
                "command": f'python3 -c "{bootstrap}"',
                "commandWindows": f'python -c "{bootstrap}"',
                "timeout": 10,
                "statusMessage": "Checking project hygiene cycle",
            }
        ],
    }


def install_hook(root: Path) -> None:
    """Merge one trigger into the existing project representation, preserving other hooks."""
    path = root / ".codex" / "hooks.json"
    toml_path = root / ".codex" / "config.toml"
    inline = tomllib.loads(toml_path.read_text(encoding="utf-8")) if toml_path.exists() else {}
    definition = hook_definition()
    if "hooks" in inline:
        if path.exists():
            raise ValueError("project has two hook representations; reconcile before enabling")
        groups: list[Json] = inline["hooks"].get("SessionStart", [])
        if any(group == definition for group in groups):
            return
        # Append a table only when the existing representation is an array of tables.
        # Inline SessionStart arrays cannot be safely extended without a TOML writer.
        text = toml_path.read_text(encoding="utf-8")
        if groups and "[[hooks.SessionStart]]" not in text:
            raise ValueError(
                "inline hook array needs manual merge; automatic dispatch remains disabled"
            )
        handler = definition["hooks"][0]
        addition = '\n[[hooks.SessionStart]]\nmatcher = "^(startup|resume)$"\n'
        addition += "\n[[hooks.SessionStart.hooks]]\n"
        addition += (
            "\n".join(f"{key} = {json.dumps(value)}" for key, value in handler.items()) + "\n"
        )
        tomllib.loads(text + addition)  # Validate BEFORE replacing a project configuration.
        toml_path.write_text(text + addition, encoding="utf-8")
    else:
        hooks: Json = read_json(path) if path.exists() else {"hooks": {}}
        groups = hooks.setdefault("hooks", {}).setdefault("SessionStart", [])
        if definition not in groups:
            groups.append(definition)
        write_json(path, hooks)


def enable(cycle: Cycle, github: GitHub) -> None:
    with exclusive(cycle.directory):
        cycle.load()  # A baseline and unchanged eligibility are required.
        proof = read_json(cycle.directory / "preflight.json")
        if not proof.get("completed") or not proof.get("github_write_permission"):
            raise RuntimeError("run a successful preflight first")
        # Label creation is the only remote setup write. No test issue is created.
        labels = github.api("/labels?per_page=100")
        if not any(label["name"] == "project-hygiene" for label in labels):
            github.api(
                "/labels",
                payload={
                    "name": "project-hygiene",
                    "color": "5319e7",
                    "description": "Evidence-backed documentation, test and onboarding maintenance",
                },
            )
        install_hook(cycle.root)
        cycle.config["enabled"] = True
        write_json(cycle.root / ".codex" / "hygiene.json", cycle.config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("status", "install", "preflight", "enable", "hook", "check", "audit")
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--python", type=Path, help="preflight: native Python usable in runner sandbox"
    )
    parser.add_argument("--uv", type=Path, help="preflight: repository-pinned uv executable")
    args = parser.parse_args()
    try:
        if os.environ.get(WORKER_ENV) and args.action in ("hook", "check", "audit"):
            return 0
        cycle = context(args.root.resolve())
        github = GitHub(cycle.root, cycle.config["repository"])
        if args.action == "hook":
            spawn_check(cycle)
        elif args.action == "status":
            print(json.dumps(status(cycle, github), indent=2))
        elif args.action == "install":
            with exclusive(cycle.directory):
                state = cycle.install(github.issues(), github.target())
            print(json.dumps(state["installation"], indent=2))
        elif args.action == "preflight":
            print(json.dumps(preflight(cycle, github, python=args.python, uv=args.uv), indent=2))
        elif args.action == "enable":
            enable(cycle, github)
            print("Dispatch enabled; review and trust the exact SessionStart hook in Codex /hooks.")
        else:
            print(
                cycle.dispatch(
                    github,
                    lambda run, directory, issues: isolated_audit(cycle, run, directory, issues),
                    manual=args.action == "audit",
                    dry_run=args.dry_run,
                )
            )
        return 0
    except Exception as error:
        # Background failure never blocks ordinary work. Both hooks and operators
        # get an actionable message; only explicit operator commands fail.
        print(f"project-hygiene: {error}; checkpoint retained; retry with check", file=sys.stderr)
        return 0 if args.action == "hook" else 1


if __name__ == "__main__":
    raise SystemExit(main())
