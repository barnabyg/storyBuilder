# Project hygiene cycle

The local cycle checks at Codex session startup/resume and dispatches one
background audit after five new implementation tickets. Configuration is in
[`.codex/hygiene.json`](../../.codex/hygiene.json); the independent Python helper is
[`tools/hygiene.py`](../../tools/hygiene.py). It requires Python 3.12+, Git, native
`codex` and `gh` executables, a Codex login, and GitHub repository write access
through the host keyring. Desktop connector authentication is separate.

Completed work means a GitHub issue closed with reason `completed`, labelled
`ready-for-agent`, and outside the configured hygiene/control/cancelled/duplicate
exclusions. Owner-only `ready-for-human` issues do not count. Work may reach `main`
directly or through a PR, so a linked merged PR is not required. Immutable
issue IDs prevent counting reopened/reclosed work twice. All API pages are read;
an outage preserves the checkpoint and leaves a diagnostic.

The installation baseline is the remote default-branch commit plus all eligible
IDs at installation. It is separate from a successful audit. The first automatic
audit follows the **next** five completions. Setup reruns preserve the baseline,
pending snapshot and publication ledger; changing eligibility requires explicit
baseline migration rather than reinstalling over the state.

## Operation

Run from the repository root (the helper itself also supports `--root`):

```powershell
python tools/hygiene.py status
python tools/hygiene.py check --dry-run
python tools/hygiene.py audit
```

`status` reads live GitHub state without writing local state. `check --dry-run`
simulates threshold/retry selection without consuming tickets or launching an
agent. `audit` explicitly requests an immediate audit, even below the threshold
or with automatic dispatch disabled. It can publish up to three hygiene issues;
it never fixes application code. `check` retries a failed captured run before
considering a new snapshot. The saved model response and publication ledger make
partially published runs retryable without creating the same issues twice.

Set `enabled` to `false` in `.codex/hygiene.json` to disable automatic dispatch;
this preserves pending work. To restore it after reviewing the installation:

```powershell
python tools/hygiene.py preflight
python tools/hygiene.py enable
```

Review and trust the exact project SessionStart definition in Codex `/hooks`.
Codex skips untrusted or changed hooks; see the [official hook documentation](https://learn.chatgpt.com/docs/hooks).
The project trigger matches only `startup` and `resume`, runs a fast launcher,
and suppresses itself in audit workers. Project hooks require a trusted project
configuration layer and local orchestration. Cloud-orchestrated chats do not run
local command hooks. There is no second CI or recurring schedule trigger.

For a new local clone, run `install`, `preflight`, then `enable` after reviewing
the configuration and running verification. These commands use the host credential
store and need permission to write Git metadata and project `.codex` files.
`enable` creates the `project-hygiene` label if missing and reuses `needs-triage`.
It does not create a test issue or mark a preflight as an audit.

If a Windows sandbox cannot launch the system/virtual-environment Python, pass
`preflight --python <native-python-path>` with a sandbox-compatible interpreter.
The verified paths are saved in local runtime state, never committed. This
installation uses the bundled Codex Python runtime and a cached repository-pinned
uv executable. If the global uv differs from the project pin, pass
`preflight --uv <pinned-uv-path>`; preflight rejects a mismatched version. A moved
or removed runtime needs a new preflight. The helper stages a copy of the pinned
uv executable inside each audit checkout so the Windows sandbox can launch it,
then removes that exact copy during cleanup. Host ACLs and tools stay unchanged.

## Runtime, isolation and limits

State is stored at `<git-common-dir>/project-hygiene/state.json`; `git rev-parse
--path-format=absolute --git-common-dir` locates it. The Git directory already
excludes it from tracked files. Atomic writes and a kernel lock shared across
all this clone's worktrees serialize the **whole** claim/audit/publication run.
The lock releases on process exit; retry claims a saved run under a new owner
token, fencing obsolete finalization. This is a **single-machine** installation.
Use a shared state backend before adding another machine or CI trigger.

`dispatch.log` captures background errors. `runs/<run-id>/` contains the captured
commit/tickets/cutoff, versioned copies of the installed personal audit skill and
contract, agent log, result, and final report. Skill criteria remain in the
personal `project-hygiene` skill; this repository holds project facts and machinery.
No credentials are stored in configuration or reports.

The supervisor creates a `codex/hygiene-*` branch-backed checkout of the recorded default-branch commit
under `.nde/hygiene-checkouts/<run-id>`, outside protected Git metadata. It sets a
recursion guard before starting Codex with workspace writes and approval policy
`never`. GitHub publication is performed by the host helper, scoped to this
repository, after checking existing open and closed issues. Markers are stable
across runs and commits. Manual equivalents are reused and dismissed findings
remain suppressed. New recurrence evidence for a resolved finding is retained
in the report for owner triage rather than automatically opening another issue.

The agent has a 20-minute budget. A timeout terminates its process tree. Clean
worktrees are removed; changed, untracked or ignored diagnostic outputs are
deliberately retained with `retained-checkout.txt` in the run report directory.
Clean audit branches use normal safe deletion; a refusal retains the branch and
records the reason in `retained-branch.txt`. Read-only preflight checkouts are detached.
Inspect and preserve useful output before removing a retained checkout with
`git worktree remove`; never force-clean another active audit. If the supervisor
itself was forcibly stopped, inspect its logged child process before cleaning
the recorded checkout. Reports and unresolved snapshots stay available for retry.

Zero findings can succeed. Success consumes exactly the captured IDs, and only
after the report and all intended issue writes are saved. Later completions stay
pending. Findings beyond the cap and unexamined areas belong in the report.

## Verification

`uv run --locked pytest tests/test_hygiene.py` uses temporary state and fake
GitHub/runner boundaries. It covers threshold/coalescing, exclusions, reclosures,
worktree locking and recovery, recursion suppression, reruns, API and runner
failures, incomplete audits, ambiguous publication, manual/dismissed equivalents,
and exact snapshot consumption. It never closes live issues.

`uv run --locked nde-verify --no-dashboard` is the canonical full verification.
The live preflight separately verifies authentication, a clean remote snapshot,
bundled instruction loading and noninteractive sandbox tool execution without
publishing issues. Neither fixture tests nor that preflight constitute an actual
end-to-end audit/publication run. No initial audit is requested by installation.

The helper and its boundary tests have narrowly scoped Ruff `S603` exceptions
for subprocess calls. Each call passes a constrained native executable and an
argument array, with a local justification. Review these exceptions if executable
selection, arguments or shell interpretation change; no shell string is executed
by those calls.
