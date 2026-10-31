# Verification

`uv run nde-verify` is the canonical, non-mutating verification command. It
runs the seven gates in order (see the [README](../README.md#verification))
and exits non-zero if any gate fails or prints a warning.

## Quick check before pushing

`uv run nde-verify --quick` runs every gate except the tests. The tests can take many
minutes on Windows, and a full run cut short there never reaches the gates after them.
Run the quick check, plus the tests your change affects, before pushing. It is not full
verification and says so when it ends. CI always runs everything.

Gates 6 and 7 need the `uv` version pinned in `pyproject.toml`. If the `uv` on your PATH
is a different version, set the `UV` environment variable to the pinned one's path. In
gate 6 the secret scan runs first, so a wrong `uv` version cannot hide a leaked secret.

## Live dashboard

An interactive local run also starts a live dashboard and prints its address
first:

```text
TEST_DASHBOARD_URL=http://127.0.0.1:8765/
```

Open it to watch the active gate, per-test progress (done, passed, failed,
skipped and the test now running), elapsed time, recent output, failures and
the final result. The page polls `/api/state` twice a second.

The dashboard only observes. Gate order, gate output and the exit status are
identical with and without it; the `TEST_DASHBOARD_URL` line is the only
addition to the output. If the dashboard cannot start, `nde-verify` prints a
note to stderr and runs without it. If it fails mid-run, the gates carry on
unchanged and a note on stderr says the dashboard stopped updating.

### URL and port

- The server listens on `127.0.0.1` only, on port **8765** by default
  (`NDE_DASHBOARD_PORT` changes it).
- If that port is taken, for example by a second verification running at the
  same time, the run uses a free port instead and prints that URL. Every run
  prints the URL it actually uses.
- After the gates finish, the server stays up for up to 3 seconds, until an
  open page has fetched the final state, then exits. The page keeps showing
  the final result after the server has gone.

### Turning it off

The dashboard does not start when any of these hold:

- `uv run nde-verify --no-dashboard`
- the environment variable `NDE_DASHBOARD=0`
- the environment variable `CI` is set (GitHub Actions sets it), so CI stays
  headless. CI produces no static report; the job log is the record.

### Test-runner integration

Per-test progress comes from pytest's own lifecycle hooks. The tests gate
always runs `pytest -p nde.verify.pytest_reporter`, so its command line is the
same with and without the dashboard. The plugin (`src/nde/verify/pytest_reporter.py`)
is inert unless `nde-verify` passes it `NDE_DASHBOARD_EVENTS_URL`, the
dashboard's `http://127.0.0.1:<port>/api/events` endpoint. It then posts:

| Event | Hook |
| --- | --- |
| `tests_collected` | `pytest_collection_finish` |
| `test_started` | `pytest_runtest_logstart` |
| `test_finished` (passed, failed or skipped) | `pytest_runtest_logreport` |

The plugin switches itself off at the first failed post, so a dashboard that
has gone away never slows down or breaks the tests. Gate-level events
(`run_started`, `gate_started`, `output`, `gate_finished`, `run_finished`)
come from the gate runner through an observer whose exceptions are contained
(`src/nde/verify/runner.py`).

### Endpoints

| Method and path | Purpose |
| --- | --- |
| `GET /` | the dashboard page |
| `GET /api/state` | the current snapshot as JSON |
| `POST /api/events` | one JSON event; 204 when applied, 400 when malformed or unknown |
