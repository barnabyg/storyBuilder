"""The G-E decision through public commands: a tiny CPU checkpoint and fake providers.

Every protocol, run and review here is synthetic and explicitly test-only, so every
decision written by a public command stays scientifically pending. Genuine outcomes
are exercised only by promoting a real decision's content in memory.
"""

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from nde.conditional import ge, ge_decision, ge_review
from nde.corpus.records_io import create_record, read_sealed
from nde.runs.runner import RunOptions
from tests.fake_openai import FakeOpenAI
from tests.test_g1 import git_fixture
from tests.test_ge import (
    QUESTION,
    Fixture,
    GEProvider,
    ge_command,
    prepared,
    registration,
    restored,
)

# E's brief-macro and pooled rates are 0.5 and the reference's 1.0, so a margin and
# minimum of 0.5 sit exactly on both validator boundaries. Seed 17 samples one owner
# pair and one slot where E has no plan, so equivalent ratings give a share of 0.5.
BOUNDARY: dict[str, Any] = {
    "validator_comparison": {
        "analysis": "pass_rate_difference",
        "margin": 0.5,
        "minimum_candidate_rate": 0.5,
        "aggregation": "brief_macro",
    },
    "structural_criterion": {
        "analysis": "owner_paired_competitive_share",
        "question": QUESTION,
        "minimum_share": 0.5,
    },
    "owner_review": {
        "reviewer": "Synthetic owner",
        "sample_size": 2,
        "sample_seed": 17,
        "minutes_per_item": 10,
        "max_minutes": 20,
    },
}
RULE = {"aggregation": "pooled", "margin": 0.5, "minimum_candidate_rate": 0.5}
SHARE = {"question": QUESTION, "minimum_share": 0.5}


@pytest.fixture(scope="module")
def registered(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return prepared(tmp_path_factory.mktemp("ge-decision"))


def commit(repo: Path, *names: str) -> None:
    git_fixture(repo, "add", *names)
    git_fixture(repo, "commit", "-q", "-m", "synthetic test-only G-E evidence")


def new_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    git_fixture(path, "init", "-q")
    return path


def attempt(
    repo: Path,
    fixture: Fixture,
    name: str,
    *,
    source: Path | None,
    claim: bool = True,
    **changes: Any,
) -> tuple[Path, Path]:
    """Register, commit and run a protocol; `source` stories are copied instead of generated."""
    folder = repo / name
    path = registration(folder, fixture, id=f"synthetic-test-only-ge-{name}", **changes)
    protocol = folder / "protocol.json"
    assert ge_command("register", path, "--out", protocol) == 0
    commit(repo, f"{name}/protocol.json")
    runs = repo.parent / f"runs-{repo.name}-{name}"

    def generate(_: object, options: RunOptions) -> Path:
        assert source is not None
        run = options.runs_dir / "copied"
        ignored = shutil.ignore_patterns("ge-run.json", "ge-run.md", "ge-evidence")
        shutil.copytree(source, run, ignore=ignored)
        return run

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("FAKE_API_KEY", "test-key")
        if source is not None:
            patch.setattr(ge, "execute", generate)
        assert ge_command("run", protocol, "--runs-dir", runs) == 0
    if claim:
        commit(repo, f"{name}/protocol.run-claim.json")
    [run] = list(runs.iterdir())
    return protocol, run


def packet(protocol: Path, run: Path, folder: Path) -> Path:
    assert ge_command("packet", protocol, run, "--out", folder) == 0
    return folder


def review(folder: Path, prefer: str, **fields: Any) -> None:
    """A synthetic owner review of every owner item, preferring E, the reference or neither."""
    value = ge_review.read_packet(folder)
    key = read_sealed(folder / "key.json")["items"]
    ratings = [
        {
            "item": item["item"],
            "plans": {label: plan["token"] for label, plan in item["plans"].items()},
            "rating": prefer
            if prefer == "equivalent"
            else next(label for label, side in key[item["item"]].items() if side == prefer),
            "note": "Synthetic fixture note.",
        }
        for item in value["items"]
        if item["kind"] == "owner"
    ]
    path = folder.with_name(f"{folder.name}-review-{len(list(folder.parent.iterdir()))}.json")
    review_value = {"packet_id": value["id"], "reviewer": "Synthetic owner", "ratings": ratings}
    path.write_text(json.dumps(review_value | fields))
    # A synthetic review finishes the mechanics but never completes the owner evidence.
    assert ge_command("review", folder, path) == 1


def decide(
    protocol: Path, run: Path, out: Path, review_packet: Path | None = None
) -> tuple[int, dict[str, Any]]:
    extra = [] if review_packet is None else ["--packet", review_packet]
    code = ge_command("decide", protocol, run, "--out", out, *extra)
    return code, read_sealed(out / "decision.json")


def preserved(record: dict[str, Any], protocol: Path, run: Path) -> None:
    """Evidence, costs, failures and identities are carried over from the verified run."""
    saved = ge.read_run(run)
    assert record["evidence"] == saved["evidence"]
    assert record["costs"] == saved["costs"]
    assert record["failures"] == saved["failures"]
    assert record["units"] == saved["units"]
    assert record["run"]["record_id"] == saved["id"]
    assert record["run"]["claim_id"] == read_sealed(ge.claim_path(protocol))["id"]
    assert record["judges"] == "diagnostic_only"


def no_launch(patch: pytest.MonkeyPatch) -> None:
    """Fail the test if reporting generates, builds evidence or trains anything."""

    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("reporting must not launch anything")

    patch.setattr(ge, "execute", refuse)
    patch.setattr(ge, "prepare", refuse)
    patch.setattr(ge.evidence_report, "write", refuse)


@pytest.mark.ml
def test_a_decision_waits_for_a_recorded_run_and_a_current_owner_review(
    tmp_path: Path, registered: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = new_repo(tmp_path / "a")
    protocol, run = attempt(
        repo, registered, "first", source=registered.run, claim=False, **BOUNDARY
    )
    no_launch(monkeypatch)
    runs = sorted(run.parent.iterdir())
    # The run must be recorded in the repository's history before it can be decided.
    code, record = decide(protocol, run, repo / "d-uncommitted")
    assert (code, record["outcome"], record["criteria_outcome"]) == (1, "inconsistent", "pending")
    assert any("run claim is not committed" in problem for problem in record["problems"])
    commit(repo, "first/protocol.run-claim.json")
    # Without the owner's packet the structural criterion is pending; the validator
    # comparison is measured at its inclusive boundaries.
    code, record = decide(protocol, run, repo / "d-no-packet")
    assert (code, record["outcome"], record["criteria_outcome"]) == (1, "pending", "pending")
    validator = record["validator"]
    assert validator["status"] == "passed"
    assert validator["difference"] == validator["lower_bound"] == -0.5
    assert validator["candidate_rate"] == validator["minimum_candidate_rate"] == 0.5
    assert validator["counts"]["E"]["no-output"] == 2
    assert validator["counts"]["E"]["expected"] == 4
    assert record["structural"]["reason"] == "no owner review packet was supplied"
    assert not ge_decision.settlement_path(protocol).exists()
    folder = packet(protocol, run, repo / "packet")
    code, record = decide(protocol, run, repo / "d-unreviewed", folder)
    assert record["outcome"] == "pending"
    assert "await the registered reviewer's actual ratings" in record["structural"]["reason"]
    # A packet relabelled as real is not this run's packet; a changed plan makes it stale.
    real = repo / "relabelled"
    real.mkdir()
    shutil.copy2(folder / "key.json", real / "key.json")
    value = copy.deepcopy(read_sealed(folder / "packet.json"))
    value["test_only"] = False
    value["run"] = ge.relative(run.resolve(), real.resolve())
    del value["id"], value["version"]
    create_record(real / "packet.json", value)
    _, record = decide(protocol, run, repo / "d-relabelled", real)
    assert record["outcome"] == "inconsistent"
    assert "the review packet's test-only status differs from its run" in record["problems"]
    plan = run / "stories/E/brief-01/r1/plan.json"
    with restored(plan):
        plan.write_text(plan.read_text() + " ")
        _, record = decide(protocol, run, repo / "d-stale", folder)
    assert record["outcome"] == "inconsistent"
    assert any(problem.startswith("review packet stale") for problem in record["problems"])
    assert any("stories/E/brief-01/r1/plan.json" in problem for problem in record["problems"])
    assert not ge_decision.settlement_path(protocol).exists()
    assert sorted(run.parent.iterdir()) == runs


@pytest.mark.ml
def test_complete_synthetic_evidence_meets_inclusive_boundaries_settles_and_stays_pending(
    tmp_path: Path,
    registered: Fixture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = new_repo(tmp_path / "a")
    protocol, run = attempt(repo, registered, "first", source=registered.run, **BOUNDARY)
    no_launch(monkeypatch)
    runs = sorted(run.parent.iterdir())
    folder = packet(protocol, run, repo / "packet")
    # Complete synthetic evidence applies both criteria, yet stays scientifically pending.
    review(folder, "equivalent")
    out = repo / "decision"
    code, record = decide(protocol, run, out, folder)
    assert code == 1
    assert (record["criteria_outcome"], record["outcome"]) == ("passed", "pending")
    assert ge_decision.SYNTHETIC in record["pending"]
    structural = record["structural"]
    assert (structural["competitive"], structural["not_competitive"]) == (1, 1)
    assert structural["share"] == structural["minimum_share"] == 0.5
    assert structural["status"] == "passed"
    assert record["owner_review"]["evidence"] == "synthetic"
    assert [r["classification"] for r in record["owner_review"]["reviews"]] == ["synthetic"]
    assert record["action"]["result"] == "none"
    assert record["action"]["phase_8"] == "not_eligible"
    assert record["action"]["reference_planner"] == {
        "condition": "C",
        "identification": "owner C selection",
        "status": "retained",
    }
    preserved(record, protocol, run)
    assert record["g3"]["status"] == "test-only"
    assert record["history"]["first_attempt"] is None
    markdown = (out / "decision.md").read_text()
    assert markdown.startswith("# G-E decision: pending (test-only)")
    assert "## Validator comparison (E minus reference)" in markdown
    # Location-independent: the sealed decision names no absolute path.
    text = (out / "decision.json").read_text()
    assert str(tmp_path) not in text
    assert tmp_path.as_posix() not in text
    # The complete criteria result settles the protocol; it replays and is never replaced.
    settlement = read_sealed(ge_decision.settlement_path(protocol))
    assert settlement["decision_id"] == record["id"]
    assert settlement["criteria_outcome"] == "passed"
    # The owner ratings it applied are frozen beside the protocol, to be committed.
    frozen = read_sealed(ge_decision.ratings_path(protocol))
    assert frozen["reviews"] == {
        row["file"]: row["id"] for row in record["owner_review"]["reviews"]
    }
    assert frozen["packet_id"] == record["owner_review"]["packet"]["id"]
    capsys.readouterr()
    assert ge_command("replay", out) == 1
    assert "frozen owner ratings are not committed" in capsys.readouterr().err
    commit(repo, "first/protocol.review-claim.json", "first/protocol.decision-claim.json")
    assert ge_command("replay", out) == 0
    assert ge_command("decide", protocol, run, "--packet", folder, "--out", repo / "again") == 2
    assert not (repo / "again").exists()
    # A later superseding review is never applied; a changed plan makes the decision stale.
    review(folder, "C")
    assert ge_command("replay", out) == 0
    plan = run / "stories/E/brief-01/r1/plan.json"
    with restored(plan):
        plan.write_text(plan.read_text() + " ")
        assert ge_command("replay", out) == 1
    assert sorted(run.parent.iterdir()) == runs


@pytest.mark.ml
def test_inconsistent_evidence_never_unblinds_and_the_first_applied_ratings_are_frozen(
    tmp_path: Path, registered: Fixture
) -> None:
    repo = new_repo(tmp_path / "a")
    protocol, run = attempt(
        repo, registered, "first", source=registered.run, claim=False, **BOUNDARY
    )
    folder = packet(protocol, run, repo / "packet")
    review(folder, "C")
    # Complete ratings on inconsistent evidence (an uncommitted run claim) stay blinded.
    _, record = decide(protocol, run, repo / "d-inconsistent", folder)
    assert record["outcome"] == "inconsistent"
    structural = record["structural"]
    assert (structural["status"], structural["reason"]) == ("pending", ge_decision.WITHHELD)
    assert (structural["items"], structural["share"], structural["competitive"]) == ([], None, 0)
    assert not ge_decision.ratings_path(protocol).exists()
    # Nothing was unblinded, so the owner may still revise; the first application freezes.
    review(folder, "equivalent")
    commit(repo, "first/protocol.run-claim.json")
    _, record = decide(protocol, run, repo / "d-applied", folder)
    assert record["criteria_outcome"] == "passed"
    used = {row["file"]: row["id"] for row in record["owner_review"]["reviews"]}
    assert len(used) == 2
    assert read_sealed(ge_decision.ratings_path(protocol))["reviews"] == used
    commit(repo, "first/protocol.review-claim.json")
    # Even without the (uncommitted) settlement, a later decision applies only the frozen
    # ratings: re-rating after the outcomes were shown changes nothing.
    ge_decision.settlement_path(protocol).unlink()
    review(folder, "C")
    _, again = decide(protocol, run, repo / "d-again", folder)
    assert {row["file"]: row["id"] for row in again["owner_review"]["reviews"]} == used

    assert (again["criteria_outcome"], again["structural"]["share"]) == ("passed", 0.5)


def content(
    validator: dict[str, Any], structural: dict[str, Any], **changes: Any
) -> dict[str, Any]:
    """Decision content as `assess` builds it, for known-answer outcome rules."""
    value: dict[str, Any] = {
        "validator": validator,
        "structural": structural,
        "problems": [],
        "test_only": False,
        "owner_review": {"evidence": "actual"},
        "protocol": {"attempt": 1},
        "reference": {"condition": "D", "identification": "replayed actual G-D pass"},
    }
    return value | changes


def side(rate: float | None, unknown: int = 0) -> dict[str, Any]:
    return {"rate": rate, "counts": {"expected": 4, "unknown": unknown}}


def measured(
    candidate: float, reference: float, unknown: list[str] | None = None
) -> dict[str, Any]:
    return {
        "aggregation": "pooled",
        "candidate": side(candidate, len(unknown or [])),
        "reference": side(reference),
        "complete": not unknown,
        "unknown_units": unknown or [],
    }


def outcomes(*values: str) -> list[dict[str, Any]]:
    return [
        {"item": f"item-{n:02d}", "slot": f"brief-01/r{n}", "kind": "owner", "outcome": value}
        for n, value in enumerate(values, 1)
    ]


def test_criteria_known_answers_at_threshold_boundaries() -> None:
    criterion = ge_decision.validator_criterion
    # Inclusive margin and minimum, in E's direction.
    assert criterion(measured(0.5, 1.0), RULE)["status"] == "passed"
    assert criterion(measured(0.5, 1.0), RULE | {"margin": 0.49})["meets_margin"] is False
    assert criterion(measured(0.5, 1.0), RULE | {"minimum_candidate_rate": 0.51})["status"] == (
        "failed"
    )
    # A better reference is never read as E's advantage.
    assert criterion(measured(1.0, 0.5), RULE)["difference"] == 0.5
    assert criterion(measured(0.4, 0.9), RULE)["status"] == "failed"
    # Floating-point rates on the boundary meet it: 0.7 - 0.8 is not exactly -0.1.
    assert criterion(measured(0.7, 0.8), RULE | {"margin": 0.1})["status"] == "passed"
    # Unknown units, another aggregation and a missing run stay pending.
    pending = criterion(measured(0.5, 1.0, ["E/brief-01/r1"]), RULE)
    assert (pending["status"], pending["unknown_units"]) == ("pending", ["E/brief-01/r1"])
    assert pending["counts"]["E"]["unknown"] == 1
    assert criterion(measured(0.5, 1.0), RULE | {"aggregation": "brief_macro"})["status"] == (
        "pending"
    )
    assert criterion(None, RULE)["status"] == "pending"

    structural = ge_decision.structural_criterion
    assert structural(outcomes("competitive", "not_competitive"), SHARE, "")["status"] == "passed"
    assert (
        structural(outcomes("competitive", "not_competitive"), SHARE | {"minimum_share": 0.51}, "")[
            "status"
        ]
        == "failed"
    )
    # Excluded slots are reported and leave the denominator; pending items and no pair wait.
    excluded = structural(outcomes("competitive", "excluded"), SHARE, "")
    assert (excluded["denominator"], excluded["excluded"], excluded["share"]) == (
        1,
        ["brief-01/r2"],
        1.0,
    )
    assert structural(outcomes("excluded"), SHARE, "")["status"] == "pending"
    assert structural(outcomes("competitive", "pending"), SHARE, "")["status"] == "pending"
    assert structural(None, SHARE, "no review")["reason"] == "no review"


def test_genuine_outcomes_retain_the_reference_and_a_pass_only_permits_preparing_phase_8() -> None:
    """Fixtures cannot hold actual owner evidence; the outcome rules are promoted in memory."""
    passed = ge_decision.validator_criterion(measured(0.5, 1.0), RULE)
    failed = ge_decision.validator_criterion(measured(0.4, 1.0), RULE)
    share = ge_decision.structural_criterion(outcomes("competitive", "competitive"), SHARE, "")
    waiting = ge_decision.structural_criterion(None, SHARE, "owner review missing")

    result = ge_decision.conclude(content(passed, share))
    assert (result["criteria_outcome"], result["outcome"]) == ("passed", "passed")
    assert result["action"]["result"] == "eligible_to_prepare_phase_8"
    assert result["action"]["phase_8"] == "eligible_to_prepare"
    assert "not proof that E improves final stories" in result["action"]["statement"]
    assert result["action"]["reference_planner"]["status"] == "retained"
    assert result["action"]["automatic"] == ge_decision.NOTHING_LAUNCHED
    # A failure keeps the identified reference and permits only the single iteration.
    result = ge_decision.conclude(content(failed, share))
    assert (result["outcome"], result["action"]["result"]) == ("failed", "iterate_once_or_stop")
    assert result["action"]["reference_planner"]["condition"] == "D"
    assert result["action"]["phase_8"] == "not_eligible"
    result = ge_decision.conclude(content(failed, share, protocol={"attempt": 2}))
    assert (result["outcome"], result["action"]["result"]) == ("failed", "stop")
    assert "Stop: keep the identified D reference planner" in result["action"]["statement"]
    # A known failure still waits for every required criterion; nothing is decided early.
    result = ge_decision.conclude(content(failed, waiting))
    assert (result["criteria_outcome"], result["outcome"]) == ("pending", "pending")
    # Synthetic evidence applies the criteria but stays pending; problems decide nothing.
    for synthetic in ({"test_only": True}, {"owner_review": {"evidence": "synthetic"}}):
        result = ge_decision.conclude(content(passed, share, **synthetic))
        assert (result["criteria_outcome"], result["outcome"]) == ("passed", "pending")
        assert result["action"]["result"] == "none"
    result = ge_decision.conclude(content(passed, share, problems=["stale"]))
    assert (result["criteria_outcome"], result["outcome"]) == ("pending", "inconsistent")


def protocol_record(
    name: str, briefs: list[str], *, selection: str = "selection-1", checksum: str = "a" * 64
) -> dict[str, Any]:
    """The parts of a run C-reference protocol that the history compares."""
    reference = {
        "condition": "C",
        "pipeline": "baseline_c",
        "params": {"tier": "M"},
        "identification": "owner C selection",
        "reason": "no G-D decision registered",
        "gd_decision": None,
        "c_selection": {
            "path": f"../{name}/selection.json",
            "sha256": checksum,
            "id": selection,
            "tier": "M",
            "test_only": False,
        },
    }
    return {
        "id": name,
        "test_only": False,
        "frozen": {
            "reference": reference,
            "cohort": {"briefs": dict.fromkeys(briefs, "b" * 64)},
            "iteration": {"attempt": 1, "previous": None},
        },
    }


def test_comparisons_are_keyed_on_the_canonical_reference_identity() -> None:
    first = protocol_record("first", ["brief-01", "brief-02"])
    found = ge_decision.History(protocols={"first": first}, claims={"first"})
    # Another run first attempt against the same reference is a reset, whatever its dev
    # briefs, and whatever bytes or path hold the same owner C selection.
    for other in (
        protocol_record("disjoint", ["brief-03"]),
        protocol_record("reserialised", ["brief-01"], checksum="c" * 64),
    ):
        found.protocols[other["id"]] = other
        found.claims.add(other["id"])
        [problem] = ge_decision.history_problems(other, other["id"], found)
        assert problem.startswith("history reset")
        assert problem.endswith(": first")
        del found.protocols[other["id"]]
    # Another selection, a test-only comparison or an unrun registration is not this one.
    other = protocol_record("reselected", ["brief-01"], selection="selection-2")
    unrun = protocol_record("unrun", ["brief-01"])
    synthetic = protocol_record("synthetic", ["brief-01"]) | {"test_only": True}
    found.protocols |= {"reselected": other, "unrun": unrun, "synthetic": synthetic}
    found.claims |= {"reselected", "synthetic"}
    assert ge_decision.history_problems(first, "first", found) == []


def iteration_config(folder: Path, fixture: Fixture, fake: FakeOpenAI) -> None:
    """E with another guidance scale, generated by the running fake provider."""
    config = yaml.safe_load(fixture.config.read_text())
    config["providers"]["fake"]["base_url"] = fake.base_url
    config["stories"]["conditions"][0]["params"]["guidance_scale"] = 1.5
    folder.mkdir(parents=True)
    (folder / "ge.yaml").write_text(yaml.safe_dump(config))


@pytest.mark.ml
def test_the_single_iteration_is_bound_to_the_settled_first_attempt_and_history_cannot_reset(
    tmp_path: Path, registered: Fixture, fake_llm: FakeOpenAI
) -> None:
    repo = new_repo(tmp_path / "b")
    first, first_run = attempt(repo, registered, "first", source=registered.run)
    first_packet = packet(first, first_run, repo / "first-packet")
    review(first_packet, "C")
    fake_llm.default = GEProvider()
    iteration_config(repo / "second", registered, fake_llm)
    bound = {"attempt": 2, "previous_protocol": "../first/protocol.json", "change": "Guidance."}
    second, second_run = attempt(
        repo, registered, "second", source=None, config="ge.yaml", iteration=bound
    )
    second_packet = packet(second, second_run, repo / "second-packet")
    review(second_packet, "C")
    # The iteration is decided only against the first attempt's settled decision, and
    # only with its own packet.
    code, record = decide(second, second_run, repo / "d-early", first_packet)
    assert (code, record["outcome"]) == (1, "inconsistent")
    assert any("has no settled decision" in problem for problem in record["problems"])
    assert "the review packet belongs to another protocol" in record["problems"]
    # The first attempt fails both criteria; test-only, so its outcome stays pending.
    _, decided = decide(first, first_run, repo / "first-decision", first_packet)
    assert (decided["criteria_outcome"], decided["outcome"]) == ("failed", "pending")
    assert decided["validator"]["meets_margin"] is False
    assert decided["structural"]["share"] == 0
    commit(
        repo,
        "first/protocol.decision-claim.json",
        "first/protocol.review-claim.json",
        "first-decision",
    )
    out = repo / "second-decision"
    code, record = decide(second, second_run, out, second_packet)
    assert code == 1
    assert record["problems"] == []
    assert record["criteria_outcome"] == "failed"
    first_attempt = record["history"]["first_attempt"]
    assert first_attempt["decision_id"] == decided["id"]
    assert first_attempt["protocol_id"] == read_sealed(first)["id"]
    assert first_attempt["claim_id"] == read_sealed(ge.claim_path(first))["id"]
    assert first_attempt["decision"] == "../first-decision"
    commit(
        repo,
        "second/protocol.decision-claim.json",
        "second/protocol.review-claim.json",
        "second-decision",
    )
    # Promoted in memory, the failed iteration stops and keeps the identified reference.
    promoted = record | {"test_only": False, "owner_review": {"evidence": "actual"}}
    result = ge_decision.conclude(promoted)
    assert (result["outcome"], result["action"]["result"]) == ("failed", "stop")
    assert result["action"]["reference_planner"]["condition"] == "C"
    # Replaying the iteration replays the first attempt's decision too.
    assert ge_command("replay", out) == 0
    # A second iteration of the same first attempt is run: every decision in it is invalid.
    shutil.copytree(repo / "second", repo / "third", ignore=shutil.ignore_patterns("protocol*"))
    again = bound | {"change": "Another guidance scale."}
    third, third_run = attempt(
        repo, registered, "third", source=second_run, config="ge.yaml", iteration=again
    )
    _, record = decide(third, third_run, repo / "third-decision")
    assert any("more than one development iteration" in p for p in record["problems"])
    assert ge_command("replay", out) == 1
    # Deleting the first attempt and registering a fresh one does not reset the history.
    git_fixture(repo, "rm", "-r", "-q", "first")
    git_fixture(repo, "commit", "-q", "-m", "synthetic history reset")
    loosened = BOUNDARY["validator_comparison"] | {"aggregation": "pooled"}
    reset, reset_run = attempt(
        repo, registered, "reset", source=registered.run, validator_comparison=loosened
    )
    _, record = decide(reset, reset_run, repo / "reset-decision")
    assert record["outcome"] == "inconsistent"
    [problem] = [p for p in record["problems"] if p.startswith("history reset")]
    assert decided["protocol"]["id"] in problem


@pytest.mark.ml
def test_no_iteration_follows_a_passed_first_attempt(
    tmp_path: Path, registered: Fixture, fake_llm: FakeOpenAI
) -> None:
    repo = new_repo(tmp_path / "c")
    first, first_run = attempt(repo, registered, "first", source=registered.run, **BOUNDARY)
    folder = packet(first, first_run, repo / "first-packet")
    review(folder, "equivalent")
    _, decided = decide(first, first_run, repo / "first-decision", folder)
    assert decided["criteria_outcome"] == "passed"
    commit(
        repo,
        "first/protocol.decision-claim.json",
        "first/protocol.review-claim.json",
        "first-decision",
    )

    fake_llm.default = GEProvider()
    iteration_config(repo / "second", registered, fake_llm)
    bound = {"attempt": 2, "previous_protocol": "../first/protocol.json", "change": "Guidance."}
    second, second_run = attempt(
        repo, registered, "second", source=None, config="ge.yaml", iteration=bound, **BOUNDARY
    )
    _, record = decide(second, second_run, repo / "second-decision")
    assert "the first attempt passed; no development iteration follows a pass" in record["problems"]
    assert record["outcome"] == "inconsistent"
