"""Verification steps that need more than a single tool invocation.

Run as `python -m nde.verify.steps <step>`. Each step exits non-zero on failure
and writes everything to stdout so the runner can scan it for warnings.
"""

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

# Files excluded from the secret scan, each with its justification.
_SECRET_SCAN_EXCLUDES: dict[str, str] = {
    # SHA-256 checksums of prompt templates: high-entropy hex by design, and
    # JSON has no comments to carry an inline allowlist pragma.
    "prompts/checksums.json": "prompt template checksums",
    # SHA-256 checksums of the briefs in frozen splits, for the same reasons.
    "benchmark/frozen.json": "frozen brief split checksums",
    "tests/fixtures/frozen-split/frozen.json": "checksums of the frozen fixture split",
    "benchmark/rubric/checksums.json": "rubric version checksums",
    # SHA-256 checksums recorded for each calibration source; YAML written by
    # `nde calibration fetch`, which cannot place inline pragmas.
    "data/manifest.calibration.yaml": "calibration source checksums",
    # Hex High Entropy String: three public source checksums, already bound
    # into the owner's G-schema packet. Preserve those exact manifest bytes;
    # review this exception if the manifest is regenerated or its fields change.
    "data/manifest.gschema-20261027.yaml": "immutable gold source checksums",
    # HexHighEntropyString: exact SHA-256 of the synthetic embedded source and
    # evidence spans. JSON cannot carry an inline pragma. Revisit if this fixture
    # gains non-checksum high-entropy fields; source identity is verified in tests.
    "src/nde/evidence/controls/evidence.json": "synthetic source/span checksum fixture",
    # Hex High Entropy String: generated, public SHA-256 identities in these
    # exact evidence documents. JSON cannot carry inline pragmas, and changing
    # immutable evidence would invalidate its recorded identities. Review these
    # exceptions whenever the documents are regenerated or gain new field kinds.
    "docs/evidence/g-schema-2026-10-27/decision.json": "owner G-schema evidence identities",
    "docs/evidence/dev-tier-selection-2026-10-27/decision.json": "owner dev decision identities",
    # Also Secret Keyword: api_key_env names DEEPSEEK_API_KEY; it is an
    # environment-variable reference, not a credential. Revisit if binding
    # formats change; credential values must never be stored in this document.
    "docs/evidence/dev-tier-selection-2026-10-27/selection.json": "immutable selection/env name",
    # Hex High Entropy String: this one frozen dev cohort and its matching
    # preregistration contain only public content/config/selection hashes.
    # Review on regeneration or schema changes; do not generalise to a folder.
    "benchmark/pilots/mb-20261027/frozen.json": "pilot brief checksums",
    "benchmark/prereg/dev-mb-pilot-20261027-cohort.json": "pilot cohort checksums",
    "benchmark/prereg/dev-mb-pilot-20261027.json": "pilot registration content identities",
}


def _uv() -> str:
    return os.environ.get("UV", "uv")


def _run(args: Sequence[str]) -> int:
    print(f"$ {' '.join(args)}", flush=True)
    # S603: arguments are fixed by this module, not taken from user input.
    return subprocess.run(args, check=False).returncode  # noqa: S603


def audit() -> int:
    """Audit every locked dependency, in all groups, for known vulnerabilities."""
    with tempfile.TemporaryDirectory() as tmp:
        requirements = Path(tmp) / "requirements.txt"
        code = _run(
            [
                _uv(),
                "export",
                "--frozen",
                "--all-groups",
                "--no-emit-project",
                "--format",
                "requirements-txt",
                "--quiet",
                "--output-file",
                str(requirements),
            ]
        )
        if code != 0:
            return code
        return _run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--requirement",
                str(requirements),
                "--disable-pip",
                "--require-hashes",
                "--strict",
                "--progress-spinner",
                "off",
            ]
        )


def _candidate_files() -> list[str]:
    # S607: git is resolved from PATH on purpose; it is a developer prerequisite.
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],  # noqa: S607
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    return [
        name
        for name in listing.split("\0")
        if name and name not in _SECRET_SCAN_EXCLUDES and Path(name).is_file()
    ]


def secrets() -> int:
    """Scan tracked and untracked, non-ignored files for committed secrets."""
    files = _candidate_files()
    print(f"$ detect-secrets scan ({len(files)} files)", flush=True)
    # S603: arguments are this interpreter, a fixed module and repository file names.
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "detect_secrets", "scan", *files],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        print(completed.stdout + completed.stderr)
        return completed.returncode
    findings: dict[str, list[dict[str, object]]] = json.loads(completed.stdout)["results"]
    for name, items in sorted(findings.items()):
        for item in items:
            print(f"possible secret: {name}:{item.get('line_number')} ({item.get('type')})")
    if findings:
        print("secret scan FAILED: remove the secret, or mark a false positive inline with")
        print("`# pragma: allowlist secret` and a justification")
        return 1
    print("secret scan passed")
    return 0


def missing_from_wheel(wheel: Path, package_dir: Path) -> list[str]:
    """Source modules under `package_dir` that the wheel does not contain."""
    root = package_dir.parent
    expected = {path.relative_to(root).as_posix() for path in package_dir.rglob("*.py")}
    with zipfile.ZipFile(wheel) as archive:
        return sorted(expected - set(archive.namelist()))


def build() -> int:
    """Build the sdist and wheel, then check the wheel ships every source module."""
    with tempfile.TemporaryDirectory() as tmp:
        code = _run([_uv(), "build", "--out-dir", tmp, "--quiet"])
        if code != 0:
            return code
        [wheel] = Path(tmp).glob("*.whl")
        missing = missing_from_wheel(wheel, Path("src") / "nde")
        for name in missing:
            print(f"missing from wheel: {name}")
        return 1 if missing else 0


_STEPS = {"audit": audit, "secrets": secrets, "build": build}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in _STEPS:
        print(f"usage: python -m nde.verify.steps {{{','.join(_STEPS)}}}")
        return 2
    return _STEPS[args[0]]()


if __name__ == "__main__":
    sys.exit(main())
