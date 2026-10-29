"""Briefs on disk: one YAML file per brief, grouped by split, and frozen splits.

Layout, under the benchmark folder (default `benchmark/`):

- `<split>/<brief id>.yaml`: one brief; its file name matches its ID and its
  folder matches its split.
- `frozen.json`: for each frozen split, the SHA-256 of every brief in it.

A frozen split is fixed: editing, adding or removing one of its briefs fails
the check, the same way a changed prompt template is rejected. The dev split is
normally left unfrozen, so dev briefs can change freely.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from nde.files import checksum_lf, write_text_lf
from nde.runs.records import read_outputs
from nde.schema.brief import SPLITS, Brief, BriefContent, BriefDrafts, Split

DEFAULT_BENCHMARK_DIR = Path("benchmark")
FROZEN = "frozen.json"


class BriefError(Exception):
    """Briefs cannot be loaded, or a split cannot be frozen, as asked."""


@dataclass(frozen=True)
class BriefFile:
    path: Path
    sha256: str
    brief: Brief


@dataclass
class CheckResult:
    briefs: dict[Split, list[BriefFile]]
    problems: list[str]

    @property
    def passed(self) -> bool:
        return not self.problems


def _describe(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or '<brief>'}: {item['msg']}"
        for item in error.errors(include_url=False)
    )


def read_frozen(root: Path) -> dict[str, dict[str, str]]:
    path = root / FROZEN
    if not path.is_file():
        return {}
    data: dict[str, dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _load(root: Path, path: Path) -> tuple[BriefFile | None, str | None]:
    label = path.relative_to(root).as_posix()
    data = path.read_bytes()
    try:
        brief = Brief.model_validate(yaml.safe_load(data))
    except yaml.YAMLError as error:
        return None, f"{label}: not valid YAML: {error}"
    except ValidationError as error:
        return None, f"{label}: {_describe(error)}"
    if brief.id != path.stem:
        return None, f"{label}: file name must match id {brief.id!r}"
    if brief.split != path.parent.name:
        return (
            None,
            f"{label}: split {brief.split!r} does not match its folder {path.parent.name!r}",
        )
    return BriefFile(path, checksum_lf(data), brief), None


def check(root: Path, splits: tuple[Split, ...] = SPLITS) -> CheckResult:
    """Validate the briefs and frozen records of `splits` (by default every split).

    Files of unchecked splits are never opened: IDs shared with them are found from
    their file names alone. Writes nothing.
    """
    result = CheckResult(briefs={split: [] for split in SPLITS}, problems=[])
    for split in splits:
        for path in sorted((root / split).glob("*.yaml")):
            loaded, problem = _load(root, path)
            if problem is not None:
                result.problems.append(problem)
            if loaded is not None:
                result.briefs[split].append(loaded)
    dev_ids = (
        {file.brief.id for file in result.briefs["dev"]}
        if "dev" in splits
        else {path.stem for path in (root / "dev").glob("*.yaml")}
    )
    test_ids = (
        [file.brief.id for file in result.briefs["test"]]
        if "test" in splits
        else sorted(path.stem for path in (root / "test").glob("*.yaml"))
    )
    result.problems += [
        f"{brief_id} appears in both dev and test" for brief_id in test_ids if brief_id in dev_ids
    ]
    for split, recorded in read_frozen(root).items():
        if split not in splits:
            continue
        found = {
            path.stem: checksum_lf(path.read_bytes()) for path in (root / split).glob("*.yaml")
        }
        for brief_id in sorted(recorded.keys() | found.keys()):
            label = f"{split}/{brief_id}"
            if brief_id not in found:
                result.problems.append(f"{label} was removed after the {split} split was frozen")
            elif brief_id not in recorded:
                result.problems.append(f"{label} was added after the {split} split was frozen")
            elif found[brief_id] != recorded[brief_id]:
                result.problems.append(f"{label} changed after the {split} split was frozen")
    return result


def freeze(root: Path, split: Split) -> list[str]:
    """Record the checksums of every brief in `split`; returns the frozen brief IDs.

    Refuses when any brief is invalid, or when the split is already frozen and
    its briefs no longer match what was recorded.
    """
    result = check(root)
    frozen = read_frozen(root)
    if result.problems:
        prefix = (
            f"the {split} split is already frozen and cannot be refrozen over changes"
            if split in frozen
            else "fix these before freezing"
        )
        raise BriefError(f"{prefix}:\n" + "\n".join(result.problems))
    current = {file.brief.id: file.sha256 for file in result.briefs[split]}
    if not current:
        raise BriefError(f"the {split} split has no briefs to freeze")
    frozen[split] = dict(sorted(current.items()))
    write_text_lf(root / FROZEN, json.dumps(dict(sorted(frozen.items())), indent=2) + "\n")
    return sorted(current)


def split_checksum(root: Path, split: Split) -> str | None:
    """One checksum for a whole frozen split, or None when the split is not frozen."""
    recorded = read_frozen(root).get(split)
    if recorded is None:
        return None
    canonical = json.dumps(recorded, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_briefs(root: Path, splits: tuple[Split, ...] = SPLITS) -> dict[str, Brief]:
    """Every valid brief of `splits` by ID; raises BriefError if their check fails."""
    result = check(root, splits)
    if result.problems:
        raise BriefError(f"briefs under {root} fail the check:\n" + "\n".join(result.problems))
    return {file.brief.id: file.brief for files in result.briefs.values() for file in files}


DRAFT_HEADER = (
    "# Draft brief for review: edit or delete it. To keep it, add `id: brief-NN` and\n"
    "# `split: dev` or `split: test`, then move it to benchmark/<split>/<id>.yaml.\n"
)


def write_drafts(run_dir: Path, out: Path) -> list[Path]:
    """Write every brief drafted in a run as its own YAML file under `out`, for review."""
    drafts = [
        BriefContent.model_validate(brief)
        for output in read_outputs(run_dir)
        if output.response_model == "brief_drafts" and isinstance(output.parsed, dict)
        for brief in BriefDrafts.model_validate(output.parsed).briefs
    ]
    if not drafts:
        raise BriefError(f"run {run_dir} has no brief drafts")
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, draft in enumerate(drafts, start=1):
        path = out / f"{run_dir.name}-{index:03d}.yaml"
        body = yaml.safe_dump(draft.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
        write_text_lf(path, DRAFT_HEADER + body)
        paths.append(path)
    return paths
