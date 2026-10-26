"""Publish a complete immutable artifact directory, or leave no ready directory."""

import shutil
import tempfile
from pathlib import Path

from nde.corpus.acquisition import CorpusError
from nde.files import write_text_lf


def publish(out: Path, files: dict[str, str]) -> None:
    if out.exists():
        if any(
            not (out / name).is_file() or (out / name).read_bytes() != text.encode()
            for name, text in files.items()
        ):
            raise CorpusError("immutable artifacts differ; create a new version")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        for name, text in files.items():
            write_text_lf(staging / name, text)
        staging.rename(out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
