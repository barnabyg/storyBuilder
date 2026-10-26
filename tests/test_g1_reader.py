"""Synthetic reader acceptance; these scores are not owner evidence."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nde.eval.batch import (
    BatchError,
    Judgement,
    ShownPair,
    View,
    close_batch,
    record_judgement,
    write_batch,
)
from nde.eval.reader import create_app
from nde.eval.rubric import load_rubric
from tests.conftest import REPO


def test_gate_reader_requires_both_primary_and_originality_before_persisting(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "synthetic"
    write_batch(
        batch,
        View(
            pairs=[ShownPair(a="Synthetic story one", b="Synthetic story two")],
            rubric=load_rubric(REPO / "benchmark/rubric", 1),
            optional_scores=True,
            primary_dimension="coherence",
            required_score_dimensions=["coherence", "originality"],
        ),
        View(),
    )
    client = TestClient(create_app(batch))
    page = client.get("/pairs/1").text
    for side in ("a", "b"):
        for dimension in ("coherence", "originality"):
            assert f'name="score-{side}-{dimension}" value="1" required' in page
    form = {"preferred": "A", "strength": "2", "score-a-coherence": "4", "score-b-coherence": "2"}
    assert client.post("/pairs/1", data=form).status_code == 422
    assert client.post("/close").status_code == 409
    form |= {"score-a-originality": "3", "score-b-originality": "3"}
    assert client.post("/pairs/1", data=form, follow_redirects=False).status_code == 303
    assert client.get("/api/pairs/1").json()["judgement"]["a"]["scores"] == {
        "coherence": 4,
        "originality": 3,
    }
    assert client.post("/close", follow_redirects=False).status_code == 303
    assert client.get("/api/progress").json()["closed"] is True
    assert "sealed" not in page
    assert client.get("/sealed/key.json").status_code == 404


def test_incomplete_persisted_gate_scores_cannot_close_the_batch(tmp_path: Path) -> None:
    batch = tmp_path / "synthetic"
    write_batch(
        batch,
        View(
            pairs=[ShownPair(a="Synthetic one", b="Synthetic two")],
            rubric=load_rubric(REPO / "benchmark/rubric", 1),
            optional_scores=True,
            required_score_dimensions=["coherence", "originality"],
        ),
        View(),
    )
    record_judgement(batch, Judgement(pair=1, preferred="A", strength=1, at="synthetic"))
    with pytest.raises(BatchError, match="required"):
        close_batch(batch)
