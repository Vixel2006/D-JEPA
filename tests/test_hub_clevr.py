from PIL import Image

import djepa.data.clevr as clevr
from djepa.config import DataConfig


def _fake_row(question, answer, color=(10, 20, 30)):
    return {
        "texts": [{"user": f"Question: {question}", "assistant": f"Answer: {answer}."}],
        "images": [Image.new("RGB", (32, 32), color=color)],
    }


def test_question_classification_and_answer_normalisation():
    assert clevr.classify_clevr_question("How many red cubes are there?") == "counting"
    assert clevr.classify_clevr_question("What color is the sphere?") == "attribute"
    assert clevr.classify_clevr_question("What is to the left of the cube?") == "spatial_relation"
    assert clevr.normalize_answer("Answer: Blue.") == "blue"


def test_hub_items_are_streamed_into_clevr_records(monkeypatch):
    rows = [
        _fake_row("How many red cubes are there?", "2"),
        _fake_row("What color is the sphere?", "blue"),
        _fake_row("What is to the left of the cube?", "cube"),
    ]
    monkeypatch.setattr(clevr, "_load_hub_rows", lambda config: rows)

    items, answers, images = clevr.build_hub_clevr_items(
        DataConfig(source="clevr", max_samples=2, answer_vocab_size=10)
    )

    assert len(items) == 2
    assert answers == ["2", "blue"]
    assert all(item.image is not None for item in items)
    assert items[0].category == "counting"
    assert items[1].category == "attribute"
    assert images[0].size == (32, 32)
