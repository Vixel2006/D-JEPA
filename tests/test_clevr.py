import random

from djepa.data.clevr import generate_procedural_clevr_scene, split_indices


def test_procedural_scene_has_expected_structure():
    rng = random.Random(0)
    image, qa_list = generate_procedural_clevr_scene(rng)
    assert image.size == (224, 224)
    assert {qa["category"] for qa in qa_list} == {
        "counting",
        "attribute",
        "spatial_relation",
    }
    assert all(qa["question"] for qa in qa_list)
    assert all(qa["answer"] for qa in qa_list)


def test_split_indices_matches_original_default():
    dataset = list(range(4000))
    train, test = split_indices(dataset)
    assert len(test) == 800
    assert len(train) == 3200
    assert train[-1] == 3199
    assert test[0] == 3200


def test_split_indices_honours_explicit_test_size():
    dataset = list(range(100))
    train, test = split_indices(dataset, test_size=25)
    assert len(test) == 25
    assert len(train) == 75
