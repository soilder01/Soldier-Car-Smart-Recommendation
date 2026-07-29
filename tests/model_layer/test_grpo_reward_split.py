from collections import Counter

from scripts import freeze_grpo_reward_split as split


def test_reward_visible_split_is_deterministic_16_train_4_dev():
    train, dev, manifest = split.build_split()

    assert len(train) == 16
    assert len(dev) == 4
    assert Counter(row["intent"] for row in train) == {
        "recommend": 4,
        "compare": 4,
        "knowledge": 4,
        "sales": 4,
    }
    assert Counter(row["intent"] for row in dev) == {
        "recommend": 1,
        "compare": 1,
        "knowledge": 1,
        "sales": 1,
    }
    assert [row["id"] for row in dev] == [
        "reward-recommend-002",
        "reward-compare-004",
        "reward-knowledge-003",
        "reward-sales-004",
    ]
    assert manifest["gradient_policy"] == {
        "train_16": "eligible_for_gradient_updates",
        "dev_4": "evaluation_and_early_stopping_only_no_gradient",
    }


def test_reward_train_and_dev_union_exactly_reconstruct_frozen_source():
    train, dev, _manifest = split.build_split()
    source = split.load_jsonl(split.SOURCE_PATH)

    assert {
        split.normalized_sha256(row["id"]) for row in train + dev
    } == {
        split.normalized_sha256(row["id"]) for row in source
    }
    assert {
        split.normalized_sha256(row["query"]) for row in train + dev
    } == {
        split.normalized_sha256(row["query"]) for row in source
    }
    assert {
        split.normalized_sha256(row["id"]) for row in train
    }.isdisjoint(
        split.normalized_sha256(row["id"]) for row in dev
    )
    assert {
        split.normalized_sha256(row["query"]) for row in train
    }.isdisjoint(
        split.normalized_sha256(row["query"]) for row in dev
    )
