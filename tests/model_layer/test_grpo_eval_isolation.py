import copy

import pytest

from scripts import validate_grpo_eval_isolation as isolation


def _case(case_id: str, query: str) -> dict:
    return {
        "id": case_id,
        "query": query,
        "intent": "knowledge",
        "expected_tools": ["retrieve_knowledge_base"],
        "optional_tools": ["search_web_info"],
        "forbidden_tools": [
            "extract_user_profile",
            "search_and_rank_vehicles",
            "generate_sales_talk",
        ],
        "allowed_models": [],
    }


def test_normalization_makes_nfkc_and_whitespace_collisions_visible():
    assert isolation.normalize(" ＣＡＳＥ-１ \n") == "case-1"
    assert isolation.normalize("电池  安全\n怎么查") == "电池 安全 怎么查"


def test_disjoint_check_rejects_same_normalized_query_with_different_ids():
    left = isolation.fingerprint_rows(
        [_case("left-1", " 电池  安全怎么查 ")],
        label="left",
    )
    right = isolation.fingerprint_rows(
        [_case("right-1", "电池 安全怎么查")],
        label="right",
    )

    with pytest.raises(ValueError, match="normalized query SHA overlap"):
        isolation.assert_disjoint(left, right)


def test_disjoint_check_rejects_same_normalized_id_with_different_queries():
    left = isolation.fingerprint_rows(
        [_case("ＧＲＰＯ-１", "问题甲")],
        label="left",
    )
    right = isolation.fingerprint_rows(
        [_case("grpo-1", "问题乙")],
        label="right",
    )

    with pytest.raises(ValueError, match="normalized ID SHA overlap"):
        isolation.assert_disjoint(left, right)


def test_current_grpo_final_eval_is_fail_closed_and_physically_isolated():
    report = isolation.validate_isolation()

    assert report["status"] == "passed"
    assert report["split_counts"] == {
        "reward_visible": 20,
        "sft_held_out": 40,
        "grpo_final": 40,
    }
    assert all(
        item["normalized_id_sha256_overlap"] == 0
        and item["normalized_query_sha256_overlap"] == 0
        for item in report["pairwise_eval_isolation"]
    )
    assert report["grpo_final_vs_sft_source"] == {
        "normalized_id_sha256_overlap": 0,
        "normalized_query_sha256_overlap": 0,
    }
    assert report["grpo_final_vs_historical_query_corpus"][
        "normalized_query_sha256_overlap"
    ] == 0
    assert report["manifest"]["all_candidate_indices_above_650"] is True
    assert report["reward_train_dev"]["counts"] == {
        "train": 16,
        "dev": 4,
    }
    assert report["reward_train_dev"]["train_dev_isolation"] == {
        "normalized_id_sha256_overlap": 0,
        "normalized_query_sha256_overlap": 0,
    }
    assert report["reward_train_dev"]["union_exactly_matches_reward_visible"] is True
    assert all(
        item["normalized_id_sha256_overlap"] == 0
        and item["normalized_query_sha256_overlap"] == 0
        for item in report["reward_train_dev"]["against_read_only_eval"]
    )


def test_fingerprint_rejects_duplicate_rows_inside_one_split():
    row = _case("case-1", "问题")
    duplicate = copy.deepcopy(row)
    duplicate["id"] = "case-2"

    with pytest.raises(ValueError, match="duplicate normalized query"):
        isolation.fingerprint_rows([row, duplicate], label="duplicate")
