from training.grpo.reward_fn import (
    EvidenceClaim,
    IntentResponseSpec,
    deterministic_intent_response_check,
    score_grounded_answer,
)


def _compare_evidence():
    return (
        EvidenceClaim(
            canonical_entity="小米 SU7",
            canonical_attribute="specs.cltc_range",
            canonical_value="700km",
            source_tool="search_and_rank_vehicles",
            source_locator="named_vehicles[0].specs.cltc_range",
            entity_aliases=("小米SU7",),
            attribute_aliases=("续航",),
            anchor_tokens=("性能", "补能"),
        ),
        EvidenceClaim(
            canonical_entity="特斯拉 Model 3",
            canonical_attribute="specs.cltc_range",
            canonical_value="713km",
            source_tool="search_and_rank_vehicles",
            source_locator="named_vehicles[1].specs.cltc_range",
            entity_aliases=("特斯拉Model 3", "Model 3"),
            attribute_aliases=("续航",),
            anchor_tokens=("性能", "补能"),
        ),
        EvidenceClaim(
            canonical_entity="问界 M7",
            canonical_attribute="specs.seats",
            canonical_value="5座",
            source_tool="search_and_rank_vehicles",
            source_locator="supplemental_vehicles[0].specs.seats",
            entity_aliases=("问界M7",),
            attribute_aliases=("座位",),
            anchor_tokens=("空间",),
        ),
    )


def _compare_spec():
    return IntentResponseSpec(
        prompt_id="reward-compare-test",
        intent="compare",
        target_entities=("小米 SU7", "特斯拉 Model 3"),
        query_anchor_tokens=("性能", "补能"),
        query_attribute_anchors=("specs.cltc_range",),
        minimum_supported_claims=2,
    )


def test_compare_requires_truthful_query_bound_claim_for_each_target():
    result = deterministic_intent_response_check(
        answer=(
            "小米SU7的续航为700km。"
            "特斯拉Model 3的续航为713km。"
        ),
        spec=_compare_spec(),
        evidence_claims=_compare_evidence(),
    )

    assert result.passed is True
    assert result.matched_entities == ("小米 SU7", "特斯拉 Model 3")
    assert result.matched_attributes == ("specs.cltc_range",)


def test_compare_rejects_wrong_value_even_when_both_entities_are_present():
    result = deterministic_intent_response_check(
        answer=(
            "小米SU7的续航为755km。"
            "特斯拉Model 3的续航为713km。"
        ),
        spec=_compare_spec(),
        evidence_claims=_compare_evidence(),
    )

    assert result.passed is False
    assert result.reason == "unsupported_target_value"


def test_compare_rejects_safe_but_vacuous_unrelated_evidence_sentences():
    result = deterministic_intent_response_check(
        answer=(
            "小米SU7和特斯拉Model 3都值得考虑。"
            "问界M7的座位数为5座。"
            "这里列出了两条格式完整的证据句。"
        ),
        spec=_compare_spec(),
        evidence_claims=_compare_evidence(),
    )

    assert result.passed is False
    assert result.reason == "missing_query_bound_claim_for_target"


def test_knowledge_requires_two_exact_claims_bound_to_frozen_anchor():
    evidence = (
        EvidenceClaim(
            canonical_entity="蔚来 ES6",
            canonical_attribute="kb.swap_scenario",
            canonical_value="高速长途可通过换电减少补能等待",
            source_tool="retrieve_knowledge_base",
            source_locator="results[0].content#span-1",
            entity_aliases=("蔚来ES6",),
            attribute_aliases=("换电场景",),
            anchor_tokens=("换电", "长途"),
        ),
        EvidenceClaim(
            canonical_entity="蔚来 ES6",
            canonical_attribute="kb.swap_scenario",
            canonical_value="无家充用户可在换电站补能",
            source_tool="retrieve_knowledge_base",
            source_locator="results[0].content#span-2",
            entity_aliases=("蔚来ES6",),
            attribute_aliases=("换电场景",),
            anchor_tokens=("换电", "无家充"),
        ),
        EvidenceClaim(
            canonical_entity="蔚来 ES6",
            canonical_attribute="specs.seats",
            canonical_value="5座",
            source_tool="search_and_rank_vehicles",
            source_locator="named_vehicles[0].specs.seats",
            entity_aliases=("蔚来ES6",),
            attribute_aliases=("座位",),
            anchor_tokens=("空间",),
        ),
    )
    spec = IntentResponseSpec(
        prompt_id="reward-knowledge-test",
        intent="knowledge",
        target_entities=("蔚来 ES6",),
        query_anchor_tokens=("换电",),
        minimum_supported_claims=2,
    )

    passed = deterministic_intent_response_check(
        answer=(
            "蔚来ES6换电场景：高速长途可通过换电减少补能等待。"
            "蔚来ES6换电场景：无家充用户可在换电站补能。"
        ),
        spec=spec,
        evidence_claims=evidence,
    )
    vacuous = deterministic_intent_response_check(
        answer="蔚来ES6的座位数为5座。换电很方便，具体以官方为准。",
        spec=spec,
        evidence_claims=evidence,
    )

    assert passed.passed is True
    assert vacuous.passed is False
    assert vacuous.reason == "insufficient_anchor_bound_claims"


def test_recommendation_must_bind_decision_to_real_candidate_and_evidence():
    evidence = (
        EvidenceClaim(
            canonical_entity="小鹏 G6",
            canonical_attribute="specs.fast_charge",
            canonical_value="20分钟",
            source_tool="search_and_rank_vehicles",
            source_locator="vehicles[0].specs.fast_charge",
            entity_aliases=("小鹏G6",),
            attribute_aliases=("快充", "补能"),
            anchor_tokens=("快充", "补能"),
        ),
    )
    spec = IntentResponseSpec(
        prompt_id="reward-recommend-test",
        intent="recommend",
        target_entities=("小鹏 G6",),
        query_anchor_tokens=("快充",),
        query_attribute_anchors=("specs.fast_charge",),
        minimum_supported_claims=1,
        decision_tokens=("推荐", "首选"),
    )

    passed = deterministic_intent_response_check(
        answer="首选小鹏G6，小鹏G6的快充时间为20分钟。",
        spec=spec,
        evidence_claims=evidence,
    )
    fake = deterministic_intent_response_check(
        answer="首选问界M7，小鹏G6的快充时间为20分钟。",
        spec=spec,
        evidence_claims=evidence,
    )

    assert passed.passed is True
    assert fake.passed is False
    assert fake.reason == "decision_not_bound_to_evidence_candidate"


def test_grounding_score_exposes_four_finite_components():
    result = score_grounded_answer(
        answer=(
            "小米SU7的续航为700km。"
            "特斯拉Model 3的续航为713km。"
        ),
        spec=_compare_spec(),
        evidence_claims=_compare_evidence(),
    )

    assert result.gate == "passed"
    assert result.total > 0.0
    for value in (
        result.factual_precision,
        result.required_coverage,
        result.source_integrity,
        result.concision,
    ):
        assert 0.0 <= value <= 1.0
