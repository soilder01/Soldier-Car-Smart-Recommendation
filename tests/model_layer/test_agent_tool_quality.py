import json

from app.services import agent_graph


def _rank(args):
    return json.loads(agent_graph.search_and_rank_vehicles.invoke(args))


def test_extract_profile_preserves_energy_consumption_without_fake_value_focus():
    profile = json.loads(
        agent_graph.extract_user_profile.invoke(
            {
                "query": "预算28万，三口之家，主要城市通勤，想要插混SUV，关注空间和能耗，请推荐。",
                "budget_max": 280000,
            }
        )
    )

    assert profile["budget_max"] == 280000
    assert profile["family_size"] == 3
    assert profile["preferred_type"] == "SUV"
    assert profile["preferred_energy"] == "插混"
    assert "空间" in profile["concerns"]
    assert "能耗" in profile["concerns"]
    assert "性价比" not in profile["concerns"]


def test_plugin_hybrid_query_returns_multiple_non_bev_candidates():
    rows = _rank(
        {
            "budget_max": 280000,
            "preferred_type": "SUV",
            "preferred_energy": "插混",
            "concerns": "空间,能耗",
            "top_k": 5,
        }
    )

    assert len(rows) >= 3
    assert all(row["energy"] in {"插混", "增程"} for row in rows)
    assert rows[0]["energy"] == "插混"
    assert any(row["energy"] == "增程" for row in rows)
    assert len({row["score"] for row in rows}) > 1
    assert any("广义插电备选" in row.get("selection_note", "") for row in rows)


def test_search_tool_returns_grounded_vehicle_specs():
    rows = _rank(
        {
            "budget_max": 280000,
            "preferred_type": "SUV",
            "preferred_energy": "插混",
            "concerns": "空间,能耗",
            "top_k": 3,
        }
    )
    first = rows[0]

    assert first["full_name"] == "比亚迪宋PLUS DM-i"
    assert first["price"] == "12.98-16.98万"
    assert first["specs"]["price_range"] == "12.98-16.98万"
    assert first["specs"]["match_score"] == "97.7分"
    assert first["specs"]["cltc_range"] == "1100km"
    assert first["specs"]["battery"] == "18.3kWh"
    assert first["specs"]["fast_charge"] == "35分钟"
    assert first["specs"]["seats"] == "5座"
    assert first["specs"]["wheelbase"] == "2765mm"
    assert first["specs"]["trunk_volume"] == "574L"
    assert "price_min_yuan" not in first["specs"]
    assert "price_max_yuan" not in first["specs"]
    assert first["energy_evidence"] == ["低油耗", "插混油耗低"]
    assert "official_fuel_consumption_l_per_100km" in first["known_missing_specs"]


def test_compare_lookup_returns_every_named_vehicle_with_specs():
    result = _rank(
        {
            "model_names": ["比亚迪宋PLUS DM-i", "吉利银河L7"],
            "concerns": "空间,能耗",
            "top_k": 5,
        }
    )

    lookup = result["named_vehicle_lookup"]
    assert lookup == {
        "requested_model_names": ["比亚迪宋PLUS DM-i", "吉利银河L7"],
        "resolved_model_names": ["比亚迪宋PLUS DM-i", "吉利银河L7"],
        "missing_model_names": [],
        "named_vehicle_missing": False,
    }
    assert [row["full_name"] for row in result["named_vehicles"]] == [
        "比亚迪宋PLUS DM-i",
        "吉利银河L7",
    ]
    assert all(row["specs"]["price_range"] for row in result["named_vehicles"])


def test_compare_lookup_prefers_exact_extended_model_over_base_model_prefix():
    result = _rank(
        {
            "model_names": ["享界S9", "享界 S9 增程"],
            "concerns": "商务,长途",
            "top_k": 5,
        }
    )

    lookup = result["named_vehicle_lookup"]
    assert lookup["named_vehicle_missing"] is False
    assert lookup["missing_model_names"] == []
    assert lookup["resolved_model_names"] == ["享界S9", "享界S9增程"]
    assert [row["energy"] for row in result["named_vehicles"]] == ["纯电", "增程"]
    assert all(row["specs"]["price_range"] for row in result["named_vehicles"])


def test_compare_lookup_marks_missing_named_vehicle_and_returns_neighbors():
    result = _rank(
        {
            "model_names": ["腾势D9", "问界M9"],
            "concerns": "商务,空间",
            "top_k": 5,
        }
    )

    lookup = result["named_vehicle_lookup"]
    assert lookup["named_vehicle_missing"] is True
    assert lookup["missing_model_names"] == ["腾势D9"]
    assert [row["full_name"] for row in result["named_vehicles"]] == ["问界M9"]
    assert result["supplemental_vehicles"]


def test_pure_ev_query_does_not_include_plugin_hybrids():
    rows = _rank(
        {
            "budget_max": 280000,
            "preferred_type": "SUV",
            "preferred_energy": "纯电",
            "concerns": "空间,能耗",
            "top_k": 5,
        }
    )

    assert rows
    assert all(row["energy"] == "纯电" for row in rows)


def test_recommend_prompt_defines_range_extender_policy():
    prompt = agent_graph._get_prompt_for_intent("recommend")

    assert "增程算作广义插电范畴" in prompt
    assert "如用户严格要求 PHEV" in prompt
    assert "一切硬指标，只能逐字使用 search_and_rank_vehicles 返回的 specs" in prompt
    assert "禁止凭模型记忆补充或改写任何精确数值" in prompt
    assert "specs 未返回的字段一律标注“需官方核验”" in prompt
    assert "价格只能复制完整 price_range" in prompt
    assert "不得换算成“万”" in prompt
    assert "车型硬指标只能出现在 Top 车型表格中" in prompt
    assert "车型名称中的数字不是规格" in prompt
    assert "座位数只能逐字复制 specs.seats" in prompt
    assert "不得讨论价格优惠、金融政策、购车权益、保修或交付周期" in prompt


def test_customer_service_intent_uses_research_tools_only():
    names = [tool.name for tool in agent_graph.TOOLS_BY_INTENT["customer_service"]]
    prompt = agent_graph._get_prompt_for_intent("customer_service")

    assert names == ["retrieve_knowledge_base", "search_web_info"]
    assert "不要调用 search_and_rank_vehicles" in prompt
    assert "只能写“该项工具未返回，需以官方实时信息核验”" in prompt
    assert "不得夹带车型、能源路线、预算或购车建议" in prompt


def test_compare_prompt_requires_named_lookup_and_honest_missing_branch():
    prompt = agent_graph._get_prompt_for_intent("compare")

    assert "model_names 数组传入" in prompt
    assert "named_vehicle_missing=false" in prompt
    assert "库中无此车规格" in prompt
    assert "价格只能复制完整 price_range" in prompt
    assert "表格之外不写任何带单位的车型数值" in prompt
    assert "前 3 步均为不可省略的前置工具调用" in prompt


def test_deep_search_intent_requires_multi_step_research_tools():
    names = [tool.name for tool in agent_graph.TOOLS_BY_INTENT["deep_search"]]
    prompt = agent_graph._get_prompt_for_intent("deep_search")

    assert names == [
        "extract_user_profile",
        "search_and_rank_vehicles",
        "retrieve_knowledge_base",
        "search_web_info",
        "generate_sales_talk",
    ]
    assert "observe -> decide" in prompt
    assert "一切硬指标，只能逐字使用 search_and_rank_vehicles 返回的 specs" in prompt
    assert "禁止凭模型记忆补充或改写任何精确数值" in prompt
    assert "specs 未返回的字段一律标注“需官方核验”" in prompt
    assert "价格只能复制完整 price_range" in prompt
    assert "不得换算成“万”" in prompt
    assert "车型硬指标只能出现在候选车型表格中" in prompt
    assert "车型名称中的数字不是规格" in prompt
    assert "座位数只能逐字复制 specs.seats" in prompt
    assert "不得讨论价格优惠、金融政策、购车权益、保修或交付周期" in prompt
    assert "用户未明确预算时不得自行设定任何预算档位" in prompt
    assert "分析正文不得出现任何带单位数字" in prompt
