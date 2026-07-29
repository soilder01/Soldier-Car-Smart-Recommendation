from datetime import datetime
from typing import Any, Dict, List

from app.schemas import RecommendRequest, UserProfile
from app.services.obsidian_vault import write_note
from app.services.recommender import recommend

EVALUATION_CASES = [
    {
        "id": "family-no-home-charger",
        "name": "三口之家无家充 SUV",
        "query": "预算25万以内，三口之家，上海通勤每天50公里，没有家充，关注续航、空间和智驾，推荐哪几款新能源SUV？",
        "expected_profile": {"budget_max": 250000, "city": "上海", "family_size": 3, "has_home_charger": False, "preferred_type": "SUV"},
        "expected_concerns": ["续航", "空间", "智驾", "补能"],
        "expected_energy_any": ["插混", "增程"],
        "expected_model_any": ["理想 L6", "问界 M7", "比亚迪 宋PLUS DM-i"],
    },
    {
        "id": "home-charger-pure-ev",
        "name": "有家充纯电通勤",
        "query": "北京上班通勤每天35公里，有固定车位和家充，预算30万以内，想买纯电SUV，关注智驾和安全",
        "expected_profile": {"budget_max": 300000, "city": "北京", "has_home_charger": True, "preferred_type": "SUV", "preferred_energy": "纯电"},
        "expected_concerns": ["智驾", "安全"],
        "expected_energy_any": ["纯电"],
        "expected_model_any": ["特斯拉 Model Y", "小鹏 G6", "比亚迪 宋L EV"],
    },
    {
        "id": "explicit-compare",
        "name": "点名车型对比",
        "query": "Model Y 和小鹏G6 怎么选？我预算28万，深圳用车，关注智驾、补能和保值",
        "expected_profile": {"budget_max": 280000, "city": "深圳", "explicit_vehicle_compare": True},
        "expected_concerns": ["智驾", "补能", "保值"],
        "expected_model_any": ["特斯拉 Model Y", "小鹏 G6"],
        "expected_top_all": ["特斯拉 Model Y", "小鹏 G6"],
    },
    {
        "id": "mpv-family",
        "name": "多人家庭 MPV",
        "query": "二胎家庭六个人坐，预算35万左右，广州用车，希望空间大、安全配置高，最好是MPV",
        "expected_profile": {"budget_max": 350000, "city": "广州", "family_size": 4, "preferred_type": "MPV"},
        "expected_concerns": ["空间", "安全", "性价比"],
        "expected_type_any": ["MPV"],
        "expected_model_any": ["腾势 D9 DM-i", "小鹏 X9"],
    },
    {
        "id": "social-luxury",
        "name": "社交形象与豪华感",
        "query": "预算45万，想买一辆适合商务接待和社交场合的新能源车，最好有牌面，内饰氛围好",
        "expected_profile": {"budget_max": 450000},
        "expected_concerns": ["社交形象", "内饰氛围", "品牌"],
        "expected_model_any": ["奔驰 E300L", "宝马 530Li", "享界 S9", "蔚来 ET7"],
    },
]


def _vehicle_name(item: Dict[str, Any]) -> str:
    return f"{item.get('brand', '')} {item.get('model', '')}".strip()


def _check_equal(profile: Dict[str, Any], field: str, expected: Any) -> Dict[str, Any]:
    actual = profile.get(field)
    passed = actual == expected
    return {
        "name": f"画像字段 {field}",
        "passed": passed,
        "expected": expected,
        "actual": actual,
        "message": "命中" if passed else f"期望 {expected}，实际 {actual}",
    }


def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    result = recommend(case["query"], UserProfile(), 5)
    profile = result["profile"]
    recommendations = result["recommendations"]
    top_names = [_vehicle_name(item) for item in recommendations]
    checks = []

    for field, expected in case.get("expected_profile", {}).items():
        checks.append(_check_equal(profile, field, expected))

    concerns = set(profile.get("concerns") or [])
    for concern in case.get("expected_concerns", []):
        passed = concern in concerns
        checks.append({
            "name": f"关注点 {concern}",
            "passed": passed,
            "expected": concern,
            "actual": "、".join(sorted(concerns)),
            "message": "命中" if passed else "未识别该关注点",
        })

    if case.get("expected_model_any"):
        expected = case["expected_model_any"]
        passed = any(name in top_names for name in expected)
        checks.append({
            "name": "Top5 包含期望车型",
            "passed": passed,
            "expected": " / ".join(expected),
            "actual": "、".join(top_names),
            "message": "命中" if passed else "Top5 未覆盖期望候选",
        })

    if case.get("expected_top_all"):
        expected = case["expected_top_all"]
        top_two = top_names[:2]
        passed = all(name in top_two for name in expected)
        checks.append({
            "name": "点名车型进入 Top2",
            "passed": passed,
            "expected": "、".join(expected),
            "actual": "、".join(top_two),
            "message": "命中" if passed else "点名车型未全部进入 Top2",
        })

    if case.get("expected_energy_any"):
        expected = set(case["expected_energy_any"])
        actual = [item.get("energy_type") for item in recommendations[:3]]
        passed = any(item in expected for item in actual)
        checks.append({
            "name": "Top3 能源路线匹配",
            "passed": passed,
            "expected": "、".join(expected),
            "actual": "、".join(actual),
            "message": "命中" if passed else "Top3 能源路线不符合场景预期",
        })

    if case.get("expected_type_any"):
        expected = set(case["expected_type_any"])
        actual = [item.get("vehicle_type") for item in recommendations[:3]]
        passed = any(item in expected for item in actual)
        checks.append({
            "name": "Top3 车型类别匹配",
            "passed": passed,
            "expected": "、".join(expected),
            "actual": "、".join(actual),
            "message": "命中" if passed else "Top3 车型类别不符合场景预期",
        })

    passed_count = sum(1 for item in checks if item["passed"])
    score = round(passed_count / max(len(checks), 1) * 100, 1)
    failed_checks = [item for item in checks if not item["passed"]]
    status = "pass" if score >= 85 else "warn" if score >= 70 else "fail"
    diagnosis = "推荐链路符合当前测试预期" if status == "pass" else "；".join(item["message"] for item in failed_checks[:3])
    return {
        "id": case["id"],
        "name": case["name"],
        "query": case["query"],
        "status": status,
        "score": score,
        "passed_count": passed_count,
        "check_count": len(checks),
        "profile": profile,
        "top_models": top_names,
        "recommendations": recommendations[:5],
        "checks": checks,
        "failed_checks": failed_checks,
        "diagnosis": diagnosis,
    }


def run_recommendation_evaluation(persist: bool = True) -> Dict[str, Any]:
    cases = [evaluate_case(case) for case in EVALUATION_CASES]
    passed = sum(1 for item in cases if item["status"] == "pass")
    warned = sum(1 for item in cases if item["status"] == "warn")
    failed = sum(1 for item in cases if item["status"] == "fail")
    average_score = round(sum(item["score"] for item in cases) / max(len(cases), 1), 1)
    summary = {
        "case_count": len(cases),
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "pass_rate": round(passed / max(len(cases), 1) * 100, 1),
        "average_score": average_score,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    note = {}
    if persist:
        note = persist_evaluation(summary, cases)
    return {"summary": summary, "cases": cases, "obsidian_note": note}


def persist_evaluation(summary: Dict[str, Any], cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    lines = []
    for item in cases:
        lines.append(f"- {item['name']}：{item['score']} 分，状态 {item['status']}，Top：{'、'.join(item['top_models'][:3])}，诊断：{item['diagnosis']}")
    body = f"""
## 评估概览

- 用例数：{summary['case_count']}
- 通过率：{summary['pass_rate']}%
- 平均分：{summary['average_score']}
- 通过/警告/失败：{summary['passed']} / {summary['warned']} / {summary['failed']}

## 用例结果

{chr(10).join(lines)}

## 后续优化方向

- 优先修复 fail 和 warn 用例中的画像字段误识别。
- 对 Top 推荐未命中场景，检查车型库字段和评分权重。
- 将新的真实客户问题补充为固定测试集。

## 关联节点

- [[推荐链路]]
- [[用户画像解析优化]]
- [[自生长知识库方案]]
"""
    title = f"推荐质量评估-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    path = write_note("07-测试样例", title, {
        "type": "evaluation-report",
        "source": "api/evaluation/recommendation",
        "created_at": summary["generated_at"],
        "tags": ["推荐评估", "测试用例", "质量闭环"],
    }, body)
    return {"title": title, "path": path}


AGENT_REGRESSION_CASES = [
    {
        "id": "agent-family-local",
        "name": "Agent家庭通勤本地池",
        "query": "预算25万以内，三口之家，上海通勤每天50公里，有家充，关注续航、空间和智驾，推荐新能源SUV",
        "profile": UserProfile(budget_max=250000, city="上海", family_size=3, commute_km=50, has_home_charger=True, preferred_type="SUV", concerns=["续航", "空间", "智驾"]),
        "expected_pool": "local",
        "expected_trace": ["ProfileParserTool", "CandidatePoolSelectorTool", "RankTool", "FeedbackPolicyTool", "EvidenceRetrievalTool", "RiskCheckTool", "ObsidianCaseWriterTool"],
    },
    {
        "id": "agent-no-home-fused",
        "name": "Agent无家充复杂场景",
        "query": "没有家充，预算30万，四口之家，日常通勤和周末长途怎么选新能源车",
        "profile": UserProfile(budget_max=300000, family_size=4, has_home_charger=False, concerns=["补能", "空间"]),
        "expected_pool": "fused",
        "expected_trace": ["ProfileParserTool", "CandidatePoolSelectorTool", "RankTool", "FeedbackPolicyTool", "RiskCheckTool"],
    },
    {
        "id": "agent-real-world",
        "name": "Agent真实数据扩展池",
        "query": "基于真实公开数据和欧洲海外车型规格，帮我看 Tesla、BMW、Audi 电动车怎么选",
        "profile": UserProfile(budget_max=500000, concerns=["真实数据", "续航"]),
        "expected_pool": "real",
        "expected_trace": ["ProfileParserTool", "CandidatePoolSelectorTool", "RankTool", "FeedbackPolicyTool", "EvidenceRetrievalTool"],
    },
    {
        "id": "agent-explicit-compare",
        "name": "Agent点名车型对比",
        "query": "特斯拉 Model Y 和小鹏 G6 怎么选，预算30万，主要城市通勤，关注智驾、续航和空间",
        "profile": UserProfile(budget_max=300000, commute_km=45, has_home_charger=True, preferred_type="SUV", concerns=["智驾", "续航", "空间"]),
        "expected_pool": "fused",
        "expected_trace": ["ProfileParserTool", "CandidatePoolSelectorTool", "RankTool", "FeedbackPolicyTool", "ObsidianCaseWriterTool"],
    },
]


def _check_agent_case(case: Dict[str, Any]) -> Dict[str, Any]:
    import app.services.agent_orchestrator as orchestrator

    req = RecommendRequest(query=case["query"], profile=case["profile"], top_k=5, use_deep_search=False, candidate_pool_strategy="auto")
    old_key = orchestrator.OPENAI_API_KEY
    orchestrator.OPENAI_API_KEY = ""
    try:
        result = orchestrator.recommend_with_orchestrator(req)
    finally:
        orchestrator.OPENAI_API_KEY = old_key
    recs = result.get("recommendations", [])
    trace_agents = [item.get("agent") for item in result.get("agent_trace", [])]
    pool = result.get("pool_decision", {}).get("selected_pool")
    feedback_policy = result.get("feedback_policy", {})
    checks = [
        {"name": "候选池选择", "passed": pool == case["expected_pool"], "expected": case["expected_pool"], "actual": pool, "message": "候选池符合预期" if pool == case["expected_pool"] else "候选池选择偏离预期"},
        {"name": "推荐结果非空", "passed": len(recs) > 0, "expected": "至少1条", "actual": len(recs), "message": "推荐结果已返回" if recs else "推荐结果为空"},
        {"name": "Obsidian写入", "passed": bool(result.get("obsidian_note", {}).get("path")), "expected": "存在路径", "actual": result.get("obsidian_note", {}).get("path", ""), "message": "已沉淀长期记忆" if result.get("obsidian_note", {}).get("path") else "未写入Obsidian"},
        {"name": "反馈策略结构", "passed": bool(feedback_policy.get("policy", {}).get("stability")), "expected": "包含stability", "actual": feedback_policy.get("policy", {}).get("stability", {}), "message": "反馈策略稳定性元信息存在" if feedback_policy.get("policy", {}).get("stability") else "反馈策略缺少稳定性元信息"},
    ]
    for agent in case.get("expected_trace", []):
        checks.append({"name": f"Trace包含{agent}", "passed": agent in trace_agents, "expected": agent, "actual": "、".join(trace_agents), "message": "Trace命中" if agent in trace_agents else "Trace缺失关键工具"})
    passed_count = sum(1 for item in checks if item["passed"])
    score = round(passed_count / max(len(checks), 1) * 100, 1)
    status = "pass" if score >= 90 else "warn" if score >= 75 else "fail"
    failed_checks = [item for item in checks if not item["passed"]]
    return {
        "id": case["id"],
        "name": case["name"],
        "query": case["query"],
        "status": status,
        "score": score,
        "passed_count": passed_count,
        "check_count": len(checks),
        "selected_pool": pool,
        "top_models": [_vehicle_name(item) for item in recs[:5]],
        "trace_agents": trace_agents,
        "feedback_policy_rules": feedback_policy.get("applied_rules", []),
        "obsidian_note": result.get("obsidian_note", {}),
        "checks": checks,
        "failed_checks": failed_checks,
        "diagnosis": "Agent端到端链路稳定" if status == "pass" else "；".join(item["message"] for item in failed_checks[:3]),
    }


def run_agent_regression_evaluation(persist: bool = True) -> Dict[str, Any]:
    cases = [_check_agent_case(case) for case in AGENT_REGRESSION_CASES]
    passed = sum(1 for item in cases if item["status"] == "pass")
    warned = sum(1 for item in cases if item["status"] == "warn")
    failed = sum(1 for item in cases if item["status"] == "fail")
    average_score = round(sum(item["score"] for item in cases) / max(len(cases), 1), 1)
    summary = {
        "case_count": len(cases),
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "pass_rate": round(passed / max(len(cases), 1) * 100, 1),
        "average_score": average_score,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    note = persist_agent_regression(summary, cases) if persist else {}
    return {"summary": summary, "cases": cases, "obsidian_note": note}


def persist_agent_regression(summary: Dict[str, Any], cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    lines = []
    for item in cases:
        lines.append(f"- {item['name']}：{item['score']} 分，候选池 {item['selected_pool']}，Trace：{'、'.join(item['trace_agents'])}，Top：{'、'.join(item['top_models'][:3])}")
    body = f"""
## Agent 端到端回归概览

- 用例数：{summary['case_count']}
- 通过率：{summary['pass_rate']}%
- 平均分：{summary['average_score']}
- 通过/警告/失败：{summary['passed']} / {summary['warned']} / {summary['failed']}

## 用例结果

{chr(10).join(lines)}

## 覆盖链路

- 候选池选择
- Rank 排序
- FeedbackPolicyTool 策略稳定性
- RAG 证据检索
- RiskCheckTool 风险检查
- ObsidianCaseWriterTool 长期记忆写入

## 关联节点

- [[Agent工作台]]
- [[反馈驱动复盘]]
- [[推荐链路]]
"""
    title = f"Agent端到端回归评估-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    path = write_note("07-测试样例", title, {
        "type": "agent-regression-report",
        "source": "api/evaluation/agent-regression",
        "created_at": summary["generated_at"],
        "tags": ["Agent回归", "端到端测试", "质量闭环"],
    }, body)
    return {"title": title, "path": path}
