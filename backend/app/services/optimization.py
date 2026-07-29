from datetime import datetime
from typing import Any, Dict, List

from app.database import recommendation_feedback_summary
from app.services.evaluation import run_recommendation_evaluation
from app.services.obsidian_vault import write_note


def _item(title: str, priority: str, source: str, evidence: str, action: str) -> Dict[str, Any]:
    score = {"P0": 100, "P1": 85, "P2": 65, "P3": 45}.get(priority, 50)
    return {"title": title, "priority": priority, "score": score, "source": source, "evidence": evidence, "action": action}


def _safe_rate(part: int, total: int) -> float:
    return round(part / max(total, 1) * 100, 1)


def generate_feedback_review(persist: bool = True) -> Dict[str, Any]:
    feedback = recommendation_feedback_summary()
    pool_rows = feedback.get("pool_rows", [])
    scene_rows = feedback.get("scene_rows", [])
    model_rows = feedback.get("model_rows", [])
    best_pool = pool_rows[0] if pool_rows else {}
    weakest_pool = sorted(pool_rows, key=lambda item: (item.get("positive_rate", 0), -item.get("total", 0)))[0] if pool_rows else {}
    risky_scenes = [item for item in scene_rows if item.get("negative", 0) > 0]
    negative_models = [item for item in model_rows if item.get("negative", 0) > 0]
    insights: List[Dict[str, Any]] = []

    if best_pool:
        insights.append({
            "type": "candidate_pool_quality",
            "title": f"候选池质量最高：{best_pool.get('candidate_pool')}",
            "evidence": f"{best_pool.get('total', 0)} 条反馈，正反馈率 {best_pool.get('positive_rate', 0)}%。",
            "action": "在相似画像下优先保留该候选池策略，同时继续观察样本量变化。",
        })
    if weakest_pool and weakest_pool != best_pool:
        insights.append({
            "type": "candidate_pool_risk",
            "title": f"候选池需复盘：{weakest_pool.get('candidate_pool')}",
            "evidence": f"{weakest_pool.get('total', 0)} 条反馈，正反馈率 {weakest_pool.get('positive_rate', 0)}%。",
            "action": "检查该候选池是否存在价格字段估算、车型地域不匹配或解释不足问题。",
        })
    for scene in risky_scenes[:3]:
        insights.append({
            "type": "scenario_risk",
            "title": f"高风险场景：{scene.get('scenario')}",
            "evidence": f"{scene.get('total', 0)} 条反馈，负反馈率 {scene.get('negative_rate', 0)}%。",
            "action": "补充该场景的风险核验话术，并在推荐报告中提前提示限制条件。",
        })
    for model in negative_models[:3]:
        insights.append({
            "type": "model_review",
            "title": f"负反馈车型复盘：{model.get('model_name')}",
            "evidence": f"正反馈 {model.get('positive', 0)} 条，负反馈 {model.get('negative', 0)} 条。",
            "action": "回看该车型入选原因，判断是否需要降低权重或强化不适用说明。",
        })
    if feedback.get("total", 0) < 10:
        insights.append({
            "type": "sample_size",
            "title": "反馈样本仍需扩大",
            "evidence": f"当前仅 {feedback.get('total', 0)} 条反馈，低于阶段 D 的 10 条验证目标。",
            "action": "继续在推荐卡片引导销售顾问标注正负反馈和具体原因。",
        })

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feedback_total": feedback.get("total", 0),
        "positive_rate": feedback.get("positive_rate", 0),
        "pool_count": len(pool_rows),
        "risky_scene_count": len(risky_scenes),
        "negative_model_count": len(negative_models),
        "insight_count": len(insights),
    }
    note = persist_feedback_review_note(summary, feedback, insights) if persist else {}
    return {"summary": summary, "feedback": feedback, "insights": insights, "obsidian_note": note}


def generate_optimization_insights(persist: bool = True) -> Dict[str, Any]:
    evaluation = run_recommendation_evaluation(False)
    feedback = recommendation_feedback_summary()
    review = generate_feedback_review(False)
    items: List[Dict[str, Any]] = []

    for case in evaluation["cases"]:
        if case["status"] != "pass" or case["score"] < 95:
            items.append(_item(
                f"优化测试用例：{case['name']}",
                "P1" if case["score"] < 85 else "P2",
                "固定评估集",
                f"当前得分 {case['score']}，诊断：{case['diagnosis']}",
                "补充对应场景的车型标签，检查画像解析字段和评分权重。",
            ))

    for row in feedback.get("model_rows", []):
        if row.get("negative", 0) > 0:
            items.append(_item(
                f"复盘负反馈车型：{row['model_name']}",
                "P1" if row.get("negative", 0) >= row.get("positive", 0) else "P2",
                "人工反馈",
                f"负反馈 {row.get('negative', 0)} 条，正反馈 {row.get('positive', 0)} 条。",
                "查看推荐原因和用户画像，判断是价格、空间、补能还是品牌偏好导致不匹配。",
            ))

    for row in feedback.get("pool_rows", []):
        if row.get("total", 0) >= 2 and row.get("positive_rate", 0) < 70:
            items.append(_item(
                f"复盘候选池策略：{row['candidate_pool']}",
                "P1",
                "候选池反馈聚合",
                f"{row.get('total', 0)} 条反馈，正反馈率 {row.get('positive_rate', 0)}%。",
                "检查候选池选择规则是否过度依赖真实扩展数据或本地精选，必要时调整 auto 策略。",
            ))

    for row in feedback.get("scene_rows", [])[:5]:
        if row.get("negative", 0) > 0 and row.get("negative_rate", 0) >= 50:
            items.append(_item(
                f"补强高风险场景：{row['scenario']}",
                "P1" if row.get("total", 0) >= 2 else "P2",
                "场景反馈聚合",
                f"{row.get('total', 0)} 条反馈，负反馈率 {row.get('negative_rate', 0)}%。",
                "在推荐报告中增加该场景专属风险提示，并补充对比候选。",
            ))

    if feedback.get("total", 0) < 10:
        items.append(_item(
            "扩大人工反馈样本量",
            "P2",
            "反馈闭环",
            f"当前仅有 {feedback.get('total', 0)} 条反馈，样本不足以稳定调权。",
            "在每次推荐后引导销售顾问点击准确/需优化，并补充具体原因。",
        ))

    if evaluation["summary"].get("pass_rate", 0) >= 100 and feedback.get("negative", 0) == 0:
        items.append(_item(
            "补充更难的真实客户用例",
            "P3",
            "质量评估",
            "当前固定评估集全部通过，缺少边界场景压力测试。",
            "加入预算冲突、品牌偏好冲突、无家充纯电偏好、豪华商务等复杂用例。",
        ))

    items.sort(key=lambda item: item["score"], reverse=True)
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "item_count": len(items),
        "p1_count": sum(1 for item in items if item["priority"] in {"P0", "P1"}),
        "feedback_total": feedback.get("total", 0),
        "evaluation_pass_rate": evaluation["summary"].get("pass_rate", 0),
        "pool_count": len(feedback.get("pool_rows", [])),
        "risky_scene_count": review["summary"].get("risky_scene_count", 0),
    }
    note = persist_optimization_note(summary, items) if persist else {}
    return {"summary": summary, "items": items, "evaluation": evaluation["summary"], "feedback": feedback, "feedback_review": review, "obsidian_note": note}


def persist_feedback_review_note(summary: Dict[str, Any], feedback: Dict[str, Any], insights: List[Dict[str, Any]]) -> Dict[str, str]:
    pool_lines = [f"- [[候选池-{row['candidate_pool']}]]：{row['total']} 条，正反馈率 {row['positive_rate']}%" for row in feedback.get("pool_rows", [])]
    scene_lines = [f"- [[场景-{row['scenario']}]]：{row['total']} 条，负反馈率 {row['negative_rate']}%" for row in feedback.get("scene_rows", [])[:8]]
    insight_lines = [f"- {item['title']}：{item['action']}（证据：{item['evidence']}）" for item in insights]
    body = f"""
## 反馈复盘概览

- 反馈样本：{summary['feedback_total']}
- 正反馈率：{summary['positive_rate']}%
- 候选池维度：{summary['pool_count']}
- 高风险场景：{summary['risky_scene_count']}
- 负反馈车型：{summary['negative_model_count']}

## 候选池质量

{chr(10).join(pool_lines) if pool_lines else '暂无候选池反馈'}

## 场景风险

{chr(10).join(scene_lines) if scene_lines else '暂无场景反馈'}

## Agent 复盘结论

{chr(10).join(insight_lines) if insight_lines else '暂无复盘结论'}

## 关联节点

- [[推荐链路]]
- [[反馈驱动复盘]]
- [[推荐质量评估]]
- [[自生长知识库方案]]
"""
    title = f"反馈复盘-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    path = write_note("06-迭代记录", title, {
        "type": "feedback-review",
        "source": "api/feedback/review",
        "created_at": summary["generated_at"],
        "tags": ["反馈复盘", "Agent复盘", "反馈闭环"],
    }, body)
    return {"title": title, "path": path}


def persist_optimization_note(summary: Dict[str, Any], items: List[Dict[str, Any]]) -> Dict[str, str]:
    lines = [f"- [{item['priority']}] {item['title']}：{item['action']}（证据：{item['evidence']}）" for item in items]
    body = f"""
## 优化建议概览

- 建议数：{summary['item_count']}
- 高优先级：{summary['p1_count']}
- 反馈样本：{summary['feedback_total']}
- 评估通过率：{summary['evaluation_pass_rate']}%
- 候选池维度：{summary.get('pool_count', 0)}
- 高风险场景：{summary.get('risky_scene_count', 0)}

## 建议列表

{chr(10).join(lines) if lines else '暂无需要处理的优化建议'}

## 关联节点

- [[推荐链路]]
- [[用户画像解析优化]]
- [[推荐质量评估]]
- [[反馈驱动复盘]]
- [[自生长知识库方案]]
"""
    title = f"优化建议-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    path = write_note("06-迭代记录", title, {
        "type": "optimization-insight",
        "source": "api/optimization/insights",
        "created_at": summary["generated_at"],
        "tags": ["优化建议", "反馈闭环", "质量评估"],
    }, body)
    return {"title": title, "path": path}
