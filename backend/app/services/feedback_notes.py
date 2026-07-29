from typing import Any, Dict

from app.services.obsidian_vault import write_note


def save_feedback_note(feedback: Dict[str, Any]) -> Dict[str, str]:
    rating_label = {"positive": "正反馈", "negative": "负反馈"}.get(feedback.get("rating"), "中性反馈")
    model_name = feedback.get("model_name") or "未命名车型"
    candidate_pool = feedback.get("candidate_pool") or "unknown"
    scenario_tags = feedback.get("scenario_tags") or []
    title = f"推荐反馈-{feedback.get('created_at', '').replace(':', '').replace(' ', '-')}-{model_name}"
    body = f"""
## 反馈结论

- 车型：{model_name}
- 反馈：{rating_label}
- 原因：{feedback.get('reason') or '未填写'}
- 候选池：[[候选池-{candidate_pool}]]
- 场景标签：{'、'.join(scenario_tags) if scenario_tags else '未标注'}

## 原始需求

{feedback.get('query') or '未记录'}

## 画像摘要

- 预算：{feedback.get('profile', {}).get('budget_max') or '未明确'}
- 城市：{feedback.get('profile', {}).get('city') or '未明确'}
- 关注点：{'、'.join(feedback.get('profile', {}).get('concerns') or []) or '未明确'}

## 关联节点

- [[推荐链路]]
- [[推荐质量评估]]
- [[自生长知识库方案]]
- [[反馈驱动复盘]]
{chr(10).join(f'- [[场景-{tag}]]' for tag in scenario_tags)}
"""
    path = write_note("07-测试样例", title, {
        "type": "recommendation-feedback",
        "source": "api/recommendation-feedback",
        "created_at": feedback.get("created_at", ""),
        "tags": ["推荐反馈", rating_label, model_name, candidate_pool, *scenario_tags],
    }, body)
    return {"title": title, "path": path}
