import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from app.config import DATA_DIR, PROJECT_DIR
from app.database import list_vehicles

VAULT_DIR = PROJECT_DIR / "obsidian-vault"
LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


def ensure_vault() -> None:
    for folder in [
        "00-入口",
        "01-车型知识",
        "02-用户画像",
        "03-推荐系统",
        "04-RAG知识库",
        "05-自生长机制",
        "06-迭代记录",
        "07-测试样例",
        "08-推荐案例",
    ]:
        (VAULT_DIR / folder).mkdir(parents=True, exist_ok=True)
    (VAULT_DIR / ".obsidian").mkdir(parents=True, exist_ok=True)


def slug_name(value: str) -> str:
    return re.sub(r"[\\/:*?\"<>|\s]+", "-", value.strip()).strip("-") or "untitled"


def split_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
        return meta if isinstance(meta, dict) else {}, parts[2].strip()
    except Exception:
        return {}, parts[2].strip()


def title_from_file(path: Path, body: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def excerpt(body: str, limit: int = 160) -> str:
    lines = [line.strip("# *-") for line in body.splitlines() if line.strip() and not line.startswith("---")]
    text = " ".join(lines).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def read_nodes() -> List[Dict[str, Any]]:
    ensure_vault()
    nodes = []
    for path in sorted(VAULT_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue
        text = path.read_text(encoding="utf-8")
        meta, body = split_frontmatter(text)
        title = str(meta.get("title") or title_from_file(path, body))
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        links = sorted(set(LINK_RE.findall(body)))
        nodes.append({
            "id": str(path.relative_to(VAULT_DIR)).replace("\\", "/"),
            "title": title,
            "path": str(path.relative_to(VAULT_DIR)).replace("\\", "/"),
            "type": meta.get("type", "note"),
            "module": meta.get("module", ""),
            "tags": tags,
            "links": links,
            "excerpt": excerpt(body),
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return nodes


def graph() -> Dict[str, Any]:
    nodes = read_nodes()
    title_to_id = {node["title"]: node["id"] for node in nodes}
    edges = []
    for node in nodes:
        for link in node["links"]:
            target = title_to_id.get(link)
            if target:
                edges.append({"source": node["id"], "target": target, "label": link})
    tag_counter = Counter(tag for node in nodes for tag in node["tags"])
    type_counter = Counter(node["type"] for node in nodes)
    return {
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "tag_count": len(tag_counter),
            "vehicle_node_count": type_counter.get("vehicle", 0) + type_counter.get("vehicle-knowledge", 0),
            "recommendation_case_count": type_counter.get("recommendation-case", 0),
        },
        "nodes": nodes,
        "edges": edges,
        "tag_distribution": dict(tag_counter.most_common(12)),
        "type_distribution": dict(type_counter),
    }


def list_recommendation_cases(limit: int = 20) -> Dict[str, Any]:
    cases = [node for node in read_nodes() if node.get("type") == "recommendation-case"]
    cases.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return {
        "total": len(cases),
        "cases": cases[:max(1, min(limit, 100))],
    }


def write_note(folder: str, title: str, meta: Dict[str, Any], body: str) -> str:
    ensure_vault()
    path = VAULT_DIR / folder / f"{slug_name(title)}.md"
    payload = {"title": title, **meta}
    frontmatter = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).strip()
    path.write_text(f"---\n{frontmatter}\n---\n\n# {title}\n\n{body.strip()}\n", encoding="utf-8")
    return str(path.relative_to(VAULT_DIR)).replace("\\", "/")


def seed_from_project_data() -> Dict[str, Any]:
    ensure_vault()
    created = []
    vehicles = list_vehicles()[:10]
    for vehicle in vehicles:
        title = f"{vehicle['brand']} {vehicle['model']}"
        body = f"""
## 车型定位

- 能源类型：{vehicle['energy_type']}
- 车型：{vehicle['vehicle_type']}
- 价格区间：{round(vehicle['price_min'] / 10000, 1)}-{round(vehicle['price_max'] / 10000, 1)} 万
- CLTC 续航：{vehicle['cltc_range']}km
- 智驾等级：{vehicle['adas_level']}
- 座位数：{vehicle['seats']} 座

## 推荐关系

- 关联 [[车型库总览]]
- 关联 [[推荐链路]]
- 适合沉淀到 [[自生长知识库方案]]

## 卖点

{vehicle['highlights']}

## 风险提示

{vehicle['weaknesses']}
"""
        created.append(write_note("01-车型知识", title, {
            "type": "vehicle",
            "source": "data/vehicles/vehicle_database.csv",
            "tags": ["车型", vehicle["energy_type"], vehicle["vehicle_type"], vehicle["brand"]],
        }, body))

    scenarios = [
        ("三口之家城市通勤选车", "预算 25 万以内，三口之家，每天通勤 50 公里，有家充，关注续航、空间和智驾。", ["家庭用车", "城市通勤", "用户画像"]),
        ("无家充用户能源路线选择", "没有家充时，需要重点比较纯电、插混和增程在补能便利性、长途成本、保养成本上的差异。", ["补能", "增程", "插混", "纯电"]),
        ("高频竞品对比场景", "用户常把特斯拉 Model Y、小鹏 G6、比亚迪宋 L EV 放在一起比较，需要解释价格、续航、智驾、空间和品牌差异。", ["竞品对比", "销售话术"]),
    ]
    for title, content, tags in scenarios:
        body = f"""
## 场景描述

{content}

## 关联节点

- [[用户画像解析优化]]
- [[推荐链路]]
- [[车型库总览]]

## 前端测试价值

该节点用于补充 Obsidian 知识图谱展示，也可以作为推荐链路和客服问答的测试样例。
"""
        created.append(write_note("07-测试样例", title, {
            "type": "scenario",
            "source": "seed_from_project_data",
            "tags": tags,
        }, body))

    kb_dir = DATA_DIR / "knowledge_base"
    if kb_dir.exists():
        for path in sorted(kb_dir.glob("*.md"))[:6]:
            text = path.read_text(encoding="utf-8")
            title = path.stem
            body = f"""
## 来源

`data/knowledge_base/{path.name}`

## 摘要

{excerpt(text, 320)}

## 关联节点

- [[RAG知识库优化]]
- [[推荐链路]]
- [[自生长知识库方案]]
"""
            created.append(write_note("04-RAG知识库", title, {
                "type": "rag-source",
                "source": f"data/knowledge_base/{path.name}",
                "tags": ["RAG", "知识来源", "可追溯回答"],
            }, body))

    return {"created_count": len(created), "created": created, "graph": graph()}


def save_recommendation_case(query: str, profile: Dict[str, Any], recommendations: List[Dict[str, Any]], answer: str = "", explainability: Dict[str, Any] = None, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    ensure_vault()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    top = recommendations[0] if recommendations else {}
    top_name = f"{top.get('brand', '')} {top.get('model', '')}".strip() or "待推荐车型"
    title = f"推荐案例-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{top_name}"
    explainability = explainability or {}
    sources = sources or []
    pool = explainability.get("pool_decision", {})
    rec_lines = []
    for index, item in enumerate(recommendations[:5], 1):
        name = f"{item.get('brand', '')} {item.get('model', '')}".strip()
        reasons = "；".join(item.get("reasons", [])[:3]) or "待补充推荐理由"
        rec_lines.append(f"{index}. [[{name}]]：推荐分 {item.get('score', '--')}，{reasons}")
    comparison_lines = []
    for item in explainability.get("top_comparisons", [])[:3]:
        selected = "；".join(item.get("why_selected", [])[:3])
        cautions = "；".join(item.get("cautions", [])[:2])
        comparison_lines.append(f"### Top {item.get('rank')}：[[{item.get('model')}]]\n\n- 入选原因：{selected}\n- 适合人群：{item.get('best_for')}\n- 为什么不是其他车型：{item.get('why_not_others')}\n- 核验项：{cautions}")
    evidence_lines = [f"- {item.get('domain') or '证据'}：{item.get('source') or '本地规则'}｜{item.get('content', '')[:120]}" for item in (explainability.get("evidence_sources") or sources)[:5]]
    concerns = profile.get("concerns") or []
    body = f"""
## 用户需求

{query}

## 解析画像

- 预算上限：{profile.get('budget_max') or '未明确'}
- 城市：{profile.get('city') or '未明确'}
- 家庭人数：{profile.get('family_size') or '未明确'}
- 通勤距离：{profile.get('commute_km') or '未明确'} km
- 家充条件：{profile.get('has_home_charger')}
- 关注点：{'、'.join(concerns) if concerns else '未明确'}

## 候选池决策

- 候选池：{pool.get('selected_pool') or '未记录'}
- 选择原因：{pool.get('reason') or '未记录'}

## 推荐结果

{chr(10).join(rec_lines) if rec_lines else '暂无推荐结果'}

## Top3 可解释对比

{chr(10).join(comparison_lines) if comparison_lines else '暂无结构化对比'}

## 谨慎/不推荐原因

{chr(10).join(f'- {item}' for item in explainability.get('not_recommended', [])) or '暂无'}

## 风险核验清单

{chr(10).join(f'- {item}' for item in explainability.get('risk_checklist', [])) or '暂无'}

## 试驾与跟进动作

{chr(10).join(f'- {item}' for item in explainability.get('follow_up_actions', [])) or '暂无'}

## 数据证据来源

{chr(10).join(evidence_lines) if evidence_lines else '暂无证据来源'}

## Agent 推荐报告摘要

{excerpt(answer, 800) if answer else '暂无报告摘要'}

## 关联节点

- [[用户画像解析优化]]
- [[推荐链路]]
- [[自生长知识库方案]]
- [[车型库总览]]
- [[Agent 可解释推荐报告]]
"""
    path = write_note("08-推荐案例", title, {
        "type": "recommendation-case",
        "source": "api/agent/recommend",
        "created_at": now,
        "candidate_pool": pool.get("selected_pool", ""),
        "top_model": top_name,
        "tags": ["推荐案例", "用户画像", "自生长知识库", "Agent记忆", top.get("brand", "车型")],
    }, body)
    return {"path": path, "title": title, "top_model": top_name, "candidate_pool": pool.get("selected_pool", ""), "memory_sections": ["画像", "候选池", "Top3对比", "风险", "跟进动作", "证据"]}
