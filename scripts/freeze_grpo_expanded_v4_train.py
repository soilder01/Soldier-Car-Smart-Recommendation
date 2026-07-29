#!/usr/bin/env python3
"""Freeze expanded formal_v4 GRPO train data from candidate_index > 650."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from data_synth.generate_500perintent_sft import (
    _compare_expected_model_names,
    build_queries,
)
from scripts import evaluate_model_outputs as evaluator
from scripts.freeze_grpo_final_eval import SOURCE_SPECS


ROOT = Path(__file__).resolve().parents[1]
GRPO_DIR = ROOT / "data" / "model_training" / "grpo"
TRAIN_OUT = GRPO_DIR / "reward_train_expanded_v4.jsonl"
TRAIN_SHA = TRAIN_OUT.with_suffix(".sha256")
MANIFEST_OUT = GRPO_DIR / "reward_train_expanded_v4_manifest.json"
MANIFEST_SHA = MANIFEST_OUT.with_suffix(".sha256")
INPUT_MANIFEST_OUT = GRPO_DIR / "grpo_expanded_v4_input_manifest.json"
INPUT_MANIFEST_SHA = INPUT_MANIFEST_OUT.with_suffix(".sha256")
VEHICLE_CSV = ROOT / "data" / "vehicles" / "vehicle_database.csv"
HELD_OUT = ROOT / "data" / "model_training" / "eval" / "held_out.jsonl"
FINAL_OUT = (
    ROOT / "data" / "model_training" / "eval" / "grpo_final_held_out.jsonl"
)
LEGACY_DEV = GRPO_DIR / "reward_dev_4.jsonl"
LEGACY_DEV_IDS = (
    "reward-recommend-002",
    "reward-compare-004",
    "reward-knowledge-003",
    "reward-sales-004",
)
SELECTION_SEED = "grpo-expanded-v4-train"
HISTORICAL_BOUNDARY = 650
TARGET_COUNTS = {
    "recommend": 265,
    "compare": 6,
    "knowledge": 265,
    "sales": 264,
}

ATTRIBUTES = {
    "energy_type": ("vehicle.energy_type", "energy_type", "", ("能源类型", "能源")),
    "price_min": ("vehicle.price_min", "price_min", "元", ("最低价", "价格", "预算")),
    "price_max": ("vehicle.price_max", "price_max", "元", ("最高价", "价格", "预算")),
    "cltc_range": ("specs.cltc_range", "cltc_range", "km", ("CLTC续航", "续航")),
    "battery_kwh": ("specs.battery_kwh", "battery_kwh", "kWh", ("电池容量", "电池")),
    "fast_charge": ("specs.fast_charge", "fast_charge_minutes", "分钟", ("快充", "补能")),
    "seats": ("specs.seats", "seats", "座", ("座位数", "座位")),
    "drive_type": ("specs.drive_type", "drive_type", "", ("驱动形式", "动力")),
    "adas_level": ("specs.adas_level", "adas_level", "", ("辅助驾驶等级", "智驾")),
    "smart_cockpit": ("specs.smart_cockpit", "smart_cockpit", "", ("智能座舱", "座舱")),
    "wheelbase": ("specs.wheelbase", "wheelbase", "mm", ("轴距", "空间")),
    "trunk_volume": ("specs.trunk_volume", "trunk_volume", "L", ("后备箱", "空间")),
    "safety_score": ("specs.safety_score", "safety_score", "分", ("安全评分", "安全")),
    "monthly_sales": ("market.monthly_sales", "monthly_sales", "辆/月", ("月销量", "销量", "市场表现")),
}
DEFAULT_FIELDS = ("energy_type", "cltc_range", "fast_charge", "wheelbase")
DECISION_TOKENS = ("推荐", "首选", "建议优先", "备选")
COMMUNICATION_ACTION_TOKENS = ("建议", "试驾", "核验", "了解", "跟进")
DEFAULT_CONCERNS = ("空间", "补能")
CONCERN_RULES = (
    ("插混车电池安全和保修", ("能源类型", "电池", "安全", "保修")),
    ("低使用成本和能耗", ("使用成本", "能耗", "补能")),
    ("没有家充的补能价值", ("家充", "补能", "续航")),
    ("国产新能源品牌高端感", ("品牌", "高端感", "座舱")),
    ("辅助驾驶安全边界", ("智驾", "安全", "边界")),
    ("预算有限但想要大空间", ("预算", "价格", "空间")),
    ("两款车型之间的犹豫", ("对比", "选择", "核验")),
    ("等待降价且不夸大权益", ("价格", "权益", "核验")),
    ("后期维保网点", ("维保", "售后", "品牌")),
    ("工具未返回的油耗数值", ("油耗", "能耗", "核验")),
    ("增程技术路线", ("增程", "能源类型", "补能")),
    ("智能座舱学习成本", ("座舱", "学习成本", "体验")),
    ("试驾后觉得动力一般", ("试驾", "动力", "驾驶感")),
    ("二胎家庭后备箱", ("后备箱", "空间", "家庭")),
    ("首次购买新能源", ("新能源", "入门", "核验")),
    ("最低价承诺", ("价格", "权益", "承诺")),
    ("品牌保值率顾虑", ("品牌", "保值", "市场表现")),
    ("纯电和插混场景分析", ("纯电", "插混", "能源类型", "场景")),
    ("交付周期敏感", ("交付", "周期", "核验")),
    ("社交形象与预算平衡", ("品牌", "预算", "价格")),
    ("预算敏感", ("预算", "价格", "权益")),
    ("经常跨城", ("续航", "补能")),
    ("城市通勤", ("能耗", "通勤")),
    ("关注可靠性", ("可靠性", "安全")),
    ("带孩子出行", ("安全", "空间")),
    ("已试驾", ("试驾", "体验")),
    ("交付前确认", ("交付", "核验")),
    ("售后回访", ("售后", "跟进")),
    ("电话回访", ("跟进", "了解")),
    ("微信沟通", ("跟进", "了解")),
)
FIELD_RULES = (
    ("最低价", ("price_min", "price_max", "energy_type")),
    ("价格", ("price_min", "price_max", "energy_type")),
    ("预算", ("price_min", "price_max", "energy_type")),
    ("权益", ("price_min", "price_max", "monthly_sales")),
    ("承诺", ("price_min", "price_max", "monthly_sales")),
    ("增程", ("energy_type", "cltc_range", "fast_charge")),
    ("插混", ("energy_type", "cltc_range", "fast_charge")),
    ("纯电", ("energy_type", "cltc_range", "fast_charge")),
    ("能源类型", ("energy_type", "cltc_range", "fast_charge")),
    ("续航", ("cltc_range", "fast_charge", "energy_type")),
    ("补能", ("fast_charge", "cltc_range", "energy_type")),
    ("快充", ("fast_charge", "cltc_range", "energy_type")),
    ("家充", ("fast_charge", "cltc_range", "energy_type")),
    ("能耗", ("energy_type", "cltc_range", "fast_charge")),
    ("油耗", ("energy_type", "cltc_range", "fast_charge")),
    ("使用成本", ("energy_type", "cltc_range", "fast_charge")),
    ("空间", ("wheelbase", "trunk_volume", "seats")),
    ("后备箱", ("trunk_volume", "wheelbase", "seats")),
    ("家庭", ("seats", "wheelbase", "trunk_volume")),
    ("安全", ("safety_score", "adas_level", "energy_type")),
    ("可靠性", ("safety_score", "monthly_sales", "energy_type")),
    ("智驾", ("adas_level", "safety_score", "smart_cockpit")),
    ("边界", ("adas_level", "safety_score", "energy_type")),
    ("座舱", ("smart_cockpit", "adas_level", "energy_type")),
    ("学习成本", ("smart_cockpit", "energy_type", "cltc_range")),
    ("体验", ("drive_type", "smart_cockpit", "energy_type")),
    ("试驾", ("drive_type", "adas_level", "energy_type")),
    ("动力", ("drive_type", "energy_type", "cltc_range")),
    ("驾驶感", ("drive_type", "energy_type", "adas_level")),
    ("品牌", ("monthly_sales", "smart_cockpit", "energy_type")),
    ("高端感", ("smart_cockpit", "monthly_sales", "energy_type")),
    ("保值", ("monthly_sales", "price_min", "price_max")),
    ("市场表现", ("monthly_sales", "price_min", "price_max")),
    ("维保", ("monthly_sales", "safety_score", "energy_type")),
    ("售后", ("monthly_sales", "safety_score", "energy_type")),
    ("交付", ("monthly_sales", "energy_type", "cltc_range")),
    ("周期", ("monthly_sales", "energy_type", "cltc_range")),
    ("核验", ("energy_type", "cltc_range", "fast_charge")),
    ("对比", ("energy_type", "cltc_range", "fast_charge")),
    ("选择", ("energy_type", "cltc_range", "fast_charge")),
)


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().split()).casefold()


def normalized_sha256(value: str) -> str:
    return hashlib.sha256(normalize(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_sha(path: Path, artifact: Path) -> None:
    with path.open("x", encoding="ascii") as handle:
        handle.write(f"{sha256_file(artifact)}  {artifact.name}\n")


def vehicle_rows() -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    rows: dict[str, dict[str, str]] = {}
    lines: dict[str, int] = {}
    with VEHICLE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            name = f"{row['brand']} {row['model']}"
            rows[name] = row
            lines[name] = line
    return rows, lines


def compact(value: str) -> str:
    return normalize(value).replace(" ", "")


def resolve_model(name: str, catalog: dict[str, dict[str, str]]) -> str | None:
    key = compact(name)
    for full in catalog:
        if key == compact(full) or key == compact(full.split(" ", 1)[-1]):
            return full
    for full in catalog:
        if key in compact(full) or compact(full.split(" ", 1)[-1]) in key:
            return full
    return None


def parse_budget(query: str) -> int:
    match = re.search(r"预算(\d+)万", query)
    return int(match.group(1)) * 10000 if match else 300000


def parse_type(query: str) -> str:
    for value in ("SUV", "MPV", "轿车", "跑车"):
        if value.lower() in query.lower():
            return value
    return ""


def parse_energy(query: str) -> str:
    for value in ("插混", "增程", "纯电"):
        if value in query:
            return value
    return ""


def _append_unique(items: list[str], values: tuple[str, ...]) -> None:
    for value in values:
        if value and value not in items:
            items.append(value)


def matched_concerns(query: str) -> tuple[str, ...]:
    words: list[str] = []
    for phrase, anchors in CONCERN_RULES:
        if phrase in query:
            _append_unique(words, anchors)
    for value in (
        "智驾",
        "座舱",
        "可靠性",
        "保值",
        "续航",
        "补能",
        "空间",
        "安全",
        "品牌",
        "驾驶感",
        "油耗",
        "预算",
        "价格",
        "权益",
        "增程",
        "插混",
        "纯电",
        "交付",
        "维保",
        "售后",
        "核验",
    ):
        if value in query:
            _append_unique(words, (value,))
    return tuple(words)


def parse_concerns(query: str) -> tuple[str, ...]:
    return matched_concerns(query) or DEFAULT_CONCERNS


def score_vehicle(row: dict[str, str], query: str) -> tuple[float, str]:
    budget = parse_budget(query)
    preferred_type = parse_type(query)
    preferred_energy = parse_energy(query)
    concerns = parse_concerns(query)
    price_min = int(row["price_min"])
    price_max = int(row["price_max"])
    mid = (price_min + price_max) / 2
    score = 100 - min(abs(mid - budget) / max(budget, 1) * 20, 30)
    if preferred_type and preferred_type in row["vehicle_type"]:
        score += 10
    if preferred_energy:
        if row["energy_type"] == preferred_energy:
            score += 12
        elif preferred_energy == "插混" and row["energy_type"] == "增程":
            score += 4
        else:
            score -= 10
    text = f"{row.get('highlights','')} {row.get('suitable_scenarios','')} {row.get('adas_level','')} {row.get('smart_cockpit','')}"
    score += sum(2 for concern in concerns if concern in text)
    return score, f"{row['brand']} {row['model']}"


def select_models(intent: str, query: str, catalog: dict[str, dict[str, str]]) -> list[str]:
    if intent == "compare":
        names = _compare_expected_model_names(query)
        resolved = [resolve_model(name, catalog) for name in names]
        return [name for name in resolved if name is not None][:2]
    scored = sorted(
        (score_vehicle(row, query) for row in catalog.values()),
        key=lambda item: (-item[0], item[1]),
    )
    if intent == "recommend":
        return [name for _score, name in scored[:3]]
    if intent == "sales":
        return [scored[0][1]]
    return []


def claim(entity: str, field_name: str, row: dict[str, str], line: int, anchors: tuple[str, ...]) -> dict[str, Any]:
    canonical_attribute, column, suffix, aliases = ATTRIBUTES[field_name]
    if field_name in {"price_min", "price_max"}:
        value = f"{int(row[column]) / 10000:g}万元"
    else:
        value = f"{row[column]}{suffix}"
    model_alias = entity.split(" ", 1)[-1]
    return {
        "canonical_entity": entity,
        "canonical_attribute": canonical_attribute,
        "canonical_value": value,
        "source_tool": "frozen_local_catalog",
        "source_locator": f"data/vehicles/vehicle_database.csv:{line}.{column}",
        "entity_aliases": [entity.replace(" ", ""), model_alias],
        "attribute_aliases": list(aliases),
        "anchor_tokens": list(anchors),
    }


def knowledge_claims(query: str, index: int) -> list[dict[str, Any]]:
    if "补能" in query or "家充" in query:
        values = ("没有家充需关注补能条件", "需以官方实时信息核验")
        anchors = ("补能", "家充", "核验")
    elif "辅助驾驶" in query:
        values = ("辅助驾驶不能替代驾驶员", "需以官方实时信息核验")
        anchors = ("辅助驾驶", "安全", "核验")
    else:
        values = ("基于本地知识库证据回答", "需以官方实时信息核验")
        anchors = ("知识库", "核验")
    return [
        {
            "canonical_entity": "知识库",
            "canonical_attribute": "knowledge.anchor",
            "canonical_value": value,
            "source_tool": "frozen_local_knowledge_base",
            "source_locator": f"data/knowledge_base/*:{index}.{offset}",
            "entity_aliases": ["知识库", *anchors],
            "attribute_aliases": ["知识点", "事实"],
            "anchor_tokens": list(anchors),
        }
        for offset, value in enumerate(values, start=1)
    ]


def fields_for_query(query: str) -> tuple[str, ...]:
    fields = []
    field_tokens = list(parse_concerns(query))
    for token, _mapped_fields in FIELD_RULES:
        if token in query and token not in field_tokens:
            field_tokens.append(token)
    for field_token in field_tokens:
        for token, mapped_fields in FIELD_RULES:
            if not (
                token == field_token
                or token in field_token
                or field_token in token
            ):
                continue
            for field in mapped_fields:
                if len(fields) >= 3:
                    break
                if field not in fields:
                    fields.append(field)
            if len(fields) >= 3:
                break
    for field in DEFAULT_FIELDS:
        if len(fields) >= 3:
            break
        if field not in fields:
            fields.append(field)
    return tuple(fields[:3])


def instruction(intent: str, target_count: int) -> str:
    common = "只能使用上面的冻结证据；实体、字段和值必须逐字一致，不得补写其他数字。"
    if intent == "recommend":
        return f"{common} 使用推荐/首选/备选等决策词绑定真实候选车型，并给出证据理由。"
    if intent == "compare":
        return f"{common} 对两个目标车型分别给出至少一条证据命题。"
    if intent == "knowledge":
        return f"{common} 直接回答核心问题，至少给出两个与知识库 anchor 绑定的命题。"
    return f"{common} 写简洁销售沟通话术，包含建议/试驾/核验/了解/跟进动作，并给出产品证据。"


def build_case(row: dict[str, Any], catalog: dict[str, dict[str, str]], lines: dict[str, int]) -> dict[str, Any]:
    intent = row["intent"]
    query = row["query"]
    anchors = parse_concerns(query)
    targets = select_models(intent, query, catalog)
    claims: list[dict[str, Any]] = []
    if intent == "knowledge":
        claims = knowledge_claims(query, row["candidate_index"])
        targets = ("知识库",)
        fields = ("knowledge.anchor",)
    else:
        field_names = fields_for_query(query)
        for entity in targets:
            for field_name in field_names:
                claims.append(
                    claim(
                        entity=entity,
                        field_name=field_name,
                        row=catalog[entity],
                        line=lines[entity],
                        anchors=anchors,
                    )
                )
        fields = tuple(ATTRIBUTES[name][0] for name in field_names)
    minimum = 2 if intent == "knowledge" else len(targets) if intent == "compare" else 1
    return {
        "id": row["id"],
        "split": "train",
        "intent": intent,
        "query": query,
        "target_entities": list(targets),
        "intent_response_spec": {
            "query_anchor_tokens": list(anchors),
            "query_attribute_anchors": list(fields),
            "minimum_supported_claims": minimum,
            "decision_tokens": list(DECISION_TOKENS) if intent == "recommend" else [],
            "communication_action_tokens": list(COMMUNICATION_ACTION_TOKENS) if intent == "sales" else [],
        },
        "evidence_claims": claims,
        "generation_instruction": instruction(intent, len(targets)),
        "known_reward_contract_risks": (
            ["knowledge uses local KB anchor claims; prior probes show this intent remains a known weak-signal area"]
            if intent == "knowledge"
            else []
        ),
        "candidate_index": row["candidate_index"],
        "selection_sha256": row["selection_sha256"],
    }


def candidate_rows() -> list[dict[str, Any]]:
    final_queries = {normalized_sha256(row["query"]) for row in load_jsonl(FINAL_OUT)}
    rows: list[dict[str, Any]] = []
    for intent, spec in SOURCE_SPECS.items():
        source_intent = spec["source_intent"]
        queries = build_queries(source_intent, spec["candidate_count"])
        for index, query in enumerate(queries, start=1):
            if index <= HISTORICAL_BOUNDARY:
                continue
            if normalized_sha256(query) in final_queries:
                continue
            digest = hashlib.sha256(
                f"{SELECTION_SEED}:{intent}:{index}:{normalize(query)}".encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "id": f"expanded-v4-{intent}-{index:05d}",
                    "intent": intent,
                    "source_intent": source_intent,
                    "candidate_index": index,
                    "query": query,
                    "selection_sha256": digest,
                    "normalized_id_sha256": normalized_sha256(f"expanded-v4-{intent}-{index:05d}"),
                    "normalized_query_sha256": normalized_sha256(query),
                }
            )
    selected: list[dict[str, Any]] = []
    for intent, count in TARGET_COUNTS.items():
        candidates = [row for row in rows if row["intent"] == intent]
        candidates.sort(key=lambda row: (row["selection_sha256"], row["candidate_index"]))
        if len(candidates) < count:
            raise ValueError(f"not enough candidates for {intent}: {len(candidates)} < {count}")
        selected.extend(candidates[:count])
    selected.sort(key=lambda row: (row["intent"], row["selection_sha256"]))
    return selected


def assert_isolated(rows: list[dict[str, Any]]) -> dict[str, Any]:
    held = load_jsonl(HELD_OUT)
    final = load_jsonl(FINAL_OUT)
    row_ids = {row["normalized_id_sha256"] for row in rows}
    row_queries = {row["normalized_query_sha256"] for row in rows}
    checks = {}
    for label, other in (("held_out_40", held), ("final_40", final)):
        other_ids = {normalized_sha256(row["id"]) for row in other}
        other_queries = {normalized_sha256(row["query"]) for row in other}
        checks[label] = {
            "normalized_id_sha256_overlap": len(row_ids & other_ids),
            "normalized_query_sha256_overlap": len(row_queries & other_queries),
        }
        if checks[label]["normalized_id_sha256_overlap"] or checks[label]["normalized_query_sha256_overlap"]:
            raise ValueError(f"expanded train overlaps {label}")
    return checks


def main() -> None:
    for path in (TRAIN_OUT, TRAIN_SHA, MANIFEST_OUT, MANIFEST_SHA, INPUT_MANIFEST_OUT, INPUT_MANIFEST_SHA):
        if path.exists():
            raise FileExistsError(f"refusing overwrite: {path}")
    rows = candidate_rows()
    isolation = assert_isolated(rows)
    catalog, lines = vehicle_rows()
    cases = [build_case(row, catalog, lines) for row in rows]
    dev_rows = load_jsonl(LEGACY_DEV)
    if [row["id"] for row in dev_rows] != list(LEGACY_DEV_IDS):
        raise ValueError("fixed dev-4 prompt order drift")
    # Reuse the existing frozen dev evidence/spec cases.
    old_manifest = json.loads((GRPO_DIR / "grpo_signal_probe_input_manifest.json").read_text(encoding="utf-8"))
    dev_cases_by_id = {case["id"]: case for case in old_manifest["cases"] if case["split"] == "dev"}
    dev_cases = [dev_cases_by_id[identifier] for identifier in LEGACY_DEV_IDS]
    for case in dev_cases:
        case["split"] = "dev"
    manifest = {
        "format_version": 1,
        "status": "frozen_without_training_or_inference",
        "purpose": "expanded formal_v4 GRPO train set and fixed dev-4 probe",
        "selection": {
            "seed": SELECTION_SEED,
            "source": "candidate_index > 650 from data_synth.generate_500perintent_sft.build_queries",
            "excluded": "GRPO final-40 normalized query SHA",
            "target_counts": TARGET_COUNTS,
            "balancing_note": "compare has only 6 available tail candidates after excluding final-40; all 6 are included, the remaining 794 rows are balanced across recommend/knowledge/sales.",
        },
        "counts": {
            "train": len(rows),
            "dev": len(dev_cases),
            "train_per_intent": dict(sorted(Counter(row["intent"] for row in rows).items())),
            "dev_per_intent": dict(sorted(Counter(case["intent"] for case in dev_cases).items())),
        },
        "isolation": isolation,
        "knowledge_policy": {
            "included": True,
            "train_count": sum(row["intent"] == "knowledge" for row in rows),
            "known_limitation": "knowledge uses local KB anchor claims; prior signal probe found weak but nonzero anchor-bound pass evidence.",
        },
        "artifacts": {},
    }
    write_jsonl_exclusive(TRAIN_OUT, rows)
    manifest["artifacts"]["train"] = {
        "path": str(TRAIN_OUT.relative_to(ROOT)),
        "sha256": sha256_file(TRAIN_OUT),
        "records": len(rows),
    }
    input_manifest = {
        "format_version": 1,
        "status": "frozen_before_formal_v4_training",
        "purpose": "formal_v4 expanded GRPO training input manifest with evidence/spec",
        "counts": manifest["counts"],
        "train_source": manifest["artifacts"]["train"],
        "fixed_dev4_source": {
            "path": str(LEGACY_DEV.relative_to(ROOT)),
            "sha256": sha256_file(LEGACY_DEV),
            "records": len(dev_cases),
        },
        "cases": cases + dev_cases,
    }
    write_json_exclusive(INPUT_MANIFEST_OUT, input_manifest)
    manifest["artifacts"]["input_manifest"] = {
        "path": str(INPUT_MANIFEST_OUT.relative_to(ROOT)),
        "sha256": sha256_file(INPUT_MANIFEST_OUT),
        "records": len(input_manifest["cases"]),
    }
    write_json_exclusive(MANIFEST_OUT, manifest)
    write_sha(TRAIN_SHA, TRAIN_OUT)
    write_sha(INPUT_MANIFEST_SHA, INPUT_MANIFEST_OUT)
    write_sha(MANIFEST_SHA, MANIFEST_OUT)
    print(json.dumps({"counts": manifest["counts"], "isolation": isolation, "train_sha256": sha256_file(TRAIN_OUT), "manifest_sha256": sha256_file(MANIFEST_OUT), "input_manifest_sha256": sha256_file(INPUT_MANIFEST_OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
