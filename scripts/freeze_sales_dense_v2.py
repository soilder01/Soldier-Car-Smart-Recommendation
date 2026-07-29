#!/usr/bin/env python3
"""Freeze sales_dense_v2 data and training protocol without model calls."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "model_training" / "grpo" / "formal_v4" / "restart_1"
GRPO_DIR = ROOT / "data" / "model_training" / "grpo"
VEHICLE_CSV = ROOT / "data" / "vehicles" / "vehicle_database.csv"

SALES_OUT = BASE_DIR / "sales_dense_v2_rebuild.jsonl"
TRAIN_MIX_OUT = BASE_DIR / "sales_dense_v2_train_mix.jsonl"
MANIFEST_OUT = BASE_DIR / "sales_dense_v2_manifest.json"
PROTOCOL_OUT = BASE_DIR / "sales_dense_v2_train_protocol.json"

REWARD_FN = ROOT / "training" / "grpo" / "reward_fn.py"
NEWANCHOR_DATASET = BASE_DIR / "powered_dev_newanchor_128.jsonl"
NEWANCHOR_PROTOCOL = BASE_DIR / "newanchor_eval_protocol.json"
FOURWAY_EVAL = BASE_DIR / "newanchor_fourway_evaluations.jsonl"
FORMAL_TRAIN = GRPO_DIR / "reward_train_expanded_v4.jsonl"
FORMAL_INPUT_MANIFEST = GRPO_DIR / "grpo_expanded_v4_input_manifest.json"
FINAL_MANIFEST = ROOT / "data" / "model_training" / "eval" / "grpo_final_held_out_manifest.json"
HELD_RESERVATION_MANIFEST = (
    ROOT / "data" / "model_training" / "sft_freeze" / "reward_reservation_manifest.json"
)
PREVIOUS_TARGETED = BASE_DIR / "sales_targeted_rebuild_v1.jsonl"
CKPT100 = (
    ROOT
    / "checkpoints"
    / "grpo"
    / "formal_v4"
    / "restart_1"
    / "sales_targeted"
    / "checkpoint-100"
)

EXPECTED_REWARD_SHA = "325ad44feb83ec37c35babfed4bddb928cf400788e07735eb4631fc4af6962c8"
EXPECTED_NEWANCHOR_SHA = "e74de770e8dbc95dbbf813d87fa9b38e631941ec464f790410b2c86be41c0c2b"
EXPECTED_NEWANCHOR_PROTOCOL_SHA = "3f53319f9f4158dd1e282cbefa9f84dcdf0e295c018023cae8d1557020a94783"
EXPECTED_FOURWAY_SHA = "beeb1d41c212e3d65428ae7a380b135889786d1838cbc6656976a16e4a9a79a5"
EXPECTED_CKPT100_ADAPTER_SHA = "13942f45d58bf04843017188bc3dc35a87a6775c0fdd513e5ed9662b5119e7ff"

HISTORICAL_BOUNDARY = 650
SALES_CANDIDATE_COUNT = 4900
RECOMMEND_CANDIDATE_COUNT = 41472
SALES_DENSE_COUNT = 512
RECOMMEND_REHEARSAL_COUNT = 128
SELECTION_SEED = "sales-dense-v2"

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
DEFAULT_FIELDS = ("energy_type", "cltc_range", "fast_charge", "wheelbase", "price_min")
DENSE_DEFAULT_FIELDS = (
    "energy_type",
    "cltc_range",
    "fast_charge",
    "price_min",
    "price_max",
    "monthly_sales",
    "safety_score",
    "adas_level",
)
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


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_sha_sidecar(artifact: Path) -> None:
    with Path(str(artifact) + ".sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{sha256_file(artifact)}  {artifact.name}\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def vehicle_rows() -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    rows: dict[str, dict[str, str]] = {}
    lines: dict[str, int] = {}
    with VEHICLE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            name = f"{row['brand']} {row['model']}"
            rows[name] = row
            lines[name] = line
    return rows, lines


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


def append_unique(items: list[str], values: tuple[str, ...]) -> None:
    for value in values:
        if value and value not in items:
            items.append(value)


def matched_concerns(query: str) -> tuple[str, ...]:
    words: list[str] = []
    for phrase, anchors in CONCERN_RULES:
        if phrase in query:
            append_unique(words, anchors)
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
            append_unique(words, (value,))
    return tuple(words)


def sales_anchors(query: str) -> tuple[str, ...]:
    return matched_concerns(query) or DEFAULT_CONCERNS


def recommend_anchors(query: str) -> tuple[str, ...]:
    match = re.search(r"关注(.+?)，请推荐", query)
    if not match:
        return sales_anchors(query)
    tokens: list[str] = []
    for token in re.split(r"[、和/ ]+", match.group(1)):
        token = token.strip()
        if token:
            append_unique(tokens, (token,))
    return tuple(tokens) or DEFAULT_CONCERNS


def field_tokens_for_query(query: str, anchors: tuple[str, ...]) -> list[str]:
    tokens = list(anchors)
    for token, _mapped in FIELD_RULES:
        if token in query and token not in tokens:
            tokens.append(token)
    return tokens


def fields_for_tokens(tokens: list[str], *, limit: int, defaults: tuple[str, ...]) -> tuple[str, ...]:
    fields: list[str] = []
    for field_token in tokens:
        for token, mapped_fields in FIELD_RULES:
            if not (token == field_token or token in field_token or field_token in token):
                continue
            for field in mapped_fields:
                if field not in fields:
                    fields.append(field)
                if len(fields) >= limit:
                    return tuple(fields)
    for field in defaults:
        if field not in fields:
            fields.append(field)
        if len(fields) >= limit:
            break
    return tuple(fields)


def field_specific_anchors(query: str, field_name: str, anchors: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for token, mapped_fields in FIELD_RULES:
        if field_name not in mapped_fields:
            continue
        if token in query or any(token == anchor or token in anchor or anchor in token for anchor in anchors):
            append_unique(values, (token,))
    if not values:
        values = list(anchors[:3])
    return tuple(values)


def score_vehicle(row: dict[str, str], query: str) -> tuple[float, str]:
    budget = parse_budget(query)
    preferred_type = parse_type(query)
    preferred_energy = parse_energy(query)
    concerns = sales_anchors(query)
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
    scored = sorted(
        (score_vehicle(row, query) for row in catalog.values()),
        key=lambda item: (-item[0], item[1]),
    )
    return [name for _score, name in scored[: (3 if intent == "recommend" else 1)]]


def claim(
    *,
    entity: str,
    field_name: str,
    row: dict[str, str],
    line: int,
    anchors: tuple[str, ...],
) -> dict[str, Any]:
    canonical_attribute, column, suffix, aliases = ATTRIBUTES[field_name]
    value = f"{int(row[column]) / 10000:g}万元" if field_name in {"price_min", "price_max"} else f"{row[column]}{suffix}"
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


def instruction(intent: str) -> str:
    common = "只能使用上面的冻结证据；实体、字段和值必须逐字一致，不得补写其他数字。"
    if intent == "recommend":
        return f"{common} 使用推荐/首选/备选等决策词绑定真实候选车型，并给出证据理由。"
    return (
        f"{common} 写销售沟通话术，必须包含建议/试驾/核验/了解/跟进中的动作词；"
        "至少写出3条不同字段证据，每条证据句都要同时出现目标车型、字段名或别名、字段值、以及本次query关注点词。"
    )


def build_sales_case(row: dict[str, Any], catalog: dict[str, dict[str, str]], lines: dict[str, int]) -> dict[str, Any]:
    query = row["query"]
    anchors = sales_anchors(query)
    fields = fields_for_tokens(
        field_tokens_for_query(query, anchors),
        limit=5,
        defaults=DENSE_DEFAULT_FIELDS,
    )
    target = select_models("sales", query, catalog)[0]
    claims = [
        claim(
            entity=target,
            field_name=field,
            row=catalog[target],
            line=lines[target],
            anchors=field_specific_anchors(query, field, anchors),
        )
        for field in fields
    ]
    if len({item["canonical_attribute"] for item in claims}) < 5:
        raise ValueError(f"{row['id']}: dense sales claims are not distinct")
    return {
        "id": f"sales-dense-v2-{row['candidate_index']:05d}",
        "split": "sales_dense_v2",
        "intent": "sales",
        "query": query,
        "target_entities": [target],
        "intent_response_spec": {
            "query_anchor_tokens": list(anchors),
            "query_attribute_anchors": [ATTRIBUTES[field][0] for field in fields],
            "minimum_supported_claims": 3,
            "decision_tokens": [],
            "communication_action_tokens": list(COMMUNICATION_ACTION_TOKENS),
        },
        "evidence_claims": claims,
        "generation_instruction": instruction("sales"),
        "known_reward_contract_risks": [],
        "candidate_index": row["candidate_index"],
        "selection_sha256": row["selection_sha256"],
        "recipe_version": "sales_dense_v2",
    }


def build_recommend_case(row: dict[str, Any], catalog: dict[str, dict[str, str]], lines: dict[str, int]) -> dict[str, Any]:
    query = row["query"]
    anchors = recommend_anchors(query)
    fields = fields_for_tokens(
        field_tokens_for_query(query, anchors),
        limit=3,
        defaults=DEFAULT_FIELDS,
    )
    targets = select_models("recommend", query, catalog)
    claims = [
        claim(
            entity=entity,
            field_name=field,
            row=catalog[entity],
            line=lines[entity],
            anchors=anchors,
        )
        for entity in targets
        for field in fields
    ]
    return {
        "id": f"sales-dense-v2-recommend-rehearsal-{row['candidate_index']:05d}",
        "split": "sales_dense_v2_recommend_rehearsal",
        "intent": "recommend",
        "query": query,
        "target_entities": targets,
        "intent_response_spec": {
            "query_anchor_tokens": list(anchors),
            "query_attribute_anchors": [ATTRIBUTES[field][0] for field in fields],
            "minimum_supported_claims": 1,
            "decision_tokens": list(DECISION_TOKENS),
            "communication_action_tokens": [],
        },
        "evidence_claims": claims,
        "generation_instruction": instruction("recommend"),
        "known_reward_contract_risks": [],
        "candidate_index": row["candidate_index"],
        "selection_sha256": row["selection_sha256"],
        "recipe_version": "sales_dense_v2_recommend_rehearsal",
    }


def sales_queries(count: int) -> list[str]:
    concerns = [
        "插混车电池安全和保修",
        "低使用成本和能耗",
        "没有家充的补能价值",
        "国产新能源品牌高端感",
        "辅助驾驶安全边界",
        "预算有限但想要大空间",
        "两款车型之间的犹豫",
        "等待降价且不夸大权益",
        "后期维保网点",
        "工具未返回的油耗数值",
        "增程技术路线",
        "智能座舱学习成本",
        "试驾后觉得动力一般",
        "二胎家庭后备箱",
        "首次购买新能源",
        "最低价承诺",
        "品牌保值率顾虑",
        "纯电和插混场景分析",
        "交付周期敏感",
        "社交形象与预算平衡",
    ]
    approaches = [
        "销售怎么回应才克制？",
        "请给出开场、异议处理和跟进建议。",
        "请生成不承诺政策的沟通话术。",
        "应该如何引导用户核验官方信息？",
        "请给出以实际体验为主的沟通方案。",
    ]
    customer_profiles = ["首次购车", "已试驾", "带孩子出行", "经常跨城", "城市通勤", "关注可靠性", "预算敏感"]
    touchpoints = ["线上首轮沟通", "到店接待", "试驾后跟进", "交付前确认", "售后回访", "电话回访", "微信沟通"]
    return [
        f"客户关注{concern}，{approach.rstrip('。')}。当前为{profile}，处于{touchpoint}。"
        for concern in concerns
        for approach in approaches
        for profile in customer_profiles
        for touchpoint in touchpoints
    ][:count]


def recommend_queries(count: int) -> list[str]:
    budgets = [12, 15, 18, 20, 22, 25, 28, 30, 35, 40, 45, 50]
    families = ["单人通勤", "情侣出行", "三口之家", "二胎家庭", "老人接送孩子", "家用兼商务接待"]
    types = ["SUV", "轿车", "MPV"]
    energies = ["纯电", "插混", "增程", "新能源"]
    concerns = [
        "空间和能耗",
        "安全和续航",
        "快充和补能",
        "舒适和后备箱",
        "智驾和座舱",
        "可靠性和保值",
        "驾驶感和加速",
        "品牌和低调感",
    ]
    usages = ["城市通勤为主", "周末带家人出游", "偶尔跨城", "经常高速出行", "没有家充", "有固定家充"]
    rows = [
        f"预算{budget}万，{family}，{usage}，想买{energy}{vehicle_type}，关注{concern}，请推荐。"
        for budget in budgets
        for family in families
        for vehicle_type in types
        for energy in energies
        for concern in concerns
        for usage in usages
    ]
    return rows[:count]


def candidate_rows(intent: str, queries: list[str]) -> list[dict[str, Any]]:
    rows = []
    for index, query in enumerate(queries, start=1):
        if index <= HISTORICAL_BOUNDARY:
            continue
        digest = hashlib.sha256(
            f"{SELECTION_SEED}:{intent}:{index}:{normalize(query)}".encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "id": f"sales-dense-v2-source-{intent}-{index:05d}",
                "intent": intent,
                "candidate_index": index,
                "query": query,
                "selection_sha256": digest,
                "normalized_id_sha256": normalized_sha256(f"sales-dense-v2-source-{intent}-{index:05d}"),
                "normalized_query_sha256": normalized_sha256(query),
            }
        )
    return rows


def forbidden_sets() -> dict[str, Any]:
    formal = load_jsonl(FORMAL_TRAIN)
    newanchor = load_jsonl(NEWANCHOR_DATASET)
    previous = load_jsonl(PREVIOUS_TARGETED)
    final_manifest = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))
    reservation = json.loads(HELD_RESERVATION_MANIFEST.read_text(encoding="utf-8"))

    by_intent_index = set()
    query_shas = set()
    sources: dict[str, int] = {}
    for label, rows in (
        ("formal_train", formal),
        ("powered_dev_newanchor", newanchor),
        ("previous_sales_targeted_v1", previous),
    ):
        sources[label] = len(rows)
        for row in rows:
            if row.get("intent") in {"sales", "recommend"} and "candidate_index" in row:
                by_intent_index.add((row["intent"], int(row["candidate_index"])))
            if row.get("query"):
                query_shas.add(normalized_sha256(row["query"]))
            elif row.get("normalized_query_sha256"):
                query_shas.add(row["normalized_query_sha256"])
    sources["final40_manifest_provenance"] = len(final_manifest["provenance"])
    for row in final_manifest["provenance"]:
        if row.get("intent") in {"sales", "recommend"} and "candidate_index" in row:
            by_intent_index.add((row["intent"], int(row["candidate_index"])))
        query_shas.add(row["normalized_query_sha256"])
    held_entries = [entry for entry in reservation["entries"] if entry["source"] == "held_out"]
    reward_visible_entries = [entry for entry in reservation["entries"] if entry["source"] == "reward_visible"]
    sources["heldout40_reservation_manifest_entries"] = len(held_entries)
    sources["reward_visible_reservation_manifest_entries"] = len(reward_visible_entries)
    for entry in held_entries + reward_visible_entries:
        query_shas.add(entry["query_sha256"])
    return {"by_intent_index": by_intent_index, "query_shas": query_shas, "source_counts": sources}


def clean_candidates(intent: str, rows: list[dict[str, Any]], forbidden: dict[str, Any]) -> list[dict[str, Any]]:
    clean = []
    for row in rows:
        key = (intent, int(row["candidate_index"]))
        if key in forbidden["by_intent_index"]:
            continue
        if row["normalized_query_sha256"] in forbidden["query_shas"]:
            continue
        clean.append(row)
    return clean


def select_sales(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def hardness(row: dict[str, Any]) -> tuple[int, int, str]:
        anchors = sales_anchors(row["query"])
        fields = fields_for_tokens(field_tokens_for_query(row["query"], anchors), limit=5, defaults=DENSE_DEFAULT_FIELDS)
        hard_tokens = {"预算", "价格", "权益", "交付", "维保", "售后", "保值", "增程", "插混", "纯电", "核验", "安全", "智驾"}
        return (
            sum(token in hard_tokens for token in anchors),
            len(anchors) + len(fields),
            row["selection_sha256"],
        )

    selected = sorted(rows, key=lambda row: (-hardness(row)[0], -hardness(row)[1], hardness(row)[2], row["candidate_index"]))
    return selected[:SALES_DENSE_COUNT]


def select_recommend(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["selection_sha256"], row["candidate_index"]))[:RECOMMEND_REHEARSAL_COUNT]


def redline_hashes() -> dict[str, str]:
    expected = {
        "training/grpo/reward_fn.py": EXPECTED_REWARD_SHA,
        "data/model_training/grpo/formal_v4/restart_1/powered_dev_newanchor_128.jsonl": EXPECTED_NEWANCHOR_SHA,
        "data/model_training/grpo/formal_v4/restart_1/newanchor_eval_protocol.json": EXPECTED_NEWANCHOR_PROTOCOL_SHA,
        "data/model_training/grpo/formal_v4/restart_1/newanchor_fourway_evaluations.jsonl": EXPECTED_FOURWAY_SHA,
        "checkpoints/grpo/formal_v4/restart_1/sales_targeted/checkpoint-100/adapter_model.safetensors": EXPECTED_CKPT100_ADAPTER_SHA,
    }
    observed = {}
    for relative, digest in expected.items():
        actual = sha256_file(ROOT / relative)
        if actual != digest:
            raise RuntimeError(f"SHA mismatch: {relative}")
        observed[relative] = actual
    return observed


def quality(cases: list[dict[str, Any]]) -> dict[str, Any]:
    sales = [case for case in cases if case["intent"] == "sales"]
    return {
        "sales_count": len(sales),
        "sales_evidence_claim_count_distribution": dict(sorted(Counter(len(case["evidence_claims"]) for case in sales).items())),
        "sales_minimum_supported_claims_distribution": dict(sorted(Counter(case["intent_response_spec"]["minimum_supported_claims"] for case in sales).items())),
        "sales_query_anchor_count_distribution": dict(sorted(Counter(len(case["intent_response_spec"]["query_anchor_tokens"]) for case in sales).items())),
        "sales_query_attribute_anchor_count_distribution": dict(sorted(Counter(len(case["intent_response_spec"]["query_attribute_anchors"]) for case in sales).items())),
        "communication_action_token_full_count": sum(
            tuple(case["intent_response_spec"]["communication_action_tokens"]) == COMMUNICATION_ACTION_TOKENS
            for case in sales
        ),
        "strict_claim_checks": {
            "all_sales_have_at_least_5_distinct_claim_attributes": all(
                len({claim["canonical_attribute"] for claim in case["evidence_claims"]}) >= 5
                for case in sales
            ),
            "all_sales_claims_have_entity_field_value_and_anchor": all(
                claim["canonical_entity"]
                and claim["canonical_attribute"]
                and claim["canonical_value"]
                and claim["anchor_tokens"]
                for case in sales
                for claim in case["evidence_claims"]
            ),
        },
    }


def main() -> int:
    for path in (
        SALES_OUT,
        TRAIN_MIX_OUT,
        MANIFEST_OUT,
        PROTOCOL_OUT,
        Path(str(SALES_OUT) + ".sha256"),
        Path(str(TRAIN_MIX_OUT) + ".sha256"),
        Path(str(MANIFEST_OUT) + ".sha256"),
        Path(str(PROTOCOL_OUT) + ".sha256"),
    ):
        if path.exists():
            raise FileExistsError(f"refusing overwrite: {path}")

    hashes = redline_hashes()
    forbidden = forbidden_sets()
    catalog, lines = vehicle_rows()

    sales_candidates = clean_candidates("sales", candidate_rows("sales", sales_queries(SALES_CANDIDATE_COUNT)), forbidden)
    recommend_candidates = clean_candidates(
        "recommend",
        candidate_rows("recommend", recommend_queries(RECOMMEND_CANDIDATE_COUNT)),
        forbidden,
    )
    selected_sales = select_sales(sales_candidates)
    selected_recommend = select_recommend(recommend_candidates)
    if len(selected_sales) != SALES_DENSE_COUNT or len(selected_recommend) != RECOMMEND_REHEARSAL_COUNT:
        raise RuntimeError("not enough clean candidates for sales_dense_v2")

    sales_cases = [build_sales_case(row, catalog, lines) for row in selected_sales]
    recommend_cases = [build_recommend_case(row, catalog, lines) for row in selected_recommend]
    train_mix = sales_cases + recommend_cases
    train_mix.sort(key=lambda case: (case["intent"] != "sales", case["selection_sha256"], case["candidate_index"]))
    counts = Counter(case["intent"] for case in train_mix)
    if counts != {"sales": SALES_DENSE_COUNT, "recommend": RECOMMEND_REHEARSAL_COUNT}:
        raise RuntimeError(f"unexpected train mix counts: {dict(counts)}")

    write_jsonl_exclusive(SALES_OUT, sales_cases)
    write_sha_sidecar(SALES_OUT)
    write_jsonl_exclusive(TRAIN_MIX_OUT, train_mix)
    write_sha_sidecar(TRAIN_MIX_OUT)

    final_query_shas = {row["normalized_query_sha256"] for row in json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))["provenance"]}
    held_query_shas = {
        entry["query_sha256"]
        for entry in json.loads(HELD_RESERVATION_MANIFEST.read_text(encoding="utf-8"))["entries"]
        if entry["source"] == "held_out"
    }
    formal_query_shas = {
        row.get("normalized_query_sha256") or normalized_sha256(row["query"])
        for row in load_jsonl(FORMAL_TRAIN)
    }
    newanchor_query_shas = {normalized_sha256(row["query"]) for row in load_jsonl(NEWANCHOR_DATASET)}
    train_mix_query_shas = {normalized_sha256(case["query"]) for case in train_mix}
    train_mix_keys = {(case["intent"], int(case["candidate_index"])) for case in train_mix}
    manifest = {
        "status": "sales_dense_v2_frozen_without_training_or_inference",
        "scope": {
            "reward_fn_modified": False,
            "reward_fn_sha256": hashes["training/grpo/reward_fn.py"],
            "held_out_40_body_read": False,
            "final_40_body_read": False,
            "model_loaded": False,
            "cloud_called": False,
        },
        "recipe": {
            "version": "sales_dense_v2",
            "sales_dense_count": SALES_DENSE_COUNT,
            "recommend_rehearsal_count": RECOMMEND_REHEARSAL_COUNT,
            "total_prompts_per_logical_epoch": len(train_mix),
            "sales_ratio": SALES_DENSE_COUNT / len(train_mix),
            "sales_claims_per_case": 5,
            "sales_minimum_supported_claims": 3,
            "selection_seed": SELECTION_SEED,
            "qualitative_difference_from_v1": [
                "5 distinct query-bound evidence claims per sales prompt instead of 3",
                "minimum_supported_claims raised from 1 to 3 in data spec",
                "sales mix ratio raised from 60% to 80%",
                "recommend rehearsal uses clean unused prompt candidates instead of formal train rows",
            ],
        },
        "source_counts": {
            "sales_tail_candidates_before_exclusion": SALES_CANDIDATE_COUNT - HISTORICAL_BOUNDARY,
            "sales_clean_candidates": len(sales_candidates),
            "recommend_tail_candidates_before_exclusion": RECOMMEND_CANDIDATE_COUNT - HISTORICAL_BOUNDARY,
            "recommend_clean_candidates": len(recommend_candidates),
            "forbidden_source_counts": forbidden["source_counts"],
        },
        "counts": {
            "train_mix_total": len(train_mix),
            "train_mix_per_intent": dict(sorted(counts.items())),
        },
        "quality_checks": quality(train_mix),
        "overlap_checks": {
            "formal_train_query_sha_overlap": len(train_mix_query_shas & formal_query_shas),
            "powered_dev_newanchor_query_sha_overlap": len(train_mix_query_shas & newanchor_query_shas),
            "final_40_query_sha_overlap_manifest_only": len(train_mix_query_shas & final_query_shas),
            "held_out_40_query_sha_overlap_manifest_only": len(train_mix_query_shas & held_query_shas),
            "formal_train_candidate_index_overlap": len(
                train_mix_keys
                & {
                    (row["intent"], int(row["candidate_index"]))
                    for row in load_jsonl(FORMAL_TRAIN)
                    if row.get("intent") in {"sales", "recommend"}
                }
            ),
            "powered_dev_newanchor_candidate_index_overlap": len(
                train_mix_keys
                & {
                    (row["intent"], int(row["candidate_index"]))
                    for row in load_jsonl(NEWANCHOR_DATASET)
                    if row.get("intent") in {"sales", "recommend"}
                }
            ),
        },
        "artifacts": {
            "sales_dense_jsonl": {
                "path": str(SALES_OUT.relative_to(ROOT)),
                "records": len(sales_cases),
                "sha256": sha256_file(SALES_OUT),
            },
            "train_mix_jsonl": {
                "path": str(TRAIN_MIX_OUT.relative_to(ROOT)),
                "records": len(train_mix),
                "sha256": sha256_file(TRAIN_MIX_OUT),
            },
            "manifest": {"path": str(MANIFEST_OUT.relative_to(ROOT))},
        },
        "redline_hashes": hashes,
    }
    for key, value in manifest["overlap_checks"].items():
        if value != 0:
            raise RuntimeError(f"overlap check failed: {key}={value}")
    write_json_exclusive(MANIFEST_OUT, manifest)
    write_sha_sidecar(MANIFEST_OUT)

    protocol = {
        "status": "frozen_before_sales_dense_v2_training",
        "mode": "continue_from_sales_targeted_checkpoint_100_sales_dense_v2",
        "red_lines": {
            "reward_fn_modified": False,
            "reward_fn_sha256_must_equal": EXPECTED_REWARD_SHA,
            "no_held_out_40_body_read": True,
            "no_final_40_body_read": True,
            "do_not_modify_frozen_protocols_or_eval_sets": True,
            "no_secret_or_base_url_plaintext_output": True,
        },
        "initialization": {
            "from_checkpoint_label": "sales-targeted-checkpoint-100",
            "path": "checkpoints/grpo/formal_v4/restart_1/sales_targeted/checkpoint-100/adapter_model.safetensors",
            "adapter_sha256": EXPECTED_CKPT100_ADAPTER_SHA,
            "continue_training_not_from_scratch": True,
        },
        "training_data_mixture": {
            "path": str(TRAIN_MIX_OUT.relative_to(ROOT)),
            "sha256": sha256_file(TRAIN_MIX_OUT),
            "total_prompts_per_logical_epoch": len(train_mix),
            "ratio_summary": "sales_dense_v2 80% + clean recommend rehearsal 20%",
            "sales_dense_count": SALES_DENSE_COUNT,
            "recommend_rehearsal_count": RECOMMEND_REHEARSAL_COUNT,
        },
        "training_hyperparameters": {
            "fp16": True,
            "bf16": False,
            "beta": 0.01,
            "num_generations": 8,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_prompt_length": 2560,
            "max_completion_length": 512,
            "learning_rate": 2e-6,
            "lr_scheduler": "cosine",
            "warmup_steps": 40,
            "max_optimizer_steps": 200,
            "save_every_steps": 50,
            "health_window_steps": 50,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "optimizer": "paged_adamw_8bit",
            "seed": 20260725,
        },
        "abort_policy": {
            "kl_abort": "only if health-window KL mean >= 10 * KL0_mean",
            "nonfinite_abort": True,
            "oom_abort": True,
            "kl_p95": "monitor_only",
        },
        "pre_registered_gate": {
            "evaluation_protocol": "data/model_training/grpo/formal_v4/restart_1/newanchor_eval_protocol.json",
            "evaluation_dataset": "data/model_training/grpo/formal_v4/restart_1/powered_dev_newanchor_128.jsonl",
            "candidate_checkpoint_steps": [50, 100, 150, 200],
            "sales_mean_core_must_be_strictly_greater_than": 0.8864,
            "sales_threshold_source": "cloud_seedpro_ark_ep_masked sales newanchor score rounded per user instruction",
            "recommend_mean_core_must_be_at_least": 0.800018918256223,
            "recommend_threshold_source": "checkpoint-300 recommend newanchor mean_core from frozen newanchor_fourway_evaluations.jsonl",
            "failure_action": "if no checkpoint satisfies both gates, mark this round invalid; do not switch to composite selection",
            "selection_if_multiple_pass": "highest sales mean_core; tie earlier checkpoint step",
        },
        "post_training_evaluation": {
            "reuse_existing_cloud_newanchor_result": True,
            "existing_fourway_path": "data/model_training/grpo/formal_v4/restart_1/newanchor_fourway_evaluations.jsonl",
            "existing_fourway_sha256": EXPECTED_FOURWAY_SHA,
            "raw_answers_persisted": False,
            "per_prompt_trace_required": True,
        },
        "expected_hashes": {
            **hashes,
            str(SALES_OUT.relative_to(ROOT)): sha256_file(SALES_OUT),
            str(TRAIN_MIX_OUT.relative_to(ROOT)): sha256_file(TRAIN_MIX_OUT),
            str(MANIFEST_OUT.relative_to(ROOT)): sha256_file(MANIFEST_OUT),
        },
    }
    write_json_exclusive(PROTOCOL_OUT, protocol)
    write_sha_sidecar(PROTOCOL_OUT)

    print(
        json.dumps(
            {
                "status": "frozen",
                "sales_sha256": sha256_file(SALES_OUT),
                "train_mix_sha256": sha256_file(TRAIN_MIX_OUT),
                "manifest_sha256": sha256_file(MANIFEST_OUT),
                "protocol_sha256": sha256_file(PROTOCOL_OUT),
                "counts": manifest["counts"],
                "quality_checks": manifest["quality_checks"],
                "overlap_checks": manifest["overlap_checks"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
