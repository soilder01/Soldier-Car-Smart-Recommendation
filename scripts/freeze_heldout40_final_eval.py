#!/usr/bin/env python3
"""Freeze held-out-40 reward cases and final evaluation protocol."""

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
OUT_DATASET = BASE_DIR / "held_out_40_frozen_eval.jsonl"
OUT_PROTOCOL = BASE_DIR / "held_out_40_final_protocol.json"
HELD_OUT = ROOT / "data" / "model_training" / "eval" / "held_out.jsonl"
REWARD_FN = ROOT / "training" / "grpo" / "reward_fn.py"
VEHICLE_CSV = ROOT / "data" / "vehicles" / "vehicle_database.csv"
NEWANCHOR = BASE_DIR / "powered_dev_newanchor_128.jsonl"
NEWANCHOR_PROTOCOL = BASE_DIR / "newanchor_eval_protocol.json"
LOCAL_VS_CLOUD_RULE = BASE_DIR / "local_vs_cloud_rule.json"
SALES_TARGETED_PROTOCOL = BASE_DIR / "sales_targeted_train_protocol.json"
SALES_DENSE_PROTOCOL = BASE_DIR / "sales_dense_v2_train_protocol.json"

EXPECTED_REWARD_SHA = "325ad44feb83ec37c35babfed4bddb928cf400788e07735eb4631fc4af6962c8"
EXPECTED_HELD_OUT_SHA = "964fc352d1c83fa2738042d377c8070d6e355c51a5ddbb36c1fc9a9b99771a79"
EXPECTED_NEWANCHOR_SHA = "e74de770e8dbc95dbbf813d87fa9b38e631941ec464f790410b2c86be41c0c2b"
EXPECTED_NEWANCHOR_PROTOCOL_SHA = "3f53319f9f4158dd1e282cbefa9f84dcdf0e295c018023cae8d1557020a94783"
EXPECTED_LOCAL_VS_CLOUD_RULE_SHA = "51cf3eea9f4452908920246e825b87af1100e6090f2656bbd1af2e982dec6ed2"
EXPECTED_SALES_TARGETED_PROTOCOL_SHA = "f0a1e7545e45ea142f13ed4d674d5d1fe1280915878c0645e75b559db2554f8c"
EXPECTED_SALES_DENSE_PROTOCOL_SHA = "b24f9dd32723087bf1c7d8224f01d350694c3f527d1a123d43130b2a5bbaeb14"

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

SALES_CONCERN_RULES = (
    ("预算有限", ("预算", "价格", "空间")),
    ("城市家庭", ("预算", "空间", "家庭")),
    ("实用型", ("空间", "价格", "可靠性")),
    ("鸿蒙座舱", ("座舱", "智驾")),
    ("主动安全", ("安全", "智驾")),
    ("旅行车", ("空间", "家庭")),
    ("换电服务", ("换电", "补能", "服务")),
    ("长续航快充", ("续航", "快充", "补能")),
    ("底盘质感", ("底盘", "舒适", "驾驶感")),
    ("行政接待", ("商务", "舒适", "品牌")),
    ("华为智驾", ("智驾", "安全")),
    ("舒适配置", ("舒适", "配置")),
    ("高级底盘", ("底盘", "舒适")),
    ("空间和价格优势", ("空间", "价格")),
    ("高端商务", ("商务", "品牌", "舒适")),
    ("服务体验", ("服务", "品牌")),
    ("长途出行", ("续航", "补能", "底盘")),
    ("插混车电池安全和保修", ("能源类型", "电池", "安全", "保修")),
    ("低使用成本和能耗", ("使用成本", "能耗", "补能")),
    ("没有家充的补能价值", ("家充", "补能", "续航")),
    ("辅助驾驶安全边界", ("智驾", "安全", "边界")),
    ("预算有限但想要大空间", ("预算", "价格", "空间")),
    ("两款车型之间的犹豫", ("对比", "选择", "核验")),
    ("等待降价且不夸大权益", ("价格", "权益", "核验")),
    ("后期维保网点", ("维保", "售后", "品牌")),
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
    ("换电", ("fast_charge", "cltc_range", "monthly_sales")),
    ("家充", ("fast_charge", "cltc_range", "energy_type")),
    ("能耗", ("energy_type", "cltc_range", "fast_charge")),
    ("省油", ("energy_type", "cltc_range", "fast_charge")),
    ("使用成本", ("energy_type", "cltc_range", "fast_charge")),
    ("空间", ("wheelbase", "trunk_volume", "seats")),
    ("后备箱", ("trunk_volume", "wheelbase", "seats")),
    ("家庭", ("seats", "wheelbase", "trunk_volume")),
    ("六座", ("seats", "wheelbase", "trunk_volume")),
    ("安全", ("safety_score", "adas_level", "energy_type")),
    ("可靠性", ("safety_score", "monthly_sales", "energy_type")),
    ("智驾", ("adas_level", "safety_score", "smart_cockpit")),
    ("智能驾驶", ("adas_level", "safety_score", "smart_cockpit")),
    ("边界", ("adas_level", "safety_score", "energy_type")),
    ("座舱", ("smart_cockpit", "adas_level", "energy_type")),
    ("鸿蒙", ("smart_cockpit", "adas_level", "energy_type")),
    ("配置", ("smart_cockpit", "adas_level", "safety_score")),
    ("舒适", ("wheelbase", "seats", "smart_cockpit")),
    ("底盘", ("drive_type", "wheelbase", "safety_score")),
    ("操控", ("drive_type", "adas_level", "energy_type")),
    ("体验", ("drive_type", "smart_cockpit", "energy_type")),
    ("试驾", ("drive_type", "adas_level", "energy_type")),
    ("动力", ("drive_type", "energy_type", "cltc_range")),
    ("驾驶感", ("drive_type", "energy_type", "adas_level")),
    ("加速", ("drive_type", "energy_type", "cltc_range")),
    ("品牌", ("monthly_sales", "smart_cockpit", "energy_type")),
    ("商务", ("smart_cockpit", "monthly_sales", "wheelbase")),
    ("服务", ("monthly_sales", "safety_score", "energy_type")),
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")


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
            rows[f"{row['brand']} {row['model']}"] = row
            lines[f"{row['brand']} {row['model']}"] = line
    return rows, lines


def append_unique(items: list[str], values: tuple[str, ...]) -> None:
    for value in values:
        if value and value not in items:
            items.append(value)


def direct_tokens(query: str) -> tuple[str, ...]:
    found: list[str] = []
    for value in (
        "智驾",
        "智能驾驶",
        "座舱",
        "鸿蒙",
        "可靠性",
        "保值",
        "续航",
        "补能",
        "快充",
        "换电",
        "空间",
        "后备箱",
        "安全",
        "主动安全",
        "品牌",
        "驾驶感",
        "操控",
        "加速",
        "底盘",
        "舒适",
        "能耗",
        "省油",
        "预算",
        "价格",
        "配置",
        "服务",
        "增程",
        "插混",
        "纯电",
        "长途",
        "商务",
        "家庭",
        "六座",
    ):
        if value in query:
            append_unique(found, (value,))
    return tuple(found)


def sales_anchors(query: str) -> tuple[str, ...]:
    anchors: list[str] = []
    for phrase, values in SALES_CONCERN_RULES:
        if phrase in query:
            append_unique(anchors, values)
    append_unique(anchors, direct_tokens(query))
    return tuple(anchors) or DEFAULT_CONCERNS


def recommend_anchors(query: str) -> tuple[str, ...]:
    # This deliberately avoids sales-policy phrase expansion and keeps to
    # product-feature tokens, matching the existing recommend exam policy.
    anchors = list(direct_tokens(query))
    if "800V" in query or "800v" in query.casefold():
        append_unique(anchors, ("快充", "补能"))
    if "市区代步" in query or "城市代步" in query:
        append_unique(anchors, ("空间", "能耗"))
    if "露营" in query:
        append_unique(anchors, ("空间", "续航"))
    if "行政" in query:
        append_unique(anchors, ("商务", "舒适"))
    return tuple(anchors) or DEFAULT_CONCERNS


def fields_for_anchors(query: str, anchors: tuple[str, ...], *, limit: int) -> tuple[str, ...]:
    tokens = list(anchors)
    for token, _mapped in FIELD_RULES:
        if token in query and token not in tokens:
            tokens.append(token)
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
    for field in DEFAULT_FIELDS:
        if field not in fields:
            fields.append(field)
        if len(fields) >= limit:
            break
    return tuple(fields[:limit])


def claim(entity: str, field_name: str, row: dict[str, str], line: int, anchors: tuple[str, ...]) -> dict[str, Any]:
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


def knowledge_claims(row: dict[str, Any], catalog: dict[str, dict[str, str]], lines: dict[str, int]) -> list[dict[str, Any]]:
    entity = row["allowed_models"][0]
    anchors = recommend_anchors(row["query"])
    fields = fields_for_anchors(row["query"], anchors, limit=3)
    return [claim(entity, field, catalog[entity], lines[entity], anchors) for field in fields]


def instruction(intent: str) -> str:
    common = "只能使用上面的冻结证据；实体、字段和值必须逐字一致，不得补写其他数字。"
    if intent == "recommend":
        return f"{common} 使用推荐/首选/备选等决策词绑定真实候选车型，并给出证据理由。"
    if intent == "compare":
        return f"{common} 对两个目标车型分别给出至少一条证据命题。"
    if intent == "knowledge":
        return f"{common} 直接回答核心问题，至少给出一个与车型和query anchor绑定的命题。"
    return f"{common} 写简洁销售沟通话术，包含建议/试驾/核验/了解/跟进动作，并给出产品证据。"


def build_case(row: dict[str, Any], ordinal: int, catalog: dict[str, dict[str, str]], lines: dict[str, int]) -> dict[str, Any]:
    intent = row["intent"]
    query = row["query"]
    if intent == "sales":
        anchors = sales_anchors(query)
        fields = fields_for_anchors(query, anchors, limit=3)
        targets = row["allowed_models"][:1]
        claims = [claim(targets[0], field, catalog[targets[0]], lines[targets[0]], anchors) for field in fields]
        minimum = 1
    elif intent == "recommend":
        anchors = recommend_anchors(query)
        fields = fields_for_anchors(query, anchors, limit=3)
        targets = [model for model in row["allowed_models"] if model in catalog][:3]
        claims = [claim(entity, field, catalog[entity], lines[entity], anchors) for entity in targets for field in fields]
        minimum = 1
    elif intent == "compare":
        anchors = recommend_anchors(query)
        fields = fields_for_anchors(query, anchors, limit=3)
        targets = row["allowed_models"][:2]
        claims = [claim(entity, field, catalog[entity], lines[entity], anchors) for entity in targets for field in fields]
        minimum = len(targets)
    else:
        anchors = recommend_anchors(query)
        targets = row["allowed_models"][:1]
        claims = knowledge_claims(row, catalog, lines)
        fields = tuple(claim_row["canonical_attribute"] for claim_row in claims)
        minimum = 1
    return {
        "id": row["id"],
        "split": "held_out_40_final_eval",
        "intent": intent,
        "query": query,
        "target_entities": targets,
        "intent_response_spec": {
            "query_anchor_tokens": list(anchors),
            "query_attribute_anchors": [ATTRIBUTES[field][0] for field in fields] if intent != "knowledge" else list(fields),
            "minimum_supported_claims": minimum,
            "decision_tokens": list(DECISION_TOKENS) if intent == "recommend" else [],
            "communication_action_tokens": list(COMMUNICATION_ACTION_TOKENS) if intent == "sales" else [],
        },
        "evidence_claims": claims,
        "generation_instruction": instruction(intent),
        "known_reward_contract_risks": (
            ["report_only_data_limited"] if intent in {"compare", "knowledge"} else []
        ),
        "candidate_index": ordinal,
        "source_heldout_id": row["id"],
        "allowed_models": row["allowed_models"],
    }


def assert_hashes() -> dict[str, str]:
    expected = {
        "training/grpo/reward_fn.py": EXPECTED_REWARD_SHA,
        "data/model_training/eval/held_out.jsonl": EXPECTED_HELD_OUT_SHA,
        "data/model_training/grpo/formal_v4/restart_1/powered_dev_newanchor_128.jsonl": EXPECTED_NEWANCHOR_SHA,
        "data/model_training/grpo/formal_v4/restart_1/newanchor_eval_protocol.json": EXPECTED_NEWANCHOR_PROTOCOL_SHA,
        "data/model_training/grpo/formal_v4/restart_1/local_vs_cloud_rule.json": EXPECTED_LOCAL_VS_CLOUD_RULE_SHA,
        "data/model_training/grpo/formal_v4/restart_1/sales_targeted_train_protocol.json": EXPECTED_SALES_TARGETED_PROTOCOL_SHA,
        "data/model_training/grpo/formal_v4/restart_1/sales_dense_v2_train_protocol.json": EXPECTED_SALES_DENSE_PROTOCOL_SHA,
    }
    observed = {}
    for relative, digest in expected.items():
        actual = sha256_file(ROOT / relative)
        if actual != digest:
            raise RuntimeError(f"SHA mismatch: {relative}")
        observed[relative] = actual
    return observed


def main() -> int:
    for path in (
        OUT_DATASET,
        OUT_PROTOCOL,
        Path(str(OUT_DATASET) + ".sha256"),
        Path(str(OUT_PROTOCOL) + ".sha256"),
    ):
        if path.exists():
            raise FileExistsError(f"refusing overwrite: {path}")
    observed = assert_hashes()
    rows = load_jsonl(HELD_OUT)
    if len(rows) != 40 or Counter(row["intent"] for row in rows) != {
        "recommend": 10,
        "compare": 10,
        "knowledge": 10,
        "sales": 10,
    }:
        raise RuntimeError("held-out-40 count drift")
    catalog, lines = vehicle_rows()
    cases = [build_case(row, i, catalog, lines) for i, row in enumerate(rows, start=1)]
    write_jsonl_exclusive(OUT_DATASET, cases)
    write_sha_sidecar(OUT_DATASET)
    protocol = {
        "status": "frozen_before_held_out_40_final_evaluation",
        "purpose": "single held-out-40 terminal evaluation; not for checkpoint selection or tuning",
        "red_lines": {
            "held_out_40_accessed_before_protocol": False,
            "reward_fn_sha256_must_equal": EXPECTED_REWARD_SHA,
            "frozen_inputs_read_only": True,
            "no_generated_answer_text_persistence": True,
            "cloud_must_recompute_on_held_out_40": True,
        },
        "objects": [
            {
                "label": "sales_dense_v2_checkpoint_150",
                "type": "local_adapter",
                "path": "checkpoints/grpo/formal_v4/restart_1/sales_dense_v2/checkpoint-150",
                "role": "local representative",
            },
            {
                "label": "checkpoint-300",
                "type": "local_adapter",
                "path": "checkpoints/grpo/formal_v4/restart_1/checkpoint-300",
                "role": "original GRPO best anchor",
            },
            {
                "label": "cloud_seedpro_ark_ep_masked",
                "type": "cloud_openai_compatible",
                "role": "cloud endpoint recomputed on held-out-40",
            },
        ],
        "dataset": {
            "path": str(OUT_DATASET.relative_to(ROOT)),
            "sha256": sha256_file(OUT_DATASET),
            "source_path": "data/model_training/eval/held_out.jsonl",
            "source_sha256": EXPECTED_HELD_OUT_SHA,
            "records": len(cases),
            "counts": dict(sorted(Counter(case["intent"] for case in cases).items())),
            "construction_policy": {
                "sales": "fixed sales parser with strong query-bound anchors; 3 evidence claims; minimum_supported_claims=1",
                "recommend": "existing product-feature recommend binding logic, not sales-policy parser",
                "compare_knowledge": "report-only per-prompt trace retained; excluded from composite",
            },
        },
        "sampling": {
            "temperature": 0.8,
            "top_p": 0.95,
            "max_completion_length": 512,
            "max_total_sequence_length": 3072,
            "num_generations": 8,
            "seed": 20260725,
            "cloud_g8_realization": "8 independent n=1 chat completion calls per prompt, global concurrency 16",
        },
        "metrics": {
            "core": "0.6*factual_precision + 0.4*required_coverage",
            "composite": "0.5*recommend_mean_core + 0.5*sales_mean_core",
            "required_report": [
                "recommend",
                "sales",
                "composite",
                "local ckpt-150 minus cloud deltas",
                "local ckpt-150 minus checkpoint-300 deltas",
            ],
            "no_pass_fail_threshold": True,
            "not_a_selection_run": True,
        },
        "observed_hashes": observed,
    }
    write_json_exclusive(OUT_PROTOCOL, protocol)
    write_sha_sidecar(OUT_PROTOCOL)
    print(
        json.dumps(
            {
                "status": "frozen",
                "dataset_sha256": sha256_file(OUT_DATASET),
                "protocol_sha256": sha256_file(OUT_PROTOCOL),
                "counts": protocol["dataset"]["counts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
