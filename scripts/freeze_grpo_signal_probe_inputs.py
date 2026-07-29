#!/usr/bin/env python3
"""Freeze deterministic evidence/spec inputs for the GRPO signal probe."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GRPO_DIR = ROOT / "data" / "model_training" / "grpo"
TRAIN_PATH = GRPO_DIR / "reward_train_16.jsonl"
DEV_PATH = GRPO_DIR / "reward_dev_4.jsonl"
SOURCE_PATH = ROOT / "data" / "vehicles" / "vehicle_database.csv"
REWARD_PATH = ROOT / "training" / "grpo" / "reward_fn.py"
CONFIG_PATH = ROOT / "training" / "grpo" / "grpo_config.yaml"
AUTHORIZATION_PATH = GRPO_DIR / "grpo_dev_authorization_manifest.json"
OUTPUT_PATH = GRPO_DIR / "grpo_signal_probe_input_manifest.json"
OUTPUT_SHA_PATH = OUTPUT_PATH.with_suffix(".sha256")

EXPECTED_SHA256 = {
    TRAIN_PATH: "0390c0ee32156c02c84b08d0bc96191b0a7040a57fcaab969112f998b4539cc7",
    DEV_PATH: "bb90df011ff62cc10daf44f6eb17bbca9232d356c2695b3e2af43dcc348ae790",
    SOURCE_PATH: "f2b2b8070571e9a02513ec942b44f79bd7a1c87b083b94c434a2e2a3be6af5f7",
    REWARD_PATH: "325ad44feb83ec37c35babfed4bddb928cf400788e07735eb4631fc4af6962c8",
    CONFIG_PATH: "44d85fa22a15b0e7e648e5f8b1cceca37b50fa8773b8ccc86ee90764d00447cb",
    AUTHORIZATION_PATH: "30631cad7a8733739eafef6d285e4e0b3f007e38c1e898c064ebbf0432315bff",
}

ATTRIBUTE_SPECS = {
    "energy_type": {
        "canonical_attribute": "vehicle.energy_type",
        "aliases": ["能源类型", "能源"],
        "column": "energy_type",
        "suffix": "",
    },
    "cltc_range": {
        "canonical_attribute": "specs.cltc_range",
        "aliases": ["CLTC续航", "续航"],
        "column": "cltc_range",
        "suffix": "km",
    },
    "fast_charge": {
        "canonical_attribute": "specs.fast_charge",
        "aliases": ["快充时间", "快充", "补能"],
        "column": "fast_charge_minutes",
        "suffix": "分钟",
    },
    "seats": {
        "canonical_attribute": "specs.seats",
        "aliases": ["座位数", "座位"],
        "column": "seats",
        "suffix": "座",
    },
    "adas_level": {
        "canonical_attribute": "specs.adas_level",
        "aliases": ["辅助驾驶等级", "智驾等级", "智驾"],
        "column": "adas_level",
        "suffix": "",
    },
    "smart_cockpit": {
        "canonical_attribute": "specs.smart_cockpit",
        "aliases": ["智能座舱", "座舱"],
        "column": "smart_cockpit",
        "suffix": "",
    },
    "wheelbase": {
        "canonical_attribute": "specs.wheelbase",
        "aliases": ["轴距", "空间"],
        "column": "wheelbase",
        "suffix": "mm",
    },
    "trunk_volume": {
        "canonical_attribute": "specs.trunk_volume",
        "aliases": ["后备箱容积", "后备箱", "空间"],
        "column": "trunk_volume",
        "suffix": "L",
    },
    "safety_score": {
        "canonical_attribute": "specs.safety_score",
        "aliases": ["安全评分", "安全"],
        "column": "safety_score",
        "suffix": "分",
    },
}

CASE_CONFIG = {
    "reward-recommend-001": (("seats", "trunk_volume", "fast_charge"), ("无家充", "家庭", "空间", "补能")),
    "reward-recommend-002": (("energy_type", "wheelbase", "trunk_volume"), ("油耗", "空间", "通勤")),
    "reward-recommend-003": (("adas_level", "fast_charge"), ("智能驾驶", "快充", "长途")),
    "reward-recommend-004": (("wheelbase", "smart_cockpit"), ("后排", "舒适", "科技")),
    "reward-recommend-005": (("seats", "cltc_range", "fast_charge"), ("六座", "长途", "家庭")),
    "reward-compare-001": (("fast_charge", "adas_level"), ("性能", "补能")),
    "reward-compare-002": (("cltc_range", "trunk_volume"), ("家庭", "长途", "自驾")),
    "reward-compare-003": (("fast_charge", "cltc_range"), ("换电", "能耗", "补能")),
    "reward-compare-004": (("wheelbase", "trunk_volume"), ("旅行车", "空间", "取舍")),
    "reward-compare-005": (("wheelbase", "trunk_volume"), ("商务", "接待", "空间")),
    "reward-knowledge-001": (("energy_type", "fast_charge"), ("无家充", "补能", "油耗")),
    "reward-knowledge-002": (("fast_charge", "cltc_range"), ("800V", "快充", "长途")),
    "reward-knowledge-003": (("fast_charge", "cltc_range"), ("换电", "场景", "补能")),
    "reward-knowledge-004": (("energy_type", "cltc_range"), ("增程", "高速", "城市")),
    "reward-knowledge-005": (("safety_score", "wheelbase"), ("底盘", "安全")),
    "reward-sales-001": (("cltc_range", "safety_score"), ("能耗", "品牌", "家庭")),
    "reward-sales-002": (("cltc_range", "trunk_volume"), ("露营", "家庭", "长途")),
    "reward-sales-003": (("adas_level", "cltc_range"), ("科技", "性能", "年轻")),
    "reward-sales-004": (("seats", "safety_score"), ("六座", "家庭", "空间", "安全")),
    "reward-sales-005": (("energy_type", "trunk_volume"), ("设计", "纯电", "家庭")),
}

DECISION_TOKENS = ("推荐", "首选", "建议优先", "备选")
COMMUNICATION_ACTION_TOKENS = ("建议", "试驾", "核验", "了解", "跟进")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _catalog() -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    rows: dict[str, dict[str, str]] = {}
    line_numbers: dict[str, int] = {}
    with SOURCE_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            name = f"{row['brand']} {row['model']}"
            rows[name] = row
            line_numbers[name] = line_number
    return rows, line_numbers


def _claim(
    *,
    entity: str,
    field_name: str,
    row: dict[str, str],
    line_number: int,
    anchors: tuple[str, ...],
) -> dict[str, Any]:
    field = ATTRIBUTE_SPECS[field_name]
    value = f"{row[field['column']]}{field['suffix']}"
    model_alias = entity.split(" ", 1)[-1]
    return {
        "canonical_entity": entity,
        "canonical_attribute": field["canonical_attribute"],
        "canonical_value": value,
        "source_tool": "frozen_local_catalog",
        "source_locator": (
            f"data/vehicles/vehicle_database.csv:{line_number}."
            f"{field['column']}"
        ),
        "entity_aliases": [
            entity.replace(" ", ""),
            model_alias,
        ],
        "attribute_aliases": field["aliases"],
        "anchor_tokens": list(anchors),
    }


def _instruction(intent: str, target_count: int) -> str:
    common = (
        "只能使用上面的冻结证据；车型、字段和值必须逐字一致，不得补写其他数字。"
    )
    if intent == "recommend":
        return (
            f"{common} 使用“推荐/首选/备选”等决策词绑定真实车型，"
            "并给出至少一条证据理由。"
        )
    if intent == "compare":
        return (
            f"{common} 对全部 {target_count} 个目标车型逐一作为主体对比，"
            "每台至少给出一条证据命题。"
        )
    if intent == "knowledge":
        return f"{common} 直接回答核心问题，至少给出两个不同的证据命题。"
    return (
        f"{common} 写简洁销售沟通话术，回应关注点，至少给出一条产品证据，"
        "并包含建议、试驾、核验或跟进动作。"
    )


def build_manifest() -> dict[str, Any]:
    for path, expected in EXPECTED_SHA256.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen SHA drift: {path.relative_to(ROOT)}")
    train = load_jsonl(TRAIN_PATH)
    dev = load_jsonl(DEV_PATH)
    rows = [(row, "train") for row in train] + [
        (row, "dev") for row in dev
    ]
    if len(rows) != 20 or set(CASE_CONFIG) != {
        row["id"] for row, _ in rows
    }:
        raise ValueError("probe case configuration does not cover frozen 20")
    catalog, line_numbers = _catalog()

    cases: list[dict[str, Any]] = []
    for row, split in rows:
        fields, anchors = CASE_CONFIG[row["id"]]
        missing = [
            entity for entity in row["allowed_models"] if entity not in catalog
        ]
        if missing:
            raise ValueError(f"{row['id']}: catalog entity missing: {missing}")
        claims = [
            _claim(
                entity=entity,
                field_name=field_name,
                row=catalog[entity],
                line_number=line_numbers[entity],
                anchors=anchors,
            )
            for entity in row["allowed_models"]
            for field_name in fields
        ]
        intent = row["intent"]
        minimum = (
            2
            if intent == "knowledge"
            else len(row["allowed_models"])
            if intent == "compare"
            else 1
        )
        cases.append(
            {
                "id": row["id"],
                "split": split,
                "intent": intent,
                "query": row["query"],
                "target_entities": row["allowed_models"],
                "intent_response_spec": {
                    "query_anchor_tokens": list(anchors),
                    "query_attribute_anchors": [
                        ATTRIBUTE_SPECS[name]["canonical_attribute"]
                        for name in fields
                    ],
                    "minimum_supported_claims": minimum,
                    "decision_tokens": (
                        list(DECISION_TOKENS)
                        if intent == "recommend"
                        else []
                    ),
                    "communication_action_tokens": (
                        list(COMMUNICATION_ACTION_TOKENS)
                        if intent == "sales"
                        else []
                    ),
                },
                "evidence_claims": claims,
                "generation_instruction": _instruction(
                    intent,
                    len(row["allowed_models"]),
                ),
                "known_reward_contract_risks": (
                    [
                        "current compare gate requires exactly two targets; "
                        "this three-target case is expected to fail closed"
                    ]
                    if intent == "compare"
                    and len(row["allowed_models"]) != 2
                    else []
                ),
            }
        )

    return {
        "format_version": 1,
        "status": "frozen_before_any_signal_probe_rollout",
        "purpose": (
            "Read-only 20x8 sampling probe to test whether deterministic "
            "grounding reward has non-zero within-group variance."
        ),
        "authorization": {
            "type": "explicit_one_time_user_authorization",
            "dev_use": (
                "Read-only sampling and reward diagnostics for the fire/no-fire "
                "decision only. No dev metric may change any hyperparameter, "
                "reward rule, threshold, evidence, prompt, or data selection."
            ),
            "bound_prior_dev_authorization_sha256": EXPECTED_SHA256[
                AUTHORIZATION_PATH
            ],
        },
        "sampling_config": {
            "policy": "Qwen2.5-7B-Instruct plus frozen SFT best adapter",
            "num_generations": 8,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_prompt_length": 2560,
            "max_completion_length": 512,
            "max_total_sequence_length": 3072,
            "beta_recorded_not_used_without_kl": 0.01,
            "fp16": True,
            "bf16": False,
            "seed": 20260725,
            "gpu_processes": 1,
        },
        "fire_gate": {
            "nonzero_variance_group_ratio_lower_bound": 0.30,
            "nonzero_variance_group_ratio_clear_pass": 0.40,
            "saturation_outcome": (
                "do_not_ignite_choose_reward_hardening_or_stop_at_sft"
            ),
        },
        "execution_contract": {
            "rollout": True,
            "programmatic_reward": True,
            "loss": False,
            "backward": False,
            "optimizer_step": False,
            "checkpoint": False,
            "refuse_report_overwrite": True,
        },
        "frozen_sources": {
            str(path.relative_to(ROOT)): sha
            for path, sha in EXPECTED_SHA256.items()
        },
        "counts": {
            "cases": len(cases),
            "train": sum(case["split"] == "train" for case in cases),
            "dev": sum(case["split"] == "dev" for case in cases),
            "per_intent": dict(
                sorted(Counter(case["intent"] for case in cases).items())
            ),
            "expected_completions": len(cases) * 8,
        },
        "cases": cases,
    }


def main() -> None:
    if OUTPUT_PATH.exists() or OUTPUT_SHA_PATH.exists():
        raise FileExistsError("refusing to overwrite frozen probe inputs")
    manifest = build_manifest()
    OUTPUT_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(OUTPUT_PATH)
    OUTPUT_SHA_PATH.write_text(
        f"{digest}  {OUTPUT_PATH.name}\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "path": str(OUTPUT_PATH.relative_to(ROOT)),
                "sha256": digest,
                "counts": manifest["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
