"""Generate fail-closed 500-record teacher decision datasets per intent."""

import argparse
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.database import list_vehicles
from app.services import agent_graph
from data_synth.generate_20perintent_sft import (
    INTENTS,
    _append_failure,
    _environment,
    _generate_record_with_backoff,
    _load_existing,
    _load_failures,
    _tool_names,
    _write_dataset,
)
from data_synth.generate_sft_data import (
    audit_answer_grounding,
    audit_teacher_decision_record,
)
from data_synth.validate_tool_data import validate_record


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "model_training"
BASELINE_REWRITE_RATE = {
    "recommend": 5.0,
    "compare": 5.0,
    "customer_service": 5.0,
    "deep_search": 15.0,
    "sales": 10.0,
}
MAX_CANDIDATES = 650
MINIMUM_GATE_SAMPLES = 50


def _vehicle_name(vehicle: dict[str, Any]) -> str:
    return f"{vehicle['brand']}{vehicle['model']}"


def _dedupe(items: list[str], count: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) == count:
            return result
    raise ValueError(f"not enough unique query candidates: required {count}, got {len(result)}")


def _recommend_queries(count: int) -> list[str]:
    budgets = [12, 15, 18, 20, 22, 25, 28, 30, 35, 40, 45, 50]
    families = ["单人通勤", "情侣出行", "三口之家", "二胎家庭", "老人接送孩子", "家用兼商务接待"]
    types = ["SUV", "轿车", "MPV"]
    energies = ["纯电", "插混", "增程", "新能源"]
    concerns = [
        "空间和能耗", "安全和续航", "快充和补能", "舒适和后备箱",
        "智驾和座舱", "可靠性和保值", "驾驶感和加速", "品牌和低调感",
    ]
    usages = ["城市通勤为主", "周末带家人出游", "偶尔跨城", "经常高速出行", "没有家充", "有固定家充"]
    items = []
    for values in itertools.product(budgets, families, types, energies, concerns, usages):
        budget, family, vehicle_type, energy, concern, usage = values
        items.append(
            f"预算{budget}万，{family}，{usage}，想买{energy}{vehicle_type}，关注{concern}，请推荐。"
        )
        if len(items) >= count * 2:
            break
    return _dedupe(items, count)


def _deep_search_queries(count: int) -> list[str]:
    budgets = [15, 18, 20, 22, 25, 28, 30, 35, 40, 45, 50]
    types = ["SUV", "轿车", "MPV"]
    energies = ["纯电", "插混", "增程", "新能源"]
    focus = [
        "空间、能耗和后期使用", "智驾、快充和家庭空间",
        "补能条件和长途出行", "安全、保修表达和购车风险",
        "品牌、保值和售后网点", "舒适、座舱和商务接待",
        "续航、后备箱和座位数", "驾驶感、动力和日常通勤",
    ]
    scenarios = ["三口之家", "二胎家庭", "单人城市通勤", "家用兼业务接待", "没有家充", "经常跨城"]
    items = []
    for budget, vehicle_type, energy, aspect, scenario in itertools.product(
        budgets, types, energies, focus, scenarios
    ):
        items.append(
            f"请深度调研预算{budget}万的{energy}{vehicle_type}，{scenario}场景，重点看{aspect}。"
        )
        if len(items) >= count * 2:
            break
    return _dedupe(items, count)


def _customer_service_queries(count: int) -> list[str]:
    topics = [
        "辅助驾驶是不是自动驾驶，高速能不能放手",
        "新能源车保修政策怎么查，能不能承诺终身保修",
        "没有家充时日常补能应该注意什么",
        "冬天续航下降是不是质量问题",
        "电池安全应该如何向用户解释",
        "客户询问价格和权益时能否直接报最低价",
        "增程车和插混车在使用上有什么区别",
        "客户担心充电慢时如何解释快充",
        "智能座舱卡顿或升级问题如何回答",
        "二手保值率应该如何克制表达",
        "试驾车与交付车配置不一致怎么办",
        "官方参数与实测不一致如何解释",
        "施工路段使用辅助驾驶如何提醒",
        "电池衰减问题应该如何回答",
        "置换补贴应该如何引导核验",
        "门店覆盖少会不会影响维保",
        "工具没有官方油耗电耗时怎么回答",
        "客户追问交付周期能不能承诺日期",
        "事故后电池质保是否还有效",
        "不同城市政策差异应该怎么查询",
    ]
    styles = [
        "请给一段克制客服回复。", "请说明应如何核验官方信息。",
        "请给出风险边界提示。", "请用客服口吻回答，不要推荐车型。",
        "请给出处理步骤和注意事项。",
    ]
    cities = ["北京", "上海", "广州", "深圳", "成都", "武汉", "杭州"]
    customer_contexts = [
        "首次购车用户", "家庭用户", "通勤用户", "长途用户",
        "门店咨询用户", "线上咨询用户", "已提车用户",
    ]
    return _dedupe(
        [
            f"{topic}，{style.rstrip('。')}。用户位于{city}，属于{customer_context}。"
            for topic, style, city, customer_context in itertools.product(
                topics,
                styles,
                cities,
                customer_contexts,
            )
        ],
        count,
    )


def _sales_queries(count: int) -> list[str]:
    concerns = [
        "插混车电池安全和保修", "低使用成本和能耗", "没有家充的补能价值",
        "国产新能源品牌高端感", "辅助驾驶安全边界", "预算有限但想要大空间",
        "两款车型之间的犹豫", "等待降价且不夸大权益", "后期维保网点",
        "工具未返回的油耗数值", "增程技术路线", "智能座舱学习成本",
        "试驾后觉得动力一般", "二胎家庭后备箱", "首次购买新能源",
        "最低价承诺", "品牌保值率顾虑", "纯电和插混场景分析",
        "交付周期敏感", "社交形象与预算平衡",
    ]
    approaches = [
        "销售怎么回应才克制？", "请给出开场、异议处理和跟进建议。",
        "请生成不承诺政策的沟通话术。", "应该如何引导用户核验官方信息？",
        "请给出以实际体验为主的沟通方案。",
    ]
    customer_profiles = [
        "首次购车", "已试驾", "带孩子出行", "经常跨城",
        "城市通勤", "关注可靠性", "预算敏感",
    ]
    touchpoints = [
        "线上首轮沟通", "到店接待", "试驾后跟进", "交付前确认",
        "售后回访", "电话回访", "微信沟通",
    ]
    return _dedupe(
        [
            f"客户关注{concern}，{approach.rstrip('。')}。当前为{customer_profile}，处于{touchpoint}。"
            for concern, approach, customer_profile, touchpoint in itertools.product(
                concerns,
                approaches,
                customer_profiles,
                touchpoints,
            )
        ],
        count,
    )


def _compare_queries(count: int) -> list[str]:
    names = [_vehicle_name(vehicle) for vehicle in list_vehicles()]
    contexts = [
        "主要城市通勤，关注空间和补能。",
        "家庭出行场景，关注舒适和后备箱。",
        "偶尔跨城，关注续航和补能。",
        "商务兼家用，关注品牌和座舱。",
        "关注安全、辅助驾驶和使用边界。",
        "预算有限，关注综合取舍。",
        "有家充条件，关注纯电使用体验。",
        "没有家充，关注能源路线适配。",
    ]
    pairs = list(itertools.combinations(names, 2))
    items = [
        f"{left}和{right}怎么选？{contexts[index % len(contexts)]}"
        for index, (left, right) in enumerate(pairs)
    ]
    return _dedupe(items, count)


def _compare_expected_model_names(query: str) -> list[str]:
    compared_models, marker, _ = query.partition("怎么选？")
    if not marker:
        return []
    left, separator, right = compared_models.partition("和")
    if not separator:
        return []
    return [left.strip(), right.strip()]


def _compact_vehicle_name(value: str) -> str:
    return "".join(
        character.casefold()
        for character in value
        if character.isalnum()
    )


def _compare_pair_recalled(
    record: dict[str, Any],
    lookup: dict[str, Any],
) -> bool:
    expected_names = record.get("expected_named_models")
    if not isinstance(expected_names, list) or len(expected_names) != 2:
        expected_names = lookup.get("requested_model_names", [])
    resolved_names = lookup.get("resolved_model_names", [])
    if (
        lookup.get("named_vehicle_missing")
        or not isinstance(expected_names, list)
        or not isinstance(resolved_names, list)
        or len(expected_names) != 2
        or len(resolved_names) != 2
        or not all(
            isinstance(name, str) and name.strip()
            for name in [*expected_names, *resolved_names]
        )
    ):
        return False
    return {
        _compact_vehicle_name(name)
        for name in expected_names
    } == {
        _compact_vehicle_name(name)
        for name in resolved_names
    }


def build_queries(intent: str, count: int) -> list[str]:
    builders = {
        "recommend": _recommend_queries,
        "compare": _compare_queries,
        "customer_service": _customer_service_queries,
        "deep_search": _deep_search_queries,
        "sales": _sales_queries,
    }
    return builders[intent](count)


def _extract_compare_lookup(record: dict[str, Any]) -> dict[str, Any] | None:
    for message in record.get("messages", []):
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content") or "")
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "named_vehicle_lookup" in payload:
            return payload["named_vehicle_lookup"]
    return None


def _paths(intent: str, output_label: str = "") -> dict[str, Path]:
    if output_label and not output_label.replace("_", "").isalnum():
        raise ValueError("output_label must contain only letters, digits, and underscores")
    prefix = f"teacher_decision_500perintent_{intent}"
    if output_label:
        prefix = f"{prefix}_{output_label}"
    return {
        "dataset": OUT_DIR / f"{prefix}_sft.jsonl",
        "failures": OUT_DIR / f"{prefix}_failures.jsonl",
        "report": OUT_DIR / f"{prefix}_audit_report.md",
        "report_json": OUT_DIR / f"{prefix}_audit_report.json",
        "manifest": OUT_DIR / f"{prefix}_manifest.json",
    }


def _write_report(
    *,
    intent: str,
    target: int,
    records: dict[str, dict[str, Any]],
    failures: list[dict[str, Any]],
    gate_reason: str,
    paths: dict[str, Path],
) -> dict[str, Any]:
    accepted_rows = list(records.values())
    attempted = len(accepted_rows) + len(failures)
    truncated = sum(item.get("error_type") == "length" for item in failures)
    rewrite = sum(bool(item.get("bounded_rewrite_triggered")) for item in accepted_rows)
    avg_rounds = (
        sum(int(item.get("tool_call_rounds", 0)) for item in accepted_rows)
        / len(accepted_rows)
        if accepted_rows else 0.0
    )
    summary: dict[str, Any] = {
        "intent": intent,
        "target": target,
        "attempted": attempted,
        "accepted": len(accepted_rows),
        "success_rate": round(len(accepted_rows) / attempted * 100, 1) if attempted else 0.0,
        "truncated": truncated,
        "truncation_rate": round(truncated / attempted * 100, 1) if attempted else 0.0,
        "rewrite_triggered": rewrite,
        "rewrite_rate": round(rewrite / len(accepted_rows) * 100, 1) if accepted_rows else 0.0,
        "avg_tool_call_rounds": round(avg_rounds, 2),
        "gate_reason": gate_reason,
    }
    if intent == "compare":
        lookup_mismatch_failures = [
            failure for failure in failures
            if (
                "compare named lookup did not recall the generated catalog pair"
                in failure.get("decision_audit", [])
            )
        ]
        lookup_rows = [
            (row, lookup) for row in accepted_rows
            if (lookup := row.get("named_vehicle_lookup")) is not None
        ]
        pair_recalled = [
            (row, lookup)
            for row, lookup in lookup_rows
            if _compare_pair_recalled(row, lookup)
        ]
        recall_denominator = len(lookup_rows) + len(lookup_mismatch_failures)
        summary["compare_named_lookup_rows"] = len(lookup_rows)
        summary["compare_in_catalog_rows"] = recall_denominator
        summary["compare_both_recalled"] = len(pair_recalled)
        summary["compare_both_recall_rate"] = (
            round(len(pair_recalled) / recall_denominator * 100, 1)
            if recall_denominator else 0.0
        )
        summary["compare_named_vehicle_missing"] = sum(
            bool(lookup.get("named_vehicle_missing"))
            for _, lookup in lookup_rows
        )
        summary["compare_lookup_mismatches"] = len(lookup_mismatch_failures)

    paths["report_json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# 500/intent 教师决策统计：{intent}",
        "",
        f"- 目标成功样本：{target}",
        f"- 尝试数：{summary['attempted']}",
        f"- 成功写入：{summary['accepted']}",
        f"- 写入成功率：{summary['accepted']}/{summary['attempted']} = {summary['success_rate']:.1f}%",
        f"- 截断率：{summary['truncated']}/{summary['attempted']} = {summary['truncation_rate']:.1f}%",
        f"- rewrite 率：{summary['rewrite_triggered']}/{summary['accepted']} = {summary['rewrite_rate']:.1f}%",
        f"- 平均 tool_call 轮次：{summary['avg_tool_call_rounds']:.2f}",
        f"- 停止原因：{gate_reason or '未触发'}",
    ]
    if intent == "compare":
        lines.extend(
            [
                "",
                "## 点名车型召回",
                "",
                f"- 在库点名对：{summary['compare_in_catalog_rows']}",
                f"- 实际同时召回：{summary['compare_both_recalled']}",
                f"- 同时召回率：{summary['compare_both_recall_rate']:.1f}%",
                f"- 库外点名车样本：{summary['compare_named_vehicle_missing']}",
                f"- 点名错配失败：{summary['compare_lookup_mismatches']}",
            ]
        )
    if failures:
        lines.extend(["", "## Fail-Closed 失败摘要", ""])
        for item in failures[-30:]:
            reason = item.get("error") or item.get("decision_audit") or item.get("grounding_audit")
            lines.append(
                f"- {item.get('id')}：{item.get('error_type')}，reason={reason}"
            )
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _gate_reason(intent: str, summary: dict[str, Any]) -> str:
    if summary["attempted"] and summary["truncation_rate"] >= 5.0:
        return f"截断率{summary['truncation_rate']:.1f}%>=5%"
    if intent == "compare" and summary.get("compare_lookup_mismatches", 0):
        return (
            "compare出现点名车型错配："
            f"{summary['compare_lookup_mismatches']}条"
        )
    if summary["accepted"] >= MINIMUM_GATE_SAMPLES:
        threshold = max(15.0, BASELINE_REWRITE_RATE[intent] + 10.0)
        if summary["rewrite_rate"] > threshold:
            return (
                f"rewrite率{summary['rewrite_rate']:.1f}%"
                f">停止线{threshold:.1f}%"
            )
        if intent == "compare" and summary["compare_both_recall_rate"] < 100.0:
            return (
                "compare在库点名车型同时召回率"
                f"{summary['compare_both_recall_rate']:.1f}%<100%"
            )
    return ""


def generate_intent(
    *,
    intent: str,
    target: int,
    concurrency: int,
    max_candidates: int,
    resume: bool,
    output_label: str = "",
) -> dict[str, Any]:
    if intent not in INTENTS:
        raise ValueError(f"unsupported intent: {intent}")
    if concurrency < 1 or concurrency > 64:
        raise ValueError("concurrency must be between 1 and 64")
    if max_candidates < target:
        raise ValueError("max_candidates must be >= target")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = _paths(intent, output_label)
    queries = build_queries(intent, max_candidates)
    if resume:
        records = _load_existing(paths["dataset"])
        failures = _load_failures(paths["failures"])
    else:
        records = {}
        failures = []
        for path in paths.values():
            if path.exists():
                path.unlink()
    paths["manifest"].write_text(
        json.dumps(
            {
                "intent": intent,
                "target": target,
                "max_candidates": max_candidates,
                "output_label": output_label,
                "queries": queries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    failed_ids = {item.get("id") for item in failures}
    values = _environment(ROOT / ".env")
    client = OpenAI(
        base_url=values["ARK_BASE_URL"],
        api_key=values["ARK_API_KEY"],
        timeout=240,
        max_retries=0,
    )
    gate_reason = ""

    def checkpoint() -> dict[str, Any]:
        summary = _write_report(
            intent=intent,
            target=target,
            records=records,
            failures=failures,
            gate_reason=gate_reason,
            paths=paths,
        )
        return summary

    def worker(index: int, query: str, record_id: str) -> tuple[str, str, dict[str, Any]]:
        started = time.time()
        print(f"START {record_id} intent={intent}", flush=True)
        try:
            record = _generate_record_with_backoff(
                record_id=record_id,
                query=query,
                intent=intent,
                client=client,
                model=values["ARK_CHAT_MODEL"],
                max_steps=8,
                max_grounding_retries=1,
                max_tokens=2600,
                system_prompt=agent_graph._get_prompt_for_intent(intent),
                include_audit_metadata=True,
            )
            validate_errors = validate_record(record, held_out_ids=set())
            decision_errors = audit_teacher_decision_record(intent, record)
            grounding_errors = audit_answer_grounding(record)
            if validate_errors or decision_errors or grounding_errors:
                return (
                    "failure",
                    record_id,
                    {
                        "id": record_id,
                        "intent": intent,
                        "query": query,
                        "query_index": index,
                        "error_type": "audit",
                        "validate_record": validate_errors,
                        "decision_audit": decision_errors,
                        "grounding_audit": grounding_errors,
                        "elapsed_sec": round(time.time() - started, 2),
                    },
                )
            if intent == "compare":
                lookup = _extract_compare_lookup(record)
                if lookup is None:
                    return (
                        "failure",
                        record_id,
                        {
                            "id": record_id,
                            "intent": intent,
                            "query": query,
                            "query_index": index,
                            "error_type": "audit",
                            "validate_record": [],
                            "decision_audit": ["compare named_vehicle_lookup metadata is missing"],
                            "grounding_audit": [],
                            "elapsed_sec": round(time.time() - started, 2),
                        },
                    )
                expected_names = _compare_expected_model_names(query)
                if not _compare_pair_recalled(
                    {"expected_named_models": expected_names},
                    lookup,
                ):
                    return (
                        "failure",
                        record_id,
                        {
                            "id": record_id,
                            "intent": intent,
                            "query": query,
                            "query_index": index,
                            "error_type": "audit",
                            "validate_record": [],
                            "decision_audit": [
                                "compare named lookup did not recall the generated "
                                "catalog pair"
                            ],
                            "grounding_audit": [],
                            "elapsed_sec": round(time.time() - started, 2),
                        },
                    )
                record["named_vehicle_lookup"] = lookup
                record["expected_named_models"] = expected_names
            record["validate_record"] = validate_errors
            record["decision_audit"] = decision_errors
            record["grounding_audit"] = grounding_errors
            record["tool_names"] = _tool_names(record)
            record["query_index"] = index
            record["elapsed_sec"] = round(time.time() - started, 2)
            return "success", record_id, record
        except Exception as exc:
            message = str(exc)
            return (
                "failure",
                record_id,
                {
                    "id": record_id,
                    "intent": intent,
                    "query": query,
                    "query_index": index,
                    "error_type": "length" if "truncated" in message else type(exc).__name__,
                    "error": message,
                    "elapsed_sec": round(time.time() - started, 2),
                },
            )

    candidates = [
        (index, query, f"500pi-{intent}-{index:04d}")
        for index, query in enumerate(queries, 1)
        if f"500pi-{intent}-{index:04d}" not in records
        and f"500pi-{intent}-{index:04d}" not in failed_ids
    ]
    cursor = 0
    while len(records) < target and cursor < len(candidates) and not gate_reason:
        remaining = target - len(records)
        batch = candidates[cursor:cursor + min(concurrency, remaining)]
        cursor += len(batch)
        if not batch:
            break
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [
                executor.submit(worker, index, query, record_id)
                for index, query, record_id in batch
            ]
            for future in as_completed(futures):
                status, record_id, payload = future.result()
                if status == "success":
                    records[record_id] = payload
                    _write_dataset(records, paths["dataset"])
                    print(
                        "OK {record_id} elapsed={elapsed:.1f}s rounds={rounds} tools={tools}".format(
                            record_id=record_id,
                            elapsed=payload["elapsed_sec"],
                            rounds=payload.get("tool_call_rounds", 0),
                            tools=",".join(payload.get("tool_names", [])),
                        ),
                        flush=True,
                    )
                else:
                    failures.append(payload)
                    _append_failure(payload, paths["failures"])
                    print(
                        f"FAIL {record_id} {payload['error_type']} "
                        f"elapsed={payload['elapsed_sec']:.1f}s "
                        f"error={str(payload.get('error') or payload.get('decision_audit') or payload.get('grounding_audit'))[:160]}",
                        flush=True,
                    )
                summary = checkpoint()
                gate_reason = _gate_reason(intent, summary)
        checkpoint()

    if not gate_reason and len(records) < target:
        gate_reason = (
            f"候选query耗尽：accepted={len(records)}<target={target}"
        )
    summary = checkpoint()
    return {
        "summary": summary,
        "dataset": str(paths["dataset"]),
        "report": str(paths["report"]),
        "manifest": str(paths["manifest"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--intent", choices=INTENTS)
    group.add_argument("--all-intents", action="store_true")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output-label", default="")
    args = parser.parse_args()

    intents = INTENTS if args.all_intents else [args.intent]
    results = {}
    for intent in intents:
        results[intent] = generate_intent(
            intent=intent,
            target=args.target,
            concurrency=args.concurrency,
            max_candidates=args.max_candidates,
            resume=not args.no_resume,
            output_label=args.output_label,
        )
        if results[intent]["summary"]["gate_reason"]:
            break
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
