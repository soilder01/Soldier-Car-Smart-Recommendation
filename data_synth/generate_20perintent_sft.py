"""Generate 20 teacher-decided SFT records per intent for manual audit."""

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.services import agent_graph
from data_synth.generate_sft_data import (
    _environment,
    audit_answer_grounding,
    audit_teacher_decision_record,
    generate_teacher_decision_record,
)
from data_synth.validate_tool_data import validate_record


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "model_training"
DATASET_PATH = OUT_DIR / "teacher_decision_20perintent_sft.jsonl"
REPORT_PATH = OUT_DIR / "20perintent_audit_report.md"
REPORT_JSON_PATH = OUT_DIR / "20perintent_audit_report.json"
FAILURES_PATH = OUT_DIR / "teacher_decision_20perintent_failures.jsonl"

INTENTS = ["recommend", "compare", "customer_service", "deep_search", "sales"]
RATE_LIMIT_BACKOFF_SECONDS = (5, 10, 20, 40)

QUERY_SETS: dict[str, list[str]] = {
    "recommend": [
        "预算28万，三口之家，主要城市通勤，想要插混SUV，关注空间和能耗，请推荐。",
        "20万以内女生通勤代步，想买纯电轿车，关注安全和续航。",
        "35万以内二胎家庭，想看大空间SUV，关注后备箱和座位。",
        "没有家充，预算22万，想买新能源SUV，担心补能，请推荐。",
        "30万左右纯电SUV，重点看智驾和快充，平时城市通勤。",
        "25万左右增程SUV，偶尔长途，关注续航和舒适。",
        "15万左右家用新能源，预算有限但要空间够用。",
        "40万以内新能源车，关注社交形象和品牌，不想太高调。",
        "45万商务接待兼家用，想看大空间新能源车。",
        "18万以内第一台新能源，想省心耐用，主要上下班。",
        "预算32万，家里有充电桩，想买纯电SUV，关注安全和空间。",
        "预算26万，想买插混或增程SUV，周末会带孩子出游。",
        "预算50万，想从燃油豪华车换新能源，关注舒适和品牌。",
        "预算24万，老人接送和孩子通勤，关注上下车便利和空间。",
        "预算30万，喜欢驾驶感和加速，想买新能源SUV。",
        "预算27万，纠结纯电和增程，城市通勤多，偶尔跨城。",
        "预算23万，想买家用轿车，关注续航、补能和保值。",
        "预算60万以内豪华纯电SUV，关注品牌、安全和空间。",
        "预算21万，家用兼业务接待，关注可靠性和空间。",
        "预算38万，想买新能源MPV或大六座，关注舒适和安全。",
    ],
    "compare": [
        "Model Y和小鹏G6怎么选？预算30万，关注智驾和空间。",
        "比亚迪宋PLUS DM-i和吉利银河L7怎么选？关注空间和能耗。",
        "问界M7和理想L7适合三口之家吗？预算35万。",
        "极氪007和小米SU7怎么选？主要城市通勤，关注续航。",
        "腾势N7和智界R7对比，谁更适合家庭SUV？",
        "理想L6和问界M5都是增程，25万左右怎么选？",
        "宝马iX3和蔚来ES6对比，关注品牌和安全。",
        "比亚迪宋L EV和大众ID.4 CROZZ怎么选？",
        "享界S9和蔚来ET7商务接待哪个更合适？",
        "极氪009和腾势D9谁更适合二胎家庭？",
        "小鹏G6和智界R7如果都看智驾，差异是什么？",
        "Model 3和极氪007，20多万纯电轿车怎么选？",
        "问界M8和理想L8，大空间家庭用车怎么选？",
        "岚图FREE和问界M7，增程SUV谁更适合长途？",
        "AION Y和宋PLUS DM-i，15万级家用怎么选？",
        "宝马i5和享界S9，商务轿车怎么选？",
        "理想L6和蔚来ES6，一个增程一个纯电，怎么取舍？",
        "小米SU7和特斯拉Model 3，年轻人第一台车怎么选？",
        "腾势D9和问界M9，商务兼家用怎么选？",
        "蔚来ET7和宝马i5，品牌、空间和补能怎么对比？",
    ],
    "customer_service": [
        "辅助驾驶是不是自动驾驶？高速上能不能放手？",
        "新能源车保修政策怎么查？能不能直接承诺终身保修？",
        "没有家充，日常用车应该注意什么？",
        "冬天续航下降是不是质量问题？",
        "电池安全怎么跟客户解释才稳妥？",
        "客户问价格和权益，客服能不能直接报最低价？",
        "增程车和插混车在使用上有什么区别？",
        "客户担心充电慢，应该怎么解释快充时间？",
        "智能座舱卡顿或升级问题应该怎么回答？",
        "客户问二手保值率，客服应如何表达？",
        "客户说试驾车和交付车配置不一样怎么办？",
        "客户问官方参数和实测不一致，怎么解释？",
        "L2辅助驾驶遇到施工路段能不能自己处理？",
        "客户问电池衰减，应该如何克制回答？",
        "客户咨询置换补贴，客服应该怎么引导核验？",
        "用户问门店覆盖少会不会影响维保，怎么答？",
        "用户问油耗和电耗，工具没有官方数值时怎么答？",
        "客户追问交付周期，客服能不能承诺具体日期？",
        "客户问事故后电池质保是否还有效，怎么答？",
        "客户问不同城市政策差异，应该怎么查询？",
    ],
    "deep_search": [
        "帮我深度研究28万内插混SUV，三口之家，关注空间、能耗和后期使用。",
        "做一次30万纯电SUV深度调研，重点看智驾、快充和家庭空间。",
        "深度分析Model Y、小鹏G6、智界R7怎么选，预算30万。",
        "帮我系统研究没有家充是否适合买新能源SUV，预算22万。",
        "深度调研35万以内二胎家庭大空间新能源车。",
        "帮我研究增程SUV适不适合经常跨城出差，预算28万。",
        "做一份豪华新能源轿车调研，预算50万，商务接待为主。",
        "深度比较插混、增程、纯电三类车在城市通勤中的取舍。",
        "帮我研究20万内纯电轿车，女生通勤，关注安全和续航。",
        "深度调研45万以内MPV或大空间新能源，商务兼家用。",
        "帮我研究新能源车辅助驾驶边界和购车风险提示。",
        "深度分析低预算家庭买新能源应该优先看哪些指标。",
        "帮我研究长途场景下纯电和增程SUV补能差异。",
        "做一份品牌、保值、售后网点对购车影响的深度分析。",
        "帮我研究三口之家是否需要大五座SUV还是轿车即可。",
        "深度调研25万左右插混SUV市场，有哪些值得看。",
        "帮我研究新能源车电池安全和质保表达，给销售使用。",
        "深度比较问界、理想、岚图三类增程车定位差异。",
        "帮我研究家用车空间指标应该怎么读：轴距、后备箱、座位数。",
        "深度调研纯电SUV用户没有家充时的使用成本和风险。",
    ],
    "sales": [
        "客户担心插混车电池安全和保修，销售怎么回应？",
        "客户关注能耗，想要低使用成本，怎么介绍插混SUV？",
        "客户没有家充，销售怎么解释增程和插混的价值？",
        "客户觉得国产新能源品牌不够高端，怎么沟通？",
        "客户担心辅助驾驶安全，销售话术怎么克制？",
        "客户预算28万但想要空间大，怎么引导试驾？",
        "客户在宋PLUS和银河L7之间犹豫，怎么促成？",
        "客户想等降价，销售怎么跟进不夸大权益？",
        "客户担心后期维保网点，销售怎么说明？",
        "客户问油耗具体数值但工具没有返回，怎么回应？",
        "客户对增程是不是落后技术有疑问，怎么解释？",
        "客户喜欢智能座舱但担心学习成本，怎么沟通？",
        "客户试驾后觉得动力一般，怎么处理异议？",
        "客户二胎家庭关注后备箱，怎么设计看车路线？",
        "客户第一次买新能源，担心充电和续航，怎么开场？",
        "客户要求最低价承诺，销售应该怎么合规表达？",
        "客户对品牌保值率有顾虑，怎么客观沟通？",
        "客户比较纯电和插混，销售怎么帮他做场景分析？",
        "客户对交付周期敏感，销售怎么引导确认官方信息？",
        "客户看重社交形象但预算有限，怎么推荐话术更稳妥？",
    ],
}


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows[record["id"]] = record
    return rows


def _load_failures(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _tool_names(record: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for message in record.get("messages", []):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls", []) or []:
            names.append(call["function"]["name"])
    return names


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


def _format_pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0/0 = 0.0%"
    return f"{numerator}/{denominator} = {numerator / denominator * 100:.1f}%"


def _trajectory_markdown(record: dict[str, Any]) -> str:
    lines = [f"#### {record['id']}", "", "```json"]
    messages = record.get("messages", [])
    lines.append(json.dumps(messages, ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines)


def _write_dataset(
    records: dict[str, dict[str, Any]],
    path: Path = DATASET_PATH,
) -> None:
    path.write_text(
        "\n".join(
            json.dumps(records[key], ensure_ascii=False, separators=(",", ":"))
            for key in sorted(records)
        )
        + ("\n" if records else ""),
        encoding="utf-8",
    )


def _append_failure(
    failure: dict[str, Any],
    path: Path = FAILURES_PATH,
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(failure, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_report(
    summary: dict[str, Any],
    samples: dict[str, list[dict[str, Any]]],
    report_path: Path = REPORT_PATH,
    report_json_path: Path = REPORT_JSON_PATH,
) -> None:
    report_json_path.write_text(
        json.dumps({"summary": summary, "samples": samples}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 20/intent 教师决策审计报告",
        "",
        "## 全局统计",
        "",
        f"- 尝试数：{summary['global']['attempted']}",
        f"- 写入数：{summary['global']['accepted']}",
        f"- 写入成功率：{_format_pct(summary['global']['accepted'], summary['global']['attempted'])}",
        f"- 截断率：{_format_pct(summary['global']['truncated'], summary['global']['attempted'])}",
        f"- grounding 改写触发率：{_format_pct(summary['global']['rewrite_triggered'], summary['global']['accepted'])}",
        f"- 平均 tool_call 轮次：{summary['global']['avg_tool_call_rounds']:.2f}",
        "",
        "## 分意图统计",
        "",
        "| intent | attempted | accepted | 写入成功率 | 截断率 | rewrite触发率 | 平均tool轮次 | 停止原因 |",
        "|---|---:|---:|---|---|---|---:|---|",
    ]
    for intent in INTENTS:
        item = summary["intents"][intent]
        lines.append(
            "| {intent} | {attempted} | {accepted} | {success_rate} | {truncation_rate} | {rewrite_rate} | {avg:.2f} | {stop_reason} |".format(
                intent=intent,
                attempted=item["attempted"],
                accepted=item["accepted"],
                success_rate=_format_pct(item["accepted"], item["attempted"]),
                truncation_rate=_format_pct(item["truncated"], item["attempted"]),
                rewrite_rate=_format_pct(item["rewrite_triggered"], item["accepted"]),
                avg=item["avg_tool_call_rounds"],
                stop_reason=item["stop_reason"] or "",
            )
        )
    lines.extend(["", "## 每意图抽样轨迹"])
    for intent in INTENTS:
        lines.extend(["", f"### {intent}", ""])
        if not samples[intent]:
            lines.append("无可抽样成功轨迹。")
            continue
        for record in samples[intent]:
            lines.append(_trajectory_markdown(record))
            lines.append("")
    lines.extend(["", "## Fail-Closed 失败记录", ""])
    any_failure = False
    for intent in INTENTS:
        failures = summary["intents"][intent]["failures"]
        if not failures:
            continue
        any_failure = True
        lines.extend(["", f"### {intent}", ""])
        for failure in failures:
            reason = failure.get("error") or failure.get("decision_audit") or failure.get("grounding_audit") or failure.get("validate_record")
            lines.append(
                "- {id}：{error_type}，query={query}，reason={reason}".format(
                    id=failure.get("id", ""),
                    error_type=failure.get("error_type", ""),
                    query=failure.get("query", ""),
                    reason=reason,
                )
            )
    if not any_failure:
        lines.append("无。")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _summarize(records: dict[str, dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    by_intent_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_intent_failures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records.values():
        by_intent_records[record.get("intent", "")].append(record)
    for failure in failures:
        by_intent_failures[failure["intent"]].append(failure)

    summary = {"global": {}, "intents": {}}
    global_attempted = 0
    global_accepted = 0
    global_truncated = 0
    global_rewrite = 0
    global_rounds = 0
    for intent in INTENTS:
        accepted_rows = by_intent_records[intent]
        failure_rows = by_intent_failures[intent]
        attempted = len(accepted_rows) + len(failure_rows)
        truncated = sum(1 for item in failure_rows if item["error_type"] == "length")
        rewrite = sum(1 for item in accepted_rows if item.get("bounded_rewrite_triggered"))
        rounds = sum(int(item.get("tool_call_rounds", 0)) for item in accepted_rows)
        stop_reason = ""
        if attempted and truncated / attempted >= 0.05:
            stop_reason = "截断率>=5%，停止该意图"
        summary["intents"][intent] = {
            "attempted": attempted,
            "accepted": len(accepted_rows),
            "truncated": truncated,
            "rewrite_triggered": rewrite,
            "avg_tool_call_rounds": rounds / len(accepted_rows) if accepted_rows else 0.0,
            "stop_reason": stop_reason,
            "failures": failure_rows,
        }
        global_attempted += attempted
        global_accepted += len(accepted_rows)
        global_truncated += truncated
        global_rewrite += rewrite
        global_rounds += rounds
    summary["global"] = {
        "attempted": global_attempted,
        "accepted": global_accepted,
        "truncated": global_truncated,
        "rewrite_triggered": global_rewrite,
        "avg_tool_call_rounds": global_rounds / global_accepted if global_accepted else 0.0,
    }
    return summary


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).casefold()
    return status_code == 429 or "rate limit" in message or "serveroverloaded" in message


def _generate_record_with_backoff(**kwargs: Any) -> dict[str, Any]:
    for retry_index in range(len(RATE_LIMIT_BACKOFF_SECONDS) + 1):
        try:
            return generate_teacher_decision_record(**kwargs)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or retry_index >= len(RATE_LIMIT_BACKOFF_SECONDS):
                raise
            delay = RATE_LIMIT_BACKOFF_SECONDS[retry_index]
            print(
                f"RETRY rate_limit attempt={retry_index + 1} wait={delay}s",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable rate-limit retry state")


def generate_dataset(*, limit_per_intent: int, resume: bool) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values = _environment(ROOT / ".env")
    client = OpenAI(
        base_url=values["ARK_BASE_URL"],
        api_key=values["ARK_API_KEY"],
        timeout=240,
        max_retries=0,
    )
    records = _load_existing(DATASET_PATH) if resume else {}
    failures = _load_failures(FAILURES_PATH) if resume else []
    for intent in INTENTS:
        queries = QUERY_SETS[intent][:limit_per_intent]
        for index, query in enumerate(queries, 1):
            record_id = f"20pi-{intent}-{index:02d}"
            if record_id in records:
                print(f"SKIP {record_id} existing", flush=True)
                continue
            started = time.time()
            print(f"START {record_id} intent={intent}", flush=True)
            try:
                record = generate_teacher_decision_record(
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
                    failures.append(
                        failure := {
                            "id": record_id,
                            "intent": intent,
                            "query": query,
                            "error_type": "audit",
                            "validate_record": validate_errors,
                            "decision_audit": decision_errors,
                            "grounding_audit": grounding_errors,
                            "elapsed_sec": round(time.time() - started, 2),
                        }
                    )
                    _append_failure(failure)
                    print(
                        f"FAIL {record_id} audit elapsed={time.time() - started:.1f}s",
                        flush=True,
                    )
                    continue
                if intent == "compare":
                    lookup = _extract_compare_lookup(record)
                    if lookup is None:
                        failures.append(
                            failure := {
                                "id": record_id,
                                "intent": intent,
                                "query": query,
                                "error_type": "audit",
                                "validate_record": [],
                                "decision_audit": [
                                    "compare named_vehicle_lookup metadata is missing"
                                ],
                                "grounding_audit": [],
                                "elapsed_sec": round(time.time() - started, 2),
                            }
                        )
                        _append_failure(failure)
                        print(
                            f"FAIL {record_id} audit elapsed={time.time() - started:.1f}s",
                            flush=True,
                        )
                        continue
                    record["named_vehicle_lookup"] = lookup
                record["validate_record"] = validate_errors
                record["decision_audit"] = decision_errors
                record["grounding_audit"] = grounding_errors
                record["tool_names"] = _tool_names(record)
                record["elapsed_sec"] = round(time.time() - started, 2)
                records[record_id] = record
                _write_dataset(records)
                print(
                    "OK {record_id} elapsed={elapsed:.1f}s rounds={rounds} tools={tools}".format(
                        record_id=record_id,
                        elapsed=time.time() - started,
                        rounds=record.get("tool_call_rounds", 0),
                        tools=",".join(record.get("tool_names", [])),
                    ),
                    flush=True,
                )
            except Exception as exc:
                message = str(exc)
                error_type = "length" if "truncated" in message else type(exc).__name__
                failures.append(
                    failure := {
                        "id": record_id,
                        "intent": intent,
                        "query": query,
                        "error_type": error_type,
                        "error": message,
                        "elapsed_sec": round(time.time() - started, 2),
                    }
                )
                _append_failure(failure)
                print(
                    f"FAIL {record_id} {error_type} elapsed={time.time() - started:.1f}s error={message[:160]}",
                    flush=True,
                )
                intent_failures = [item for item in failures if item["intent"] == intent]
                intent_attempts = (
                    len([item for item in records.values() if item.get("intent") == intent])
                    + len(intent_failures)
                )
                if intent_attempts and sum(1 for item in intent_failures if item["error_type"] == "length") / intent_attempts >= 0.05:
                    break
    summary = _summarize(records, failures)
    samples = {
        intent: sorted(
            [record for record in records.values() if record.get("intent") == intent],
            key=lambda item: item["id"],
        )[:3]
        for intent in INTENTS
    }
    _write_report(summary, samples)
    return {"summary": summary, "dataset": str(DATASET_PATH), "report": str(REPORT_PATH)}


def generate_single_intent_dataset(
    *,
    intent: str,
    limit_per_intent: int,
    dataset_path: Path,
    report_path: Path,
    report_json_path: Path,
    failures_path: Path,
    resume: bool = True,
    queries: list[str] | None = None,
    record_prefix: str | None = None,
    rewrite_baseline_rate: float | None = None,
    rewrite_rise_points: float = 10.0,
    minimum_gate_samples: int = 50,
    concurrency: int = 1,
) -> dict[str, Any]:
    if intent not in INTENTS:
        raise ValueError(f"unsupported intent: {intent}")
    if concurrency < 1 or concurrency > 64:
        raise ValueError("concurrency must be between 1 and 64")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values = _environment(ROOT / ".env")
    client = OpenAI(
        base_url=values["ARK_BASE_URL"],
        api_key=values["ARK_API_KEY"],
        timeout=240,
        max_retries=0,
    )
    candidate_queries = queries or QUERY_SETS[intent]
    selected_queries = candidate_queries[:limit_per_intent]
    if len(selected_queries) < limit_per_intent:
        raise ValueError(
            f"not enough queries for {intent}: "
            f"required {limit_per_intent}, got {len(selected_queries)}"
        )
    records = _load_existing(dataset_path) if resume else {}
    failures = _load_failures(failures_path) if resume else []
    if not resume:
        for path in (dataset_path, report_path, report_json_path, failures_path):
            if path.exists():
                path.unlink()
    id_prefix = record_prefix or f"20pi-{intent}-rerun"

    def write_checkpoint() -> dict[str, Any]:
        summary = _summarize(records, failures)
        samples = {
            item: sorted(
                [record for record in records.values() if record.get("intent") == item],
                key=lambda row: row["id"],
            )[:3]
            for item in INTENTS
        }
        _write_report(
            summary,
            samples,
            report_path=report_path,
            report_json_path=report_json_path,
        )
        return summary

    pending: list[tuple[int, str, str]] = []
    for index, query in enumerate(candidate_queries, 1):
        record_id = f"{id_prefix}-{index:03d}"
        if record_id in records:
            print(f"SKIP {record_id} existing", flush=True)
            continue
        pending.append((index, query, record_id))

    def generate_one(item: tuple[int, str, str]) -> tuple[str, str, dict[str, Any]]:
        _, query, record_id = item
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
                            "error_type": "audit",
                            "validate_record": [],
                            "decision_audit": [
                                "compare named_vehicle_lookup metadata is missing"
                            ],
                            "grounding_audit": [],
                            "elapsed_sec": round(time.time() - started, 2),
                        },
                    )
                record["named_vehicle_lookup"] = lookup
            record["validate_record"] = validate_errors
            record["decision_audit"] = decision_errors
            record["grounding_audit"] = grounding_errors
            record["tool_names"] = _tool_names(record)
            record["elapsed_sec"] = round(time.time() - started, 2)
            return "success", record_id, record
        except Exception as exc:
            message = str(exc)
            error_type = "length" if "truncated" in message else type(exc).__name__
            return (
                "failure",
                record_id,
                {
                    "id": record_id,
                    "intent": intent,
                    "query": query,
                    "error_type": error_type,
                    "error": message,
                    "elapsed_sec": round(time.time() - started, 2),
                },
            )

    stop_requested = False
    for batch_start in range(0, len(pending), concurrency):
        if len(records) >= limit_per_intent:
            break
        batch = pending[batch_start:batch_start + concurrency]
        with ThreadPoolExecutor(max_workers=min(concurrency, len(batch))) as executor:
            futures = [executor.submit(generate_one, item) for item in batch]
            for future in as_completed(futures):
                status, record_id, payload = future.result()
                if status == "success":
                    records[record_id] = payload
                    _write_dataset(records, dataset_path)
                    print(
                        "OK {record_id} elapsed={elapsed:.1f}s rounds={rounds} tools={tools}".format(
                            record_id=record_id,
                            elapsed=payload.get("elapsed_sec", 0.0),
                            rounds=payload.get("tool_call_rounds", 0),
                            tools=",".join(payload.get("tool_names", [])),
                        ),
                        flush=True,
                    )
                else:
                    failures.append(payload)
                    _append_failure(payload, failures_path)
                    print(
                        f"FAIL {record_id} {payload['error_type']} "
                        f"elapsed={payload.get('elapsed_sec', 0.0):.1f}s "
                        f"error={str(payload.get('error') or payload.get('decision_audit') or payload.get('grounding_audit'))[:160]}",
                        flush=True,
                    )
                summary = write_checkpoint()
                intent_summary = summary["intents"][intent]
                accepted = intent_summary["accepted"]
                rewrite_rate = (
                    intent_summary["rewrite_triggered"] / accepted * 100
                    if accepted else 0.0
                )
                rewrite_stop_rate = (
                    max(15.0, rewrite_baseline_rate + rewrite_rise_points)
                    if rewrite_baseline_rate is not None else None
                )
                if (
                    rewrite_stop_rate is not None
                    and accepted >= minimum_gate_samples
                    and rewrite_rate > rewrite_stop_rate
                ):
                    stop_requested = True
                if (
                    intent_summary["attempted"]
                    and intent_summary["truncated"] / intent_summary["attempted"] >= 0.05
                ):
                    stop_requested = True
        if stop_requested:
            break

    summary = write_checkpoint()
    return {"summary": summary, "dataset": str(dataset_path), "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-intent", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--only-intent", choices=INTENTS)
    parser.add_argument("--output-prefix")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.only_intent:
        prefix = args.output_prefix or f"teacher_decision_20perintent_{args.only_intent}_rerun"
        result = generate_single_intent_dataset(
            intent=args.only_intent,
            limit_per_intent=args.limit_per_intent,
            dataset_path=OUT_DIR / f"{prefix}.jsonl",
            report_path=OUT_DIR / f"{prefix}_audit_report.md",
            report_json_path=OUT_DIR / f"{prefix}_audit_report.json",
            failures_path=OUT_DIR / f"{prefix}_failures.jsonl",
            resume=not args.no_resume,
            concurrency=args.concurrency,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = generate_dataset(
        limit_per_intent=args.limit_per_intent,
        resume=not args.no_resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
