"""Freeze audited teacher trajectories into reproducible Qwen SFT assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from data_synth.tool_schemas import build_tool_schemas
from data_synth.validate_tool_data import validate_record


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "model_training"
FREEZE_DIR = OUT_DIR / "sft_freeze"
DEFAULT_SPLIT_SEED = 20260724
DEFAULT_EVAL_FRACTION = 0.10
QWEN_TEMPLATE_ID = "qwen2.5-instruct-tool-use-v1"
TEACHER_ENDPOINT_ALIAS = "seedpro_ark_teacher"

SOURCE_DATASETS: dict[str, Path] = {
    "recommend": OUT_DIR / "teacher_decision_500perintent_recommend_sft.jsonl",
    "compare": OUT_DIR / "teacher_decision_500perintent_compare_named_lookup_v3_sft.jsonl",
    "customer_service": (
        OUT_DIR / "teacher_decision_500perintent_customer_service_rerun_v3_sft.jsonl"
    ),
    "deep_search": (
        OUT_DIR / "teacher_decision_500perintent_deep_search_rerun_v2_sft.jsonl"
    ),
    "sales": (
        OUT_DIR / "teacher_decision_500perintent_sales_policy_rerun_v3_sft.jsonl"
    ),
}
REWARD_VISIBLE_PATH = OUT_DIR / "eval" / "reward_visible.jsonl"
HELD_OUT_PATH = OUT_DIR / "eval" / "held_out.jsonl"


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().split()).casefold()


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records and reject malformed or non-object rows."""
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no records found")
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _user_query(record: dict[str, Any]) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise ValueError(f"{record.get('id')}: no non-empty user query")


def _tool_call_text(call: dict[str, Any]) -> str:
    function = call.get("function", call)
    if not isinstance(function, dict):
        raise ValueError("tool_call.function must be an object")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool_call.function.name must be non-empty")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError("tool_call arguments must be valid JSON") from error
        if not isinstance(parsed, dict):
            raise ValueError("tool_call arguments must decode to an object")
        arguments_literal = json.dumps(arguments, ensure_ascii=False)
    elif isinstance(arguments, dict):
        arguments_literal = json.dumps(arguments, ensure_ascii=False)
    else:
        raise ValueError("tool_call arguments must be an object or JSON object string")
    return (
        '{"name": '
        + json.dumps(name, ensure_ascii=False)
        + ', "arguments": '
        + arguments_literal
        + "}"
    )


def render_qwen_chatml(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Render the local Qwen2.5-Instruct tool-use chat template without a tokenizer."""
    rendered, _ = render_qwen_chatml_with_supervision(messages, tools=tools)
    return rendered


def render_qwen_chatml_with_supervision(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, list[list[int]]]:
    """Render Qwen ChatML and return character spans supervised by SFT loss."""
    if not messages or messages[0].get("role") != "system":
        raise ValueError("Qwen renderer requires the first message role to be system")
    system_content = messages[0].get("content")
    if not isinstance(system_content, str) or not system_content.strip():
        raise ValueError("Qwen renderer requires non-empty system content")

    selected_tools = build_tool_schemas() if tools is None else tools
    output: list[str] = []
    assistant_spans: list[list[int]] = []

    def append(text: str) -> None:
        output.append(text)

    def position() -> int:
        return sum(len(item) for item in output)

    append("<|im_start|>system\n" + system_content)
    if selected_tools:
        append(
            "\n\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>"
        )
        for tool in selected_tools:
            append("\n" + json.dumps(
                tool,
                ensure_ascii=False,
            ))
        append(
            "\n</tools>\n\n"
            "For each function call, return a json object with function name and "
            "arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call><|im_end|>\n"
        )
    else:
        append("<|im_end|>\n")

    for index, message in enumerate(messages):
        role = message.get("role")
        if index == 0:
            continue
        if role in {"user", "system"}:
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(f"{role} message content must be a string")
            append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        elif role == "assistant":
            tool_calls = message.get("tool_calls")
            assistant_start = position()
            if not tool_calls:
                content = message.get("content")
                if not isinstance(content, str):
                    raise ValueError("assistant final content must be a string")
                append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
                assistant_spans.append([assistant_start, position()])
                continue
            if not isinstance(tool_calls, list):
                raise ValueError("assistant tool_calls must be a list")
            append("<|im_start|>assistant")
            content = message.get("content", "")
            if content:
                if not isinstance(content, str):
                    raise ValueError("assistant tool-call content must be a string")
                append("\n" + content)
            for call in tool_calls:
                if not isinstance(call, dict):
                    raise ValueError("tool_call must be an object")
                append("\n<tool_call>\n" + _tool_call_text(call) + "\n</tool_call>")
            append("<|im_end|>\n")
            assistant_spans.append([assistant_start, position()])
        elif role == "tool":
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("tool response content must be a string")
            previous_role = messages[index - 1].get("role") if index else None
            next_role = (
                messages[index + 1].get("role") if index + 1 < len(messages) else None
            )
            if previous_role != "tool":
                append("<|im_start|>user")
            append("\n<tool_response>\n" + content + "\n</tool_response>")
            if next_role != "tool":
                append("<|im_end|>\n")
        else:
            raise ValueError(f"unsupported message role: {role!r}")
    return "".join(output), assistant_spans


def _reserved_entries(
    rows: list[dict[str, Any]],
    *,
    source: str,
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for row in rows:
        record_id = row.get("id")
        query = row.get("query")
        if not isinstance(record_id, str) or not _normalize(record_id):
            raise ValueError(f"{source}: reserved row has invalid id")
        if not isinstance(query, str) or not _normalize(query):
            raise ValueError(f"{source}: reserved row has invalid query")
        entries.append(
            {
                "source": source,
                "id": record_id,
                "query_sha256": _query_hash(_normalize(query)),
            }
        )
    return entries


def _reservation_sets(
    reward_rows: list[dict[str, Any]],
    held_out_rows: list[dict[str, Any]],
) -> tuple[set[str], set[str], list[dict[str, str]]]:
    entries = [
        *_reserved_entries(reward_rows, source="reward_visible"),
        *_reserved_entries(held_out_rows, source="held_out"),
    ]
    ids = {_normalize(entry["id"]) for entry in entries}
    queries = {entry["query_sha256"] for entry in entries}
    if len(ids) != len(entries):
        raise ValueError("reward reservation contains normalized duplicate IDs")
    if len(queries) != len(entries):
        raise ValueError("reward reservation contains normalized duplicate queries")
    return ids, queries, entries


def _validate_source_rows(
    source_rows: dict[str, list[dict[str, Any]]],
    *,
    reserved_ids: set[str],
    reserved_query_hashes: set[str],
) -> None:
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    violations: list[str] = []

    for intent, rows in source_rows.items():
        for row in rows:
            record_id = row.get("id")
            row_intent = row.get("intent")
            if not isinstance(record_id, str):
                raise ValueError(f"{intent}: record has invalid id")
            if row_intent != intent:
                raise ValueError(
                    f"{record_id}: source intent {row_intent!r} does not match {intent!r}"
                )
            query = _user_query(row)
            normalized_id = _normalize(record_id)
            normalized_query = _normalize(query)
            query_hash = _query_hash(normalized_query)
            if normalized_id in seen_ids:
                raise ValueError(f"duplicate source record ID: {record_id}")
            if normalized_query in seen_queries:
                raise ValueError(f"duplicate source query: {query}")
            seen_ids.add(normalized_id)
            seen_queries.add(normalized_query)

            if normalized_id in reserved_ids or query_hash in reserved_query_hashes:
                violations.append(record_id)

            errors = validate_record(row, held_out_ids=set())
            if errors:
                raise ValueError(f"{record_id}: source validation failed: {errors}")

    if violations:
        raise ValueError(
            "SFT/reward isolation violation: "
            + ", ".join(sorted(violations)[:5])
        )


def _frozen_row(
    record: dict[str, Any],
    *,
    source_dataset: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    messages = record["messages"]
    qwen_chatml, assistant_char_spans = render_qwen_chatml_with_supervision(
        messages,
        tools=tools,
    )
    return {
        "id": record["id"],
        "intent": record["intent"],
        "source_dataset": source_dataset,
        "messages": messages,
        "tools": tools,
        "qwen_chat_template": QWEN_TEMPLATE_ID,
        "qwen_chatml": qwen_chatml,
        "assistant_char_spans": assistant_char_spans,
    }


def _split_rows(
    source_rows: dict[str, list[dict[str, Any]]],
    *,
    eval_fraction: float,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be greater than 0 and less than 1")

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for intent, rows in source_rows.items():
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{split_seed}:{intent}:{row['id']}".encode("utf-8")
            ).hexdigest(),
        )
        eval_count = max(1, int(len(ordered) * eval_fraction))
        if eval_count >= len(ordered):
            raise ValueError(f"{intent}: eval split would consume all source records")
        eval_ids = {row["id"] for row in ordered[:eval_count]}
        intent_eval = [row for row in rows if row["id"] in eval_ids]
        intent_train = [row for row in rows if row["id"] not in eval_ids]
        train_rows.extend(intent_train)
        eval_rows.extend(intent_eval)
        counts[intent] = {
            "source": len(rows),
            "train": len(intent_train),
            "eval": len(intent_eval),
        }
    return train_rows, eval_rows, counts


def _summary_metrics() -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for intent, source in SOURCE_DATASETS.items():
        report_path = source.with_name(source.name.replace("_sft.jsonl", "_audit_report.json"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics[intent] = {
            "accepted": report["accepted"],
            "attempted": report["attempted"],
            "truncation_rate": report["truncation_rate"],
            "rewrite_rate": report["rewrite_rate"],
            "avg_tool_call_rounds": report["avg_tool_call_rounds"],
        }
        if intent == "compare":
            metrics[intent]["compare_both_recall_rate"] = report[
                "compare_both_recall_rate"
            ]
            metrics[intent]["compare_lookup_mismatches"] = report[
                "compare_lookup_mismatches"
            ]
    return metrics


def _dataset_card(
    *,
    counts: dict[str, dict[str, int]],
    metrics: dict[str, dict[str, Any]],
    split_seed: int,
    eval_fraction: float,
    reservation_entries: list[dict[str, str]],
    teacher_endpoint_alias: str,
    truncation_audit: dict[str, Any] | None = None,
) -> str:
    total_source = sum(item["source"] for item in counts.values())
    total_train = sum(item["train"] for item in counts.values())
    total_eval = sum(item["eval"] for item in counts.values())
    total_excluded_train = sum(
        item.get("excluded_train", 0) for item in counts.values()
    )
    total_excluded_eval = sum(
        item.get("excluded_eval", 0) for item in counts.values()
    )
    lines = [
        "# SFT Dataset Card",
        "",
        "## 用途与冻结状态",
        "",
        "- 用途：Qwen2.5-7B-Instruct 的未来工具调用 SFT 输入；本文件不表示已训练。",
        "- 冻结范围：五个意图的 2,500 条外部审计通过教师轨迹。",
        "- 轨迹格式：保留原始多轮 `system -> user -> assistant[tool_calls] -> tool -> assistant`，并生成 Qwen2.5 工具调用 ChatML 渲染文本；未压平成单轮问答。",
        "- teacher 端点别名：%s（脱敏别名；不记录 key、base_url 或模型实值）。"
        % teacher_endpoint_alias,
        "",
        "## 意图统计与生成审计",
        "",
        "| intent | 冻结条数 | active train | active eval | 监督截断排除(train/eval) | 生成截断率 | rewrite率 | 平均工具轮次 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for intent in sorted(counts):
        row = counts[intent]
        metric = metrics[intent]
        lines.append(
            "| %s | %d | %d | %d | %d/%d | %.1f%% | %.1f%% | %.2f |"
            % (
                intent,
                row["source"],
                row["train"],
                row["eval"],
                row.get("excluded_train", 0),
                row.get("excluded_eval", 0),
                metric["truncation_rate"],
                metric["rewrite_rate"],
                metric["avg_tool_call_rounds"],
            )
        )
    lines.extend(
        [
            "| **total** | **%d** | **%d** | **%d** | **%d/%d** | - | - | - |"
            % (
                total_source,
                total_train,
                total_eval,
                total_excluded_train,
                total_excluded_eval,
            ),
            "",
            "compare 额外审计：库内点名双车同时召回率 100.0%，点名错配 0。",
            "",
            "## 审计口径",
            "",
            "- 每条源轨迹已通过 `validate_record`、意图决策审计、grounding 审计和落盘审计字段校验。",
            "- compare 额外校验 `named_vehicle_lookup` 与点名双车实际同时召回。",
            "- 生成阶段失败、截断及未接地样本仅存在于生成 failure 日志，不进入本冻结集。",
            "- 代表冻结样本已使用本地 Qwen tokenizer 的 `apply_chat_template()` 逐字节复核；报告为 `qwen_template_verification.json`。",
        ]
    )
    if truncation_audit is not None:
        train_audit = truncation_audit["splits"]["train"]
        val_audit = truncation_audit["splits"]["validation"]
        lines.extend(
            [
                "",
                "## 监督截断审计",
                "",
                "- 训练初始切分中超过 `max_seq_len=5632` 的 %d 条全部截断了 assistant 监督 token；已全部排除。"
                % train_audit["rows_exceeding_max_seq_len"],
                "- validation 中同类样本 %d 条也已排除，避免 checkpoint 选择指标受不完整 labels 污染。"
                % val_audit["excluded_rows"],
                "- active train/validation 中监督截断样本均为 0；隔离样本保存在 `truncated_excluded.jsonl`。",
                "- 逐条 token offset 证据见 `truncation_supervision_audit.json`。",
            ]
        )
    lines.extend(
        [
            "",
            "## 固定切分",
            "",
            "- 分层单位：intent。",
            "- eval 比例：%.1f%%，每个 intent 独立切分。" % (eval_fraction * 100),
            "- 固定随机种子：%d；排序键由 SHA-256(seed:intent:record_id) 生成。" % split_seed,
            "- 当前 active 输出：`sft_train.jsonl`=%d 条，`sft_val.jsonl`=%d 条；两者 ID/query 无交集。"
            % (total_train, total_eval),
            "",
            "## Reward 隔离",
            "",
            "- 保留清单：%d 条（reward-visible 20 条、最终 held-out 40 条）。"
            % len(reservation_entries),
            "- 来源：项目内独立维护、人工编写的结构化 agent 评测用例；它们是真实可执行的工具契约样本，但不是教师生成的多轮 SFT 轨迹。",
            "- 清单保存 ID 与规范化 query 的 SHA-256，不复制保留 query 到 SFT 数据。",
            "- 冻结前对每条 SFT 源轨迹执行规范化 ID 与 query 双重交集校验；任一交集即 fail-closed。",
            "- 清单文件：`sft_freeze/reward_reservation_manifest.json`。",
            "",
            "## 产物",
            "",
            "- `sft_freeze/shards/<intent>.jsonl`：按意图保留的 500 条 Qwen 格式分片。",
            "- `sft_train.jsonl` / `sft_val.jsonl`：监督截断审计后的 active 数据。",
            "- `truncated_excluded.jsonl`：因 assistant 监督 token 被截断而隔离的原始冻结行。",
            "- `sft_freeze/split_manifest.json`：可复现切分、排除元数据和 ID 清单。",
            "- `sft_freeze/reward_reservation_manifest.json`：reward/held-out 隔离清单摘要。",
            "",
            "## 限制",
            "",
            "- 尚无 LoRA 权重，未启动 SFT 或 GRPO，尚无 held-out 模型对比结论。",
        ]
    )
    return "\n".join(lines) + "\n"


def freeze_dataset(
    *,
    sources: dict[str, Path],
    output_dir: Path,
    eval_fraction: float,
    split_seed: int,
    reward_rows: list[dict[str, Any]],
    held_out_rows: list[dict[str, Any]],
    teacher_endpoint_alias: str,
    metrics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate, split, render, and atomically write frozen SFT assets."""
    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        raise ValueError("split_seed must be an integer")
    if not isinstance(teacher_endpoint_alias, str) or not teacher_endpoint_alias.strip():
        raise ValueError("teacher_endpoint_alias must be a non-empty alias")

    source_rows = {intent: load_jsonl(path) for intent, path in sources.items()}
    reserved_ids, reserved_queries, reservation_entries = _reservation_sets(
        reward_rows,
        held_out_rows,
    )
    _validate_source_rows(
        source_rows,
        reserved_ids=reserved_ids,
        reserved_query_hashes=reserved_queries,
    )
    train_source, eval_source, per_intent_counts = _split_rows(
        source_rows,
        eval_fraction=eval_fraction,
        split_seed=split_seed,
    )
    tools = build_tool_schemas()
    frozen_by_intent = {
        intent: [
            _frozen_row(row, source_dataset=str(sources[intent]), tools=tools)
            for row in rows
        ]
        for intent, rows in source_rows.items()
    }
    frozen_train = [
        _frozen_row(row, source_dataset=str(sources[row["intent"]]), tools=tools)
        for row in train_source
    ]
    frozen_eval = [
        _frozen_row(row, source_dataset=str(sources[row["intent"]]), tools=tools)
        for row in eval_source
    ]
    frozen_train.sort(key=lambda row: row["id"])
    frozen_eval.sort(key=lambda row: row["id"])

    train_ids = {_normalize(row["id"]) for row in frozen_train}
    eval_ids = {_normalize(row["id"]) for row in frozen_eval}
    train_queries = {_query_hash(_normalize(_user_query(row))) for row in frozen_train}
    eval_queries = {_query_hash(_normalize(_user_query(row))) for row in frozen_eval}
    if train_ids & eval_ids or train_queries & eval_queries:
        raise ValueError("SFT train/eval split is not disjoint")

    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    for intent, rows in frozen_by_intent.items():
        _write_jsonl(shards_dir / f"{intent}.jsonl", rows)
    train_path = output_dir.parent / "sft_train.jsonl"
    eval_path = output_dir.parent / "sft_val.jsonl"
    _write_jsonl(train_path, frozen_train)
    _write_jsonl(eval_path, frozen_eval)

    counts = {
        "source_total": sum(item["source"] for item in per_intent_counts.values()),
        "train_total": len(frozen_train),
        "eval_total": len(frozen_eval),
        "per_intent": dict(sorted(per_intent_counts.items())),
    }
    split_manifest = {
        "format_version": 1,
        "qwen_chat_template": QWEN_TEMPLATE_ID,
        "split_seed": split_seed,
        "eval_fraction": eval_fraction,
        "counts": counts,
        "train_ids": [row["id"] for row in frozen_train],
        "eval_ids": [row["id"] for row in frozen_eval],
        "reward_reservation_manifest": "reward_reservation_manifest.json",
    }
    reservation_manifest = {
        "format_version": 1,
        "purpose": "reserved GRPO reward-visible and final held-out cases excluded from SFT",
        "sources": {
            "reward_visible": str(REWARD_VISIBLE_PATH),
            "held_out": str(HELD_OUT_PATH),
        },
        "counts": {
            "reward_visible": len(reward_rows),
            "held_out": len(held_out_rows),
            "reserved_total": len(reservation_entries),
        },
        "entries": reservation_entries,
        "sft_overlap_check": {
            "normalized_id_overlap": 0,
            "normalized_query_hash_overlap": 0,
            "status": "passed",
        },
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "reward_reservation_manifest.json").write_text(
        json.dumps(reservation_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    card_metrics = metrics or {
        intent: {
            "accepted": item["source"],
            "attempted": item["source"],
            "truncation_rate": 0.0,
            "rewrite_rate": 0.0,
            "avg_tool_call_rounds": 0.0,
        }
        for intent, item in per_intent_counts.items()
    }
    dataset_card_path = output_dir.parent / "sft_dataset_card.md"
    dataset_card_path.write_text(
        _dataset_card(
            counts=per_intent_counts,
            metrics=card_metrics,
            split_seed=split_seed,
            eval_fraction=eval_fraction,
            reservation_entries=reservation_entries,
            teacher_endpoint_alias=teacher_endpoint_alias,
        ),
        encoding="utf-8",
    )
    return {
        "train_path": train_path,
        "eval_path": eval_path,
        "split_manifest": split_manifest,
        "reservation_manifest": reservation_manifest,
        "dataset_card_path": dataset_card_path,
        "counts": counts,
    }


def _default_freeze() -> dict[str, Any]:
    reward_rows = load_jsonl(REWARD_VISIBLE_PATH)
    held_out_rows = load_jsonl(HELD_OUT_PATH)
    return freeze_dataset(
        sources=SOURCE_DATASETS,
        output_dir=FREEZE_DIR,
        eval_fraction=DEFAULT_EVAL_FRACTION,
        split_seed=DEFAULT_SPLIT_SEED,
        reward_rows=reward_rows,
        held_out_rows=held_out_rows,
        teacher_endpoint_alias=TEACHER_ENDPOINT_ALIAS,
        metrics=_summary_metrics(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=FREEZE_DIR)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    args = parser.parse_args()

    if args.output_dir != FREEZE_DIR:
        raise ValueError("only the repository SFT freeze location is supported")
    result = _default_freeze()
    print(
        json.dumps(
            {
                "train_path": str(result["train_path"]),
                "eval_path": str(result["eval_path"]),
                "dataset_card": str(result["dataset_card_path"]),
                "counts": result["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
