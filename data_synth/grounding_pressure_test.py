"""Grounding validator pressure test with diverse local tool evidence."""

import json
from pathlib import Path
from typing import Any

from app.services import agent_graph
from data_synth.generate_sft_data import audit_answer_grounding


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "model_training"


PRESSURE_CASES: list[dict[str, Any]] = [
    {
        "id": "p01_plugin_family",
        "intent": "recommend",
        "query": "预算28万，三口之家，城市通勤，想要插混SUV，关注空间和能耗。",
        "args": {"budget_max": 280000, "preferred_type": "SUV", "preferred_energy": "插混", "concerns": "空间,能耗", "top_k": 5},
    },
    {
        "id": "p02_pure_ev_suv",
        "intent": "recommend",
        "query": "预算28万，想买纯电SUV，关注空间和补能。",
        "args": {"budget_max": 280000, "preferred_type": "SUV", "preferred_energy": "纯电", "concerns": "空间,补能", "top_k": 5},
    },
    {
        "id": "p03_pure_ev_sedan",
        "intent": "recommend",
        "query": "20万左右纯电轿车，主要上下班通勤，关注续航。",
        "args": {"budget_max": 200000, "preferred_type": "轿车", "preferred_energy": "纯电", "concerns": "续航", "top_k": 5},
    },
    {
        "id": "p04_family_mpv",
        "intent": "recommend",
        "query": "35万以内二胎家庭，想看MPV，关注空间和舒适。",
        "args": {"budget_max": 350000, "preferred_type": "MPV", "preferred_energy": "", "concerns": "空间,舒适", "top_k": 5},
    },
    {
        "id": "p05_luxury_sedan",
        "intent": "recommend",
        "query": "50万左右商务轿车，关注品牌和座舱。",
        "args": {"budget_max": 500000, "preferred_type": "轿车", "preferred_energy": "", "concerns": "品牌,座舱", "top_k": 5},
    },
    {
        "id": "p06_no_home_charger",
        "intent": "recommend",
        "query": "22万以内没有家充，想买SUV，担心补能。",
        "args": {"budget_max": 220000, "preferred_type": "SUV", "preferred_energy": "", "concerns": "补能,空间", "top_k": 5},
    },
    {
        "id": "p07_compare_modely_g6",
        "intent": "compare",
        "query": "Model Y和小鹏G6怎么选，预算30万，关注智驾和空间。",
        "args": {"budget_max": 300000, "preferred_type": "SUV", "preferred_energy": "纯电", "concerns": "智驾,空间", "top_k": 5},
    },
    {
        "id": "p08_compare_plugin_pair",
        "intent": "compare",
        "query": "比亚迪宋PLUS DM-i和吉利银河L7怎么选？",
        "args": {"budget_max": 180000, "preferred_type": "SUV", "preferred_energy": "插混", "concerns": "空间,能耗", "top_k": 5},
    },
    {
        "id": "p09_sales_warranty",
        "intent": "sales",
        "query": "客户担心插混车电池安全和保修，怎么推荐？",
        "args": {"budget_max": 250000, "preferred_type": "SUV", "preferred_energy": "插混", "concerns": "安全,保修", "top_k": 5},
    },
    {
        "id": "p10_sales_energy",
        "intent": "sales",
        "query": "客户关注能耗，想要低使用成本，怎么介绍插混SUV？",
        "args": {"budget_max": 260000, "preferred_type": "SUV", "preferred_energy": "插混", "concerns": "能耗", "top_k": 5},
    },
    {
        "id": "p11_smart_ev",
        "intent": "recommend",
        "query": "30万纯电SUV，重点看智驾和快充。",
        "args": {"budget_max": 300000, "preferred_type": "SUV", "preferred_energy": "纯电", "concerns": "智驾,快充", "top_k": 5},
    },
    {
        "id": "p12_budget_first",
        "intent": "recommend",
        "query": "15万左右家用新能源，预算有限但要空间够用。",
        "args": {"budget_max": 150000, "preferred_type": "SUV", "preferred_energy": "", "concerns": "空间,预算有限", "top_k": 5},
    },
    {
        "id": "p13_business_mpv",
        "intent": "recommend",
        "query": "45万商务接待兼家用，想看MPV或大空间车型。",
        "args": {"budget_max": 450000, "preferred_type": "", "preferred_energy": "", "concerns": "空间,商务", "top_k": 5},
    },
    {
        "id": "p14_range_extender",
        "intent": "recommend",
        "query": "25万左右增程SUV，偶尔长途，关注续航。",
        "args": {"budget_max": 250000, "preferred_type": "SUV", "preferred_energy": "增程", "concerns": "续航", "top_k": 5},
    },
    {
        "id": "p15_luxury_ev_suv",
        "intent": "recommend",
        "query": "60万以内豪华纯电SUV，关注品牌和安全。",
        "args": {"budget_max": 600000, "preferred_type": "SUV", "preferred_energy": "纯电", "concerns": "品牌,安全", "top_k": 5},
    },
    {
        "id": "p16_city_commute_sedan",
        "intent": "recommend",
        "query": "18万以内女生通勤代步，想要纯电轿车。",
        "args": {"budget_max": 180000, "preferred_type": "轿车", "preferred_energy": "纯电", "concerns": "安全,好开", "top_k": 5},
    },
    {
        "id": "p17_large_family",
        "intent": "recommend",
        "query": "二胎家庭35万以内大空间SUV，关注后备箱和座位。",
        "args": {"budget_max": 350000, "preferred_type": "SUV", "preferred_energy": "", "concerns": "空间,后备箱", "top_k": 5},
    },
    {
        "id": "p18_long_trip",
        "intent": "recommend",
        "query": "经常跨城出差，28万以内想要补能稳的SUV。",
        "args": {"budget_max": 280000, "preferred_type": "SUV", "preferred_energy": "增程", "concerns": "长途,补能", "top_k": 5},
    },
    {
        "id": "p19_social_image",
        "intent": "recommend",
        "query": "40万以内新能源车，关注社交形象和品牌。",
        "args": {"budget_max": 400000, "preferred_type": "", "preferred_energy": "", "concerns": "社交形象,品牌", "top_k": 5},
    },
    {
        "id": "p20_customer_service",
        "intent": "sales",
        "query": "客户问没有家充还能不能买新能源SUV，怎么回答？",
        "args": {"budget_max": 260000, "preferred_type": "SUV", "preferred_energy": "", "concerns": "补能,无家充", "top_k": 5},
    },
]


def _tool_record(case_id: str, query: str, tool_content: str, answer: str) -> dict[str, Any]:
    return {
        "id": case_id,
        "messages": [
            {"role": "user", "content": query},
            {"role": "tool", "tool_call_id": f"{case_id}-search", "content": tool_content},
            {"role": "assistant", "content": answer},
        ],
    }


def _grounded_answer(rows: list[dict[str, Any]]) -> str:
    selected = rows[:3]
    lines = []
    for row in selected:
        specs = row["specs"]
        lines.append(
            f"- {row['full_name']}：{row['energy']}，价格{specs['price_range']}，"
            f"CLTC续航{specs['cltc_range']}，电池{specs['battery']}，"
            f"快充{specs['fast_charge']}，轴距{specs['wheelbase']}，"
            f"后备箱{specs['trunk_volume']}，{specs['seats']}，"
            f"匹配评分{specs['match_score']}。"
        )
    return (
        "基于工具返回的车型规格，候选对比如下：\n"
        + "\n".join(lines)
        + "\n工具未返回官方油耗、电耗和保修政策，相关信息需以品牌官方实时信息核验。"
    )


def _bad_answer(rows: list[dict[str, Any]]) -> str:
    top = rows[0]
    specs = top["specs"]
    return (
        f"{top['full_name']} 价格{specs['price_range']}，CLTC续航{specs['cltc_range']}，"
        "但我额外声称它的轴距9999mm。"
    )


def run_pressure_test() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for case in PRESSURE_CASES:
        tool_content = agent_graph.search_and_rank_vehicles.invoke(case["args"])
        rows = json.loads(tool_content)
        good_record = _tool_record(case["id"], case["query"], tool_content, _grounded_answer(rows))
        bad_record = _tool_record(case["id"], case["query"], tool_content, _bad_answer(rows))
        good_errors = audit_answer_grounding(good_record)
        bad_errors = audit_answer_grounding(bad_record)
        good_answer = good_record["messages"][-1]["content"]
        results.append(
            {
                "id": case["id"],
                "intent": case["intent"],
                "query": case["query"],
                "candidate_count": len(rows),
                "top_models": [row["full_name"] for row in rows[:3]],
                "good_errors": good_errors,
                "bad_errors": bad_errors,
                "specific_claim_count": len(good_answer.split("，")) - 1,
                "deferral_count": good_answer.count("官方实时信息核验"),
            }
        )
    false_positive = [item for item in results if item["good_errors"]]
    false_negative = [item for item in results if not item["bad_errors"]]
    summary = {
        "case_count": len(results),
        "false_positive_count": len(false_positive),
        "false_negative_count": len(false_negative),
        "pass": not false_positive and not false_negative,
        "results": results,
    }
    (OUT_DIR / "grounding_pressure_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Grounding 泛化压测报告",
        "",
        f"- case 数：{summary['case_count']}",
        f"- 误杀数：{summary['false_positive_count']}",
        f"- 漏网数：{summary['false_negative_count']}",
        f"- 结论：{'PASS' if summary['pass'] else 'FAIL'}",
        "",
        "| ID | intent | 候选数 | Top3 | good_errors | bad_errors |",
        "|---|---|---:|---|---|---|",
    ]
    for item in results:
        lines.append(
            "| {id} | {intent} | {candidate_count} | {top_models} | {good_errors} | {bad_errors} |".format(
                id=item["id"],
                intent=item["intent"],
                candidate_count=item["candidate_count"],
                top_models="、".join(item["top_models"]),
                good_errors="; ".join(item["good_errors"]) or "[]",
                bad_errors="; ".join(item["bad_errors"]) or "[]",
            )
        )
    (OUT_DIR / "grounding_pressure_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    print(json.dumps(run_pressure_test(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
