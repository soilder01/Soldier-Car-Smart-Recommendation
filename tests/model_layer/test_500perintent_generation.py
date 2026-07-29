import json

from app.database import list_vehicles
from app.services import agent_graph
from data_synth.generate_20perintent_sft import _extract_compare_lookup
from data_synth.generate_500perintent_sft import (
    _compare_pair_recalled,
    _gate_reason,
    _paths,
    _write_report,
)


def test_compare_query_catalog_names_round_trip_through_named_lookup():
    catalog = list_vehicles()

    unresolved = [
        f"{vehicle['brand']}{vehicle['model']}"
        for vehicle in catalog
        if (
            resolved := agent_graph._resolve_named_vehicle(
                f"{vehicle['brand']}{vehicle['model']}",
                catalog,
            )
        ) is None
        or resolved["id"] != vehicle["id"]
    ]

    assert unresolved == []


def test_compare_recall_counts_catalog_lookup_missing_as_a_failure():
    record = {
        "expected_named_models": ["特斯拉Model Y", "享界S9增程"],
    }
    lookup = {
        "requested_model_names": ["特斯拉Model Y", "享界S9增程"],
        "resolved_model_names": ["特斯拉Model Y"],
        "missing_model_names": ["享界S9增程"],
        "named_vehicle_missing": True,
    }

    assert _compare_pair_recalled(record, lookup) is False


def test_compare_report_keeps_generated_catalog_missing_in_recall_denominator(tmp_path):
    resolved_lookup = {
        "requested_model_names": ["特斯拉Model Y", "享界S9"],
        "resolved_model_names": ["特斯拉Model Y", "享界S9"],
        "missing_model_names": [],
        "named_vehicle_missing": False,
    }
    missing_lookup = {
        "requested_model_names": ["特斯拉Model Y", "享界S9增程"],
        "resolved_model_names": ["特斯拉Model Y"],
        "missing_model_names": ["享界S9增程"],
        "named_vehicle_missing": True,
    }
    records = {
        "compare-1": {
            "expected_named_models": ["特斯拉Model Y", "享界S9"],
            "named_vehicle_lookup": resolved_lookup,
            "tool_call_rounds": 1,
        },
        "compare-2": {
            "expected_named_models": ["特斯拉Model Y", "享界S9增程"],
            "named_vehicle_lookup": missing_lookup,
            "tool_call_rounds": 1,
        },
    }
    paths = {
        "report": tmp_path / "report.md",
        "report_json": tmp_path / "report.json",
    }

    summary = _write_report(
        intent="compare",
        target=2,
        records=records,
        failures=[],
        gate_reason="",
        paths=paths,
    )

    assert summary["compare_in_catalog_rows"] == 2
    assert summary["compare_both_recalled"] == 1
    assert summary["compare_both_recall_rate"] == 50.0


def test_compare_report_counts_explicit_lookup_mismatches_in_recall_denominator(
    tmp_path,
):
    lookup = {
        "requested_model_names": ["特斯拉Model Y", "享界S9"],
        "resolved_model_names": ["特斯拉Model Y", "享界S9"],
        "missing_model_names": [],
        "named_vehicle_missing": False,
    }
    records = {
        "compare-1": {
            "expected_named_models": ["特斯拉Model Y", "享界S9"],
            "named_vehicle_lookup": lookup,
            "tool_call_rounds": 1,
        },
    }
    failures = [
        {
            "id": "compare-2",
            "error_type": "audit",
            "decision_audit": [
                "compare named lookup did not recall the generated catalog pair"
            ],
        },
    ]
    paths = {
        "report": tmp_path / "report.md",
        "report_json": tmp_path / "report.json",
    }

    summary = _write_report(
        intent="compare",
        target=2,
        records=records,
        failures=failures,
        gate_reason="",
        paths=paths,
    )

    assert summary["compare_in_catalog_rows"] == 2
    assert summary["compare_both_recalled"] == 1
    assert summary["compare_both_recall_rate"] == 50.0
    assert summary["compare_lookup_mismatches"] == 1


def test_compare_gate_stops_on_first_explicit_lookup_mismatch():
    assert _gate_reason(
        "compare",
        {
            "attempted": 1,
            "truncation_rate": 0.0,
            "compare_lookup_mismatches": 1,
            "accepted": 0,
        },
    ) == "compare出现点名车型错配：1条"


def test_labeled_500_outputs_do_not_overwrite_prior_run():
    paths = _paths("compare", "named_lookup_v2")

    assert paths["dataset"].name == (
        "teacher_decision_500perintent_compare_named_lookup_v2_sft.jsonl"
    )


def test_compare_lookup_metadata_is_extractable_for_20_intent_records():
    record = {
        "messages": [
            {
                "role": "tool",
                "tool_call_id": "lookup",
                "content": json.dumps(
                    {
                        "named_vehicle_lookup": {
                            "requested_model_names": ["享界S9", "享界S9增程"],
                            "resolved_model_names": ["享界S9", "享界S9增程"],
                            "missing_model_names": [],
                            "named_vehicle_missing": False,
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }

    assert _extract_compare_lookup(record) == {
        "requested_model_names": ["享界S9", "享界S9增程"],
        "resolved_model_names": ["享界S9", "享界S9增程"],
        "missing_model_names": [],
        "named_vehicle_missing": False,
    }
