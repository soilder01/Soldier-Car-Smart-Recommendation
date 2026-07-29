from scripts import run_sft_heldout_eval as heldout


def test_failure_rows_contains_only_failed_cases():
    summary = {
        "cases": [
            {
                "id": "pass",
                "intent": "sales",
                "passed": True,
                "failure_reasons": [],
            },
            {
                "id": "fail",
                "intent": "recommend",
                "passed": False,
                "failure_reasons": ["format"],
            },
        ]
    }

    assert heldout.failure_rows(summary) == [
        {
            "id": "fail",
            "intent": "recommend",
            "failure_reasons": ["format"],
        }
    ]


def test_baseline_comparison_uses_frozen_six_of_forty_anchor():
    comparison = heldout.baseline_comparison(
        {
            "total_score": {
                "numerator": 22,
                "denominator": 40,
                "percentage": 55.0,
            }
        }
    )

    assert comparison == {
        "baseline": {
            "numerator": 6,
            "denominator": 40,
            "percentage": 15.0,
        },
        "sft": {
            "numerator": 22,
            "denominator": 40,
            "percentage": 55.0,
        },
        "absolute_percentage_point_change": 40.0,
        "claim_scope": (
            "tool-call contract and structured-protocol compliance; "
            "not a standalone decision-quality claim"
        ),
    }
