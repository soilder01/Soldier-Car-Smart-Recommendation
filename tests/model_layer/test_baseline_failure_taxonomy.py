from scripts import audit_baseline_failures as taxonomy


def _output(
    *,
    terminal_parse_error=None,
    trajectory=None,
):
    return {
        "terminal_parse_error": terminal_parse_error,
        "trajectory": trajectory or [],
        "runner_errors": [],
    }


def _result(
    *,
    failures=None,
    valid_argument_calls=1,
    total_tool_calls=1,
):
    return {
        "failures": failures or [],
        "valid_argument_calls": valid_argument_calls,
        "total_tool_calls": total_tool_calls,
    }


def test_classify_failure_distinguishes_format_tool_argument_and_grounding():
    invalid_xml = taxonomy.classify_failure_case(
        _output(
            terminal_parse_error="protocol failure: terminal content is not JSON",
            trajectory=[
                {
                    "role": "assistant",
                    "content": "<tool_call>{bad-json}</tool_call>",
                }
            ],
        ),
        _result(),
        grounding_errors=[],
    )
    wrong_tool = taxonomy.classify_failure_case(
        _output(),
        _result(
            failures=[
                "tool_selection: mandatory tools are missing or out of order"
            ],
        ),
        grounding_errors=[],
    )
    bad_arguments = taxonomy.classify_failure_case(
        _output(),
        _result(valid_argument_calls=0, total_tool_calls=1),
        grounding_errors=[],
    )
    grounding = taxonomy.classify_failure_case(
        _output(),
        _result(),
        grounding_errors=["unsupported hard claim"],
    )

    assert invalid_xml["categories"] == ["format_parse_failure"]
    assert invalid_xml["format_subtype"] == "invalid_tool_call_xml"
    assert wrong_tool["categories"] == ["wrong_tool"]
    assert bad_arguments["categories"] == ["argument_error"]
    assert grounding["categories"] == ["grounding_failure"]


def test_taxonomy_preserves_multilabel_counts_and_exclusive_primary_reason():
    cases = [
        {"id": "a", "intent": "recommend"},
        {"id": "b", "intent": "compare"},
    ]
    outputs = {
        "a": _output(
            terminal_parse_error="protocol failure: invalid terminal JSON",
        ),
        "b": _output(),
    }
    results = {
        "a": _result(
            failures=[
                "response_schema_invalid: protocol failure",
                "tool_selection: mandatory tools are missing",
            ],
        ),
        "b": _result(
            failures=["tool_selection: mandatory tools are missing"],
            valid_argument_calls=0,
            total_tool_calls=1,
        ),
    }

    report = taxonomy.build_failure_taxonomy(
        cases=cases,
        outputs_by_id=outputs,
        results_by_id=results,
        grounding_by_id={"a": [], "b": ["unsupported claim"]},
        passed_ids=set(),
    )

    assert report["failed_cases"] == 2
    assert report["multi_label_counts"] == {
        "format_parse_failure": 1,
        "wrong_tool": 2,
        "argument_error": 1,
        "grounding_failure": 1,
        "other_contract_failure": 0,
    }
    assert report["primary_failure_counts"] == {
        "format_parse_failure": 1,
        "wrong_tool": 1,
        "argument_error": 0,
        "grounding_failure": 0,
        "other_contract_failure": 0,
    }
