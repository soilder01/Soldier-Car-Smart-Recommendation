import inspect
from copy import deepcopy

from app.services import agent_graph
from data_synth.tool_schemas import TOOL_SCHEMAS, build_tool_schemas


JSON_TYPES = {
    str: "string",
    int: "integer",
}
JSON_ARRAY_TYPES = {
    list[str]: {"type": "array", "items": {"type": "string"}},
}


def test_tool_schemas_follow_openai_function_format_and_fixed_order():
    schemas = build_tool_schemas()

    assert [schema["function"]["name"] for schema in schemas] == [
        tool.name for tool in agent_graph.TOOLS
    ]
    for schema in schemas:
        assert set(schema) == {"type", "function"}
        assert schema["type"] == "function"

        function_schema = schema["function"]
        assert set(function_schema) == {"name", "description", "parameters"}
        assert isinstance(function_schema["description"], str)
        assert function_schema["description"]

        parameters = function_schema["parameters"]
        assert set(parameters) == {
            "type",
            "properties",
            "required",
            "additionalProperties",
        }
        assert parameters["type"] == "object"
        assert isinstance(parameters["properties"], dict)
        assert isinstance(parameters["required"], list)
        assert parameters["additionalProperties"] is False


def test_tool_schema_parameters_dynamically_match_real_signatures():
    schemas = {
        schema["function"]["name"]: schema["function"]["parameters"]
        for schema in TOOL_SCHEMAS
    }
    real_tools = {tool.name: tool for tool in agent_graph.TOOLS}

    assert set(schemas) == set(real_tools)
    for tool_name, real_tool in real_tools.items():
        signature = inspect.signature(real_tool.func)
        real_parameters = list(signature.parameters.values())
        schema_parameters = schemas[tool_name]
        properties = schema_parameters["properties"]

        assert list(properties) == [parameter.name for parameter in real_parameters]
        assert set(properties) == set(signature.parameters)
        assert schema_parameters["required"] == [
            parameter.name
            for parameter in real_parameters
            if parameter.default is inspect.Parameter.empty
        ]

        for parameter in real_parameters:
            assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            if parameter.annotation in JSON_TYPES:
                expected_property = {"type": JSON_TYPES[parameter.annotation]}
            else:
                assert parameter.annotation in JSON_ARRAY_TYPES
                expected_property = deepcopy(JSON_ARRAY_TYPES[parameter.annotation])
            if parameter.default is not inspect.Parameter.empty:
                expected_property["default"] = parameter.default
            assert properties[parameter.name] == expected_property


def test_build_tool_schemas_returns_a_deep_copy():
    schemas = build_tool_schemas()

    assert schemas == TOOL_SCHEMAS
    assert schemas is not TOOL_SCHEMAS
    assert schemas[0] is not TOOL_SCHEMAS[0]
    assert schemas[0]["function"] is not TOOL_SCHEMAS[0]["function"]
    assert (
        schemas[0]["function"]["parameters"]["properties"]
        is not TOOL_SCHEMAS[0]["function"]["parameters"]["properties"]
    )

    schemas[0]["function"]["parameters"]["properties"]["query"]["type"] = "integer"
    schemas.append({"type": "mutated"})

    assert len(TOOL_SCHEMAS) == len(agent_graph.TOOLS)
    assert (
        TOOL_SCHEMAS[0]["function"]["parameters"]["properties"]["query"]["type"]
        == "string"
    )
