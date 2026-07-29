import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.check_model_runtime import (
    _validate_model,
    collect_runtime_facts,
    evaluate_runtime_gates,
)


def _write_model(model_path: Path, missing_shard: bool = False) -> None:
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"]}),
        encoding="utf-8",
    )
    (model_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"}),
        encoding="utf-8",
    )
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (model_path / "model-00001-of-00002.safetensors").write_bytes(b"weights-1")
    if not missing_shard:
        (model_path / "model-00002-of-00002.safetensors").write_bytes(b"weights-2")


def _successful_nvidia_smi(command, **_kwargs):
    if any(item.startswith("--query-gpu=") for item in command):
        return SimpleNamespace(
            returncode=0,
            stdout=" Tesla V100-SXM2-32GB , 470.129.06 , 32510 MiB \n",
            stderr="",
        )
    return SimpleNamespace(
        returncode=0,
        stdout="Driver Version: 470.129.06   CUDA Version: 11.4",
        stderr="",
    )


def _complete_runtime_evidence() -> dict:
    return {
        "isolated_environment": True,
        "torch_cuda_available": True,
        "torch_gpu_name": "Tesla V100-SXM2-32GB",
        "torch_cuda_version": "11.3",
        "bitsandbytes_4bit_cuda_smoke": True,
        "vllm_model_load_smoke": True,
        "vllm_tool_call_smoke": True,
        "service_env_clean": True,
        "vllm_version": "0.6.3",
        "torch_version": "2.4.1+cu113",
        "bitsandbytes_version": "0.44.1",
    }


def _verified_v100_facts() -> dict:
    return {
        "gpu_name": "Tesla V100-SXM2-32GB",
        "driver_version": "470.129.06",
        "cuda_driver_version": "11.4",
        "gpu_probe": {"status": "ok", "error": None, "errors": []},
        "runtime_evidence": _complete_runtime_evidence(),
        "model_validation": {"status": "valid"},
    }


def _set_single_shard_reference(model_path: Path, reference) -> None:
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": reference}}),
        encoding="utf-8",
    )


def test_v100_requires_fp16_and_separate_vllm_environment():
    result = evaluate_runtime_gates(
        {
            "gpu_name": "Tesla V100-SXM2-32GB",
            "driver_version": "470.129.06",
            "cuda_driver_version": "11.4",
            "model_path_exists": True,
            "model_size_bytes": 29 * 1024**3,
        }
    )

    assert result["compute_capability"] == "7.0"
    assert result["hardware_support_status"] == "supported_fp16"
    assert result["training_compute_dtype"] == "float16"
    assert result["bf16_allowed"] is False
    assert result["service_env_can_install_vllm"] is False
    assert result["vllm_status"] == "requires_compatible_isolated_runtime"
    assert result["binary_runtime_compatibility"] == "unverified"
    assert result["runtime_gate"] == "blocked_until_verified"


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (FileNotFoundError("nvidia-smi"), "command_not_found"),
        (
            subprocess.CalledProcessError(
                9, ["nvidia-smi"], stderr="driver communication failed"
            ),
            "command_failed",
        ),
    ],
)
def test_collect_runtime_facts_returns_structured_gpu_error(
    monkeypatch, tmp_path, error, expected_code
):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing-model"))

    def fail_nvidia_smi(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fail_nvidia_smi)

    facts = collect_runtime_facts()

    assert facts["gpu_probe"]["status"] == "error"
    assert facts["gpu_probe"]["error"]["code"] == expected_code
    assert facts["gpu_name"] is None
    assert facts["driver_version"] is None
    assert facts["cuda_driver_version"] is None


def test_collect_runtime_facts_accepts_csv_spaces_units_and_complete_model(
    monkeypatch, tmp_path
):
    model_path = tmp_path / "model"
    _write_model(model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setattr(subprocess, "run", _successful_nvidia_smi)

    facts = collect_runtime_facts()

    assert facts["gpu_probe"]["status"] == "ok"
    assert facts["gpu_name"] == "Tesla V100-SXM2-32GB"
    assert facts["driver_version"] == "470.129.06"
    assert facts["cuda_driver_version"] == "11.4"
    assert facts["gpu_memory_total_mib"] == 32510
    assert facts["model_validation"]["status"] == "valid"
    assert facts["model_validation"]["architecture"] == "Qwen2ForCausalLM"
    assert facts["model_validation"]["referenced_shards"] == 2
    assert facts["model_validation"]["missing_shards"] == []
    assert facts["model_validation"]["empty_shards"] == []


def test_collect_runtime_facts_rejects_missing_index_shard(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    _write_model(model_path, missing_shard=True)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setattr(subprocess, "run", _successful_nvidia_smi)

    facts = collect_runtime_facts()

    assert facts["model_validation"]["status"] == "invalid"
    assert facts["model_validation"]["missing_shards"] == [
        "model-00002-of-00002.safetensors"
    ]


def test_driver_470_and_cuda_11_4_do_not_claim_vllm_ready():
    result = evaluate_runtime_gates(
        {
            "gpu_name": "Tesla V100-SXM2-32GB",
            "driver_version": "470.129.06",
            "cuda_driver_version": "11.4",
            "model_validation": {"status": "valid"},
        }
    )

    assert result["hardware_support_status"] == "supported_fp16"
    assert result["binary_runtime_compatibility"] == "unverified"
    assert result["vllm_status"] == "requires_compatible_isolated_runtime"
    assert result["runtime_gate"] == "blocked_until_verified"


def test_legacy_binary_runtime_verified_boolean_cannot_release_gate():
    result = evaluate_runtime_gates(
        {
            "gpu_name": "Tesla V100-SXM2-32GB",
            "driver_version": "470.129.06",
            "cuda_driver_version": "11.4",
            "binary_runtime_verified": True,
        }
    )

    assert result["binary_runtime_compatibility"] == "unverified"
    assert result["runtime_gate"] == "blocked_until_verified"


def test_partial_runtime_evidence_cannot_release_gate():
    evidence = _complete_runtime_evidence()
    del evidence["vllm_tool_call_smoke"]

    result = evaluate_runtime_gates(
        {
            "gpu_name": "Tesla V100-SXM2-32GB",
            "driver_version": "470.129.06",
            "cuda_driver_version": "11.4",
            "runtime_evidence": evidence,
        }
    )

    assert result["binary_runtime_compatibility"] == "unverified"
    assert result["runtime_gate"] == "blocked_until_verified"
    assert "vllm_tool_call_smoke" in result["missing_runtime_evidence"]


def test_complete_runtime_evidence_releases_gate():
    result = evaluate_runtime_gates(_verified_v100_facts())

    assert result["binary_runtime_compatibility"] == "verified"
    assert result["runtime_gate"] == "ready"
    assert result["vllm_status"] == "verified_isolated_runtime"
    assert result["overall_gate"] == "ready"
    assert result["missing_runtime_evidence"] == []


def test_torch_cuda_version_is_distinct_from_driver_reported_cuda():
    evidence = _complete_runtime_evidence()
    evidence["torch_cuda_version"] = "12.1"

    result = evaluate_runtime_gates(
        {
            "gpu_name": "Tesla V100-SXM2-32GB",
            "driver_version": "470.129.06",
            "cuda_driver_version": "11.4",
            "runtime_evidence": evidence,
        }
    )

    assert result["driver_reported_cuda_version"] == "11.4"
    assert result["torch_cuda_version"] == "12.1"
    assert result["binary_runtime_compatibility"] == "incompatible"
    assert result["runtime_gate"] == "blocked_until_verified"


def test_collect_runtime_facts_rejects_unhashable_shard_reference(
    monkeypatch, tmp_path
):
    model_path = tmp_path / "model"
    _write_model(model_path)
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["layer.1"] = ["not", "a", "filename"]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setattr(subprocess, "run", _successful_nvidia_smi)

    facts = collect_runtime_facts()

    assert facts["model_validation"]["status"] == "invalid"
    assert {
        "file": "model.safetensors.index.json",
        "code": "invalid_shard_reference",
    } in facts["model_validation"]["errors"]


@pytest.mark.parametrize(
    "reference",
    [
        "/tmp/outside.safetensors",
        "../outside.safetensors",
        "nested/model.safetensors",
        "model.bin",
        "",
        "   ",
        " model.safetensors",
        "model.safetensors ",
        "x" * 244 + ".safetensors",
        None,
        7,
        ["model.safetensors"],
        {"file": "model.safetensors"},
    ],
    ids=[
        "absolute",
        "parent-traversal",
        "nested-path",
        "wrong-extension",
        "empty",
        "whitespace-only",
        "leading-whitespace",
        "trailing-whitespace",
        "overlong",
        "null",
        "integer",
        "list",
        "object",
    ],
)
def test_model_index_rejects_unsafe_shard_references(tmp_path, reference):
    model_path = tmp_path / "model"
    _write_model(model_path)
    _set_single_shard_reference(model_path, reference)

    validation = _validate_model(model_path)

    assert validation["status"] == "invalid"
    assert validation["referenced_shards"] == 0
    assert any(
        error["code"] == "invalid_shard_reference"
        for error in validation["errors"]
    )


@pytest.mark.parametrize("reference_kind", ["absolute", "parent-traversal"])
def test_model_index_never_stats_outside_model_directory(
    monkeypatch, tmp_path, reference_kind
):
    model_path = tmp_path / "model"
    _write_model(model_path)
    outside_path = tmp_path / "outside.safetensors"
    outside_path.write_bytes(b"outside-weights")
    reference = (
        str(outside_path)
        if reference_kind == "absolute"
        else "../outside.safetensors"
    )
    _set_single_shard_reference(model_path, reference)
    referenced_path = model_path / reference
    original_stat = Path.stat

    def guarded_stat(path, *args, **kwargs):
        if path == referenced_path:
            raise AssertionError("validator attempted to stat an outside path")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)

    validation = _validate_model(model_path)

    assert validation["status"] == "invalid"
    assert any(
        error["code"] == "invalid_shard_reference"
        for error in validation["errors"]
    )


def test_model_index_rejects_symlinked_shard_without_following_it(tmp_path):
    model_path = tmp_path / "model"
    _write_model(model_path)
    outside_path = tmp_path / "outside.safetensors"
    outside_path.write_bytes(b"outside-weights")
    symlink_name = "model-symlink.safetensors"
    (model_path / symlink_name).symlink_to(outside_path)
    _set_single_shard_reference(model_path, symlink_name)

    validation = _validate_model(model_path)

    assert validation["status"] == "invalid"
    assert any(
        error["code"] == "invalid_referenced_shard"
        for error in validation["errors"]
    )


def test_model_index_returns_structured_error_when_shard_stat_fails(
    monkeypatch, tmp_path
):
    model_path = tmp_path / "model"
    _write_model(model_path)
    shard_path = model_path / "model-00001-of-00002.safetensors"
    _set_single_shard_reference(model_path, shard_path.name)
    original_stat = Path.stat

    def fail_shard_stat(path, *args, **kwargs):
        if path == shard_path:
            raise PermissionError("shard stat denied")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_shard_stat)

    validation = _validate_model(model_path)

    assert validation["status"] == "invalid"
    assert any(
        error["code"] == "shard_stat_error"
        and "shard stat denied" in error["detail"]
        for error in validation["errors"]
    )


@pytest.mark.parametrize("probe", [None, {}, {"status": "error"}])
def test_runtime_verification_requires_successful_gpu_probe(probe):
    facts = _verified_v100_facts()
    if probe is None:
        del facts["gpu_probe"]
    else:
        facts["gpu_probe"] = probe

    result = evaluate_runtime_gates(facts)

    assert result["binary_runtime_compatibility"] != "verified"
    assert result["runtime_gate"] == "blocked_until_verified"
    assert result["overall_gate"] == "blocked_until_verified"
    assert result["vllm_status"] != "verified_isolated_runtime"


@pytest.mark.parametrize(
    ("probed_gpu", "runtime_gpu"),
    [
        ("NVIDIA A100-SXM4-40GB", "Tesla V100-SXM2-32GB"),
        ("Tesla V100-SXM2-32GB", "NVIDIA A100-SXM4-40GB"),
        ("Tesla V100-SXM2-32GB", "V100"),
    ],
)
def test_runtime_verification_requires_strict_casefold_gpu_name_match(
    probed_gpu, runtime_gpu
):
    facts = _verified_v100_facts()
    facts["gpu_name"] = probed_gpu
    facts["runtime_evidence"]["torch_gpu_name"] = runtime_gpu

    result = evaluate_runtime_gates(facts)

    assert result["binary_runtime_compatibility"] != "verified"
    assert result["overall_gate"] == "blocked_until_verified"


def test_runtime_verification_accepts_casefold_equal_gpu_names():
    facts = _verified_v100_facts()
    facts["runtime_evidence"]["torch_gpu_name"] = "TESLA V100-SXM2-32GB"

    result = evaluate_runtime_gates(facts)

    assert result["binary_runtime_compatibility"] == "verified"
    assert result["overall_gate"] == "ready"


@pytest.mark.parametrize(
    ("field", "malformed_version"),
    [
        ("driver_version", "driver-470"),
        ("driver_version", "470.bad"),
        ("torch_cuda_version", "cuda-11.3"),
        ("torch_cuda_version", "11.x"),
    ],
)
def test_runtime_verification_rejects_malformed_versions(
    field, malformed_version
):
    facts = _verified_v100_facts()
    if field == "torch_cuda_version":
        facts["runtime_evidence"][field] = malformed_version
    else:
        facts[field] = malformed_version

    result = evaluate_runtime_gates(facts)

    assert result["binary_runtime_compatibility"] != "verified"
    assert result["runtime_gate"] == "blocked_until_verified"
    assert result["overall_gate"] == "blocked_until_verified"


def test_overall_gate_blocks_when_model_is_invalid_after_runtime_verified():
    facts = _verified_v100_facts()
    facts["model_validation"] = {"status": "invalid"}

    result = evaluate_runtime_gates(facts)

    assert result["binary_runtime_compatibility"] == "verified"
    assert result["runtime_gate"] == "ready"
    assert result["model_gate"] == "blocked"
    assert result["overall_gate"] == "blocked_until_verified"
    assert result["vllm_status"] == "runtime_evidence_verified_overall_blocked"
