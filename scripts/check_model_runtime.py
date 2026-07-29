#!/usr/bin/env python3
import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "Qwen2.5-7B-Instruct"
MAX_SHARD_REFERENCE_BYTES = 255


def _command_error(command: list[str], error: BaseException) -> dict[str, Any]:
    if isinstance(error, FileNotFoundError):
        code = "command_not_found"
        detail = str(error)
        returncode = None
    elif isinstance(error, subprocess.TimeoutExpired):
        code = "command_timeout"
        detail = str(error)
        returncode = None
    elif isinstance(error, subprocess.CalledProcessError):
        code = "command_failed"
        detail = (error.stderr or error.stdout or str(error)).strip()
        returncode = error.returncode
    else:
        code = "command_os_error"
        detail = str(error)
        returncode = None
    return {
        "code": code,
        "command": command,
        "returncode": returncode,
        "detail": detail,
    }


def _run_nvidia_smi(command: list[str]) -> tuple[str | None, dict[str, Any] | None]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        return completed.stdout, None
    except (
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        return None, _command_error(command, error)


def _parse_gpu_csv(output: str) -> dict[str, Any]:
    first_line = next((line for line in output.splitlines() if line.strip()), "")
    columns = [column.strip() for column in first_line.split(",")]
    if len(columns) != 3:
        raise ValueError(f"expected 3 CSV columns, got {len(columns)}")
    memory_match = re.search(r"\d+(?:\.\d+)?", columns[2])
    if memory_match is None:
        raise ValueError("GPU memory column has no numeric value")
    return {
        "gpu_name": columns[0],
        "driver_version": columns[1],
        "gpu_memory_total_mib": int(float(memory_match.group(0))),
    }


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            child_stat = child.stat(follow_symlinks=False)
        except OSError:
            continue
        if stat.S_ISREG(child_stat.st_mode):
            total += child_stat.st_size
    return total


def _read_json(path: Path, errors: list[dict[str, str]]) -> Any:
    if not path.is_file():
        errors.append({"file": path.name, "code": "missing"})
        return None
    if path.stat().st_size == 0:
        errors.append({"file": path.name, "code": "empty"})
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        errors.append(
            {"file": path.name, "code": "invalid_json", "detail": str(error)}
        )
        return None


def _is_simple_shard_reference(reference: Any) -> bool:
    if not isinstance(reference, str):
        return False
    if not reference or reference != reference.strip():
        return False
    if any(character in reference for character in ("/", "\\", "\x00", ":")):
        return False
    if any(ord(character) < 32 for character in reference):
        return False
    if not reference.endswith(".safetensors"):
        return False
    try:
        return len(reference.encode("utf-8")) <= MAX_SHARD_REFERENCE_BYTES
    except UnicodeError:
        return False


def _validate_model(model_path: Path) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    config = _read_json(model_path / "config.json", errors)
    tokenizer_config = _read_json(model_path / "tokenizer_config.json", errors)
    index = _read_json(model_path / "model.safetensors.index.json", errors)

    architectures = config.get("architectures", []) if isinstance(config, dict) else []
    architecture = architectures[0] if architectures else None
    if architecture != "Qwen2ForCausalLM" and config is not None:
        errors.append(
            {
                "file": "config.json",
                "code": "unexpected_architecture",
                "detail": str(architecture),
            }
        )
    if not isinstance(tokenizer_config, dict) and tokenizer_config is not None:
        errors.append(
            {"file": "tokenizer_config.json", "code": "expected_json_object"}
        )

    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        if index is not None:
            errors.append(
                {
                    "file": "model.safetensors.index.json",
                    "code": "invalid_weight_map",
                }
            )
        shard_names: list[str] = []
    else:
        valid_shard_names = [
            name for name in weight_map.values() if _is_simple_shard_reference(name)
        ]
        shard_names = sorted(set(valid_shard_names))
        if len(valid_shard_names) != len(weight_map):
            errors.append(
                {
                    "file": "model.safetensors.index.json",
                    "code": "invalid_shard_reference",
                }
            )

    missing_shards = []
    empty_shards = []
    for shard_name in shard_names:
        shard = model_path / shard_name
        try:
            shard_stat = shard.stat(follow_symlinks=False)
        except FileNotFoundError:
            missing_shards.append(shard_name)
            continue
        except OSError as error:
            errors.append(
                {
                    "file": shard_name,
                    "code": "shard_stat_error",
                    "detail": str(error),
                }
            )
            continue
        if not stat.S_ISREG(shard_stat.st_mode):
            errors.append(
                {
                    "file": shard_name,
                    "code": "invalid_referenced_shard",
                    "detail": "referenced shard must be a regular file",
                }
            )
        elif shard_stat.st_size <= 0:
            empty_shards.append(shard_name)

    if missing_shards:
        errors.append(
            {
                "file": "model.safetensors.index.json",
                "code": "missing_referenced_shards",
            }
        )
    if empty_shards:
        errors.append(
            {
                "file": "model.safetensors.index.json",
                "code": "empty_referenced_shards",
            }
        )
    return {
        "status": "valid" if not errors else "invalid",
        "architecture": architecture,
        "required_metadata_files": {
            name: (model_path / name).is_file()
            and (model_path / name).stat().st_size > 0
            for name in (
                "config.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
            )
        },
        "referenced_shards": len(shard_names),
        "missing_shards": missing_shards,
        "empty_shards": empty_shards,
        "errors": errors,
    }


def collect_runtime_facts() -> dict[str, Any]:
    model_path = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
    facts: dict[str, Any] = {
        "gpu_name": None,
        "driver_version": None,
        "cuda_driver_version": None,
        "gpu_memory_total_mib": None,
        "model_path": str(model_path),
        "model_path_exists": model_path.is_dir(),
        "model_size_bytes": _directory_size(model_path) if model_path.is_dir() else 0,
    }

    query = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    csv_output, csv_error = _run_nvidia_smi(query)
    errors = [csv_error] if csv_error else []
    if csv_output is not None:
        try:
            facts.update(_parse_gpu_csv(csv_output))
        except ValueError as error:
            errors.append(
                {
                    "code": "invalid_csv",
                    "command": query,
                    "returncode": 0,
                    "detail": str(error),
                }
            )

    summary_output, summary_error = _run_nvidia_smi(["nvidia-smi"])
    if summary_error:
        errors.append(summary_error)
    if summary_output is not None:
        cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", summary_output)
        if cuda_match:
            facts["cuda_driver_version"] = cuda_match.group(1)
        else:
            errors.append(
                {
                    "code": "cuda_version_not_found",
                    "command": ["nvidia-smi"],
                    "returncode": 0,
                    "detail": "CUDA Version field was not present",
                }
            )

    facts["gpu_probe"] = {
        "status": "ok" if not errors else "error",
        "error": errors[0] if errors else None,
        "errors": errors,
    }
    facts["model_validation"] = (
        _validate_model(model_path)
        if model_path.is_dir()
        else {
            "status": "invalid",
            "architecture": None,
            "required_metadata_files": {},
            "referenced_shards": 0,
            "missing_shards": [],
            "empty_shards": [],
            "errors": [{"file": str(model_path), "code": "model_path_missing"}],
        }
    )
    return facts


def _version_major(version: Any) -> int | None:
    if not isinstance(version, str):
        return None
    normalized = version.strip()
    if re.fullmatch(r"\d+(?:\.\d+)*", normalized) is None:
        return None
    return int(normalized.split(".", 1)[0])


RUNTIME_TRUE_EVIDENCE = (
    "isolated_environment",
    "torch_cuda_available",
    "bitsandbytes_4bit_cuda_smoke",
    "vllm_model_load_smoke",
    "vllm_tool_call_smoke",
    "service_env_clean",
)
RUNTIME_TEXT_EVIDENCE = (
    "torch_gpu_name",
    "torch_cuda_version",
    "vllm_version",
    "torch_version",
    "bitsandbytes_version",
)


def _missing_runtime_evidence(
    evidence: Any, expected_gpu_name: str
) -> list[str]:
    if not isinstance(evidence, dict):
        evidence = {}
    missing = [
        field for field in RUNTIME_TRUE_EVIDENCE if evidence.get(field) is not True
    ]
    missing.extend(
        field
        for field in RUNTIME_TEXT_EVIDENCE
        if not isinstance(evidence.get(field), str) or not evidence[field].strip()
    )

    torch_gpu_name = str(evidence.get("torch_gpu_name") or "").strip()
    gpu_matches = (
        bool(torch_gpu_name)
        and bool(expected_gpu_name)
        and torch_gpu_name.casefold() == expected_gpu_name.casefold()
    )
    if torch_gpu_name and not gpu_matches and "torch_gpu_name" not in missing:
        missing.append("torch_gpu_name")
    return sorted(missing)


def evaluate_runtime_gates(facts: dict[str, Any]) -> dict[str, Any]:
    gpu_name = str(facts.get("gpu_name") or "")
    compute_capability = facts.get("compute_capability")
    if "V100" in gpu_name.upper():
        compute_capability = "7.0"

    is_v100 = compute_capability == "7.0"
    hardware_status = "supported_fp16" if is_v100 else "requires_verification"
    runtime_evidence = facts.get("runtime_evidence")
    if not isinstance(runtime_evidence, dict):
        runtime_evidence = {}
    torch_cuda_version = runtime_evidence.get("torch_cuda_version")
    missing_runtime_evidence = _missing_runtime_evidence(
        runtime_evidence, gpu_name
    )
    driver_major = _version_major(facts.get("driver_version"))
    torch_cuda_major = _version_major(torch_cuda_version)
    gpu_probe = facts.get("gpu_probe")
    gpu_probe_ok = (
        isinstance(gpu_probe, dict) and gpu_probe.get("status") == "ok"
    )

    runtime_verification_blockers = [
        f"runtime_evidence.{field}" for field in missing_runtime_evidence
    ]
    if not gpu_probe_ok:
        runtime_verification_blockers.append("gpu_probe.status")
    if hardware_status != "supported_fp16":
        runtime_verification_blockers.append("hardware_support_status")
    if driver_major is None:
        runtime_verification_blockers.append("driver_version")
    if torch_cuda_major is None:
        runtime_verification_blockers.append(
            "runtime_evidence.torch_cuda_version"
        )
    runtime_verification_blockers = sorted(set(runtime_verification_blockers))

    binary_compatibility = "unverified"
    binary_reason = (
        "Complete isolated runtime evidence bound to the current GPU probe is "
        "required."
    )
    if (
        torch_cuda_major is not None
        and torch_cuda_major >= 12
        and driver_major is not None
        and driver_major < 525
    ):
        binary_compatibility = "incompatible"
        binary_reason = (
            "The reported PyTorch CUDA 12.x runtime requires a newer driver "
            "than the detected "
            f"{facts.get('driver_version')} driver."
        )
    elif not runtime_verification_blockers:
        binary_compatibility = "verified"
        binary_reason = (
            "All required isolated runtime smoke evidence matches the current "
            "supported GPU probe."
        )
    elif runtime_verification_blockers:
        binary_reason = (
            "Runtime verification is blocked by: "
            + ", ".join(runtime_verification_blockers)
            + "."
        )

    runtime_gate = (
        "ready" if binary_compatibility == "verified" else "blocked_until_verified"
    )
    model_validation = facts.get("model_validation")
    model_gate = (
        "ready"
        if isinstance(model_validation, dict)
        and model_validation.get("status") == "valid"
        else "blocked"
    )
    overall_gate = (
        "ready"
        if gpu_probe_ok
        and hardware_status == "supported_fp16"
        and binary_compatibility == "verified"
        and model_gate == "ready"
        else "blocked_until_verified"
    )
    if overall_gate == "ready":
        vllm_status = "verified_isolated_runtime"
    elif binary_compatibility == "verified":
        vllm_status = "runtime_evidence_verified_overall_blocked"
    else:
        vllm_status = "requires_compatible_isolated_runtime"

    return {
        "compute_capability": compute_capability,
        "hardware_support_status": hardware_status,
        "training_compute_dtype": "float16" if is_v100 else "requires_verification",
        "bf16_allowed": False if is_v100 else None,
        "separate_training_vllm_environment_required": True,
        "service_env_can_install_vllm": False,
        "service_env_can_install_training_cuda_packages": False,
        "driver_reported_cuda_version": facts.get("cuda_driver_version"),
        "torch_cuda_version": torch_cuda_version,
        "missing_runtime_evidence": missing_runtime_evidence,
        "runtime_verification_blockers": runtime_verification_blockers,
        "binary_runtime_compatibility": binary_compatibility,
        "binary_runtime_reason": binary_reason,
        "vllm_status": vllm_status,
        "runtime_gate": runtime_gate,
        "model_gate": model_gate,
        "overall_gate": overall_gate,
    }


def main() -> int:
    facts = collect_runtime_facts()
    print(
        json.dumps(
            {"facts": facts, "gates": evaluate_runtime_gates(facts)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
