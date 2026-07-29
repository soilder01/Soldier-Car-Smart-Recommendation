"""Five-element fail-closed entrypoint for one authorized GRPO run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FAILURE_PATH = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "grpo_formal_authorization_failure.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_authorization_sha(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.exists():
        raise FileNotFoundError(f"authorization SHA sidecar missing: {sidecar}")
    digest, filename = sidecar.read_text(
        encoding="ascii"
    ).strip().split(maxsplit=1)
    if filename != path.name:
        raise ValueError("authorization SHA sidecar filename mismatch")
    return digest


def validate_authorization(path: Path) -> dict[str, Any]:
    expected_sha256 = expected_authorization_sha(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("formal authorization manifest SHA mismatch")
    authorization = json.loads(path.read_text(encoding="utf-8"))
    gate = authorization["authorization_gate"]
    checks: dict[str, bool] = {}
    reward = gate["reward_fn"]
    checks["reward_fn_sha_match"] = (
        sha256_file(ROOT / reward["path"]) == reward["sha256"]
    )
    split = gate["split"]
    checks["split_three_file_sha_match"] = all(
        (
            sha256_file(ROOT / split["train_path"])
            == split["train_sha256"],
            sha256_file(ROOT / split["dev_path"]) == split["dev_sha256"],
            sha256_file(ROOT / split["manifest_path"])
            == split["manifest_sha256"],
        )
    )
    dev = gate["dev_authorization"]
    checks["dev_authorization_sha_match"] = (
        sha256_file(ROOT / dev["path"]) == dev["sha256"]
    )
    checks["venv_grpo_path_match"] = (
        Path(sys.prefix).resolve()
        == Path(gate["training_environment"]["path"]).resolve()
    )
    checks["local_files_only"] = gate["model_loading"] == {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if set(checks) != {
        "reward_fn_sha_match",
        "split_three_file_sha_match",
        "dev_authorization_sha_match",
        "venv_grpo_path_match",
        "local_files_only",
    }:
        raise RuntimeError("five-element authorization check set drift")
    if not all(checks.values()):
        raise RuntimeError(f"authorization gate failed: {checks}")
    return {
        "authorization": authorization,
        "checks": checks,
        "sha256": expected_sha256,
    }


def write_failure(error: Exception, authorization_path: Path) -> None:
    if FAILURE_PATH.exists():
        raise FileExistsError("authorization failure record already exists")
    FAILURE_PATH.write_text(
        json.dumps(
            {
                "status": "authorization_failed_before_model_load",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "authorization_path": str(authorization_path),
                "model_loaded": False,
                "optimizer_step": False,
                "checkpoint_saved": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if args.authorization is None:
        raise SystemExit(
            "GRPO training is gated. Explicit frozen authorization is required."
        )
    path = args.authorization
    if not path.is_absolute():
        path = ROOT / path
    try:
        receipt = validate_authorization(path)
    except Exception as error:
        write_failure(error, path)
        raise SystemExit(f"GRPO authorization failed: {error}") from error
    print(
        json.dumps(
            {
                "status": "five_element_gate_passed",
                "checks": receipt["checks"],
                "authorization_sha256": receipt["sha256"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    from training.grpo.formal_training import run_authorized_training

    raise SystemExit(run_authorized_training())


if __name__ == "__main__":
    main()
