"""Local OpenAI-compatible chat service backed by the merged Transformers model."""

from __future__ import annotations

import os
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("LOCAL_MODEL_PATH", ROOT / "merged_model")).resolve()
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "car-7b")
DEFAULT_MAX_TOKENS = int(os.getenv("LOCAL_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("LOCAL_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("LOCAL_TOP_P", "0.95"))

app = FastAPI(title="Local car-7b OpenAI-compatible service", version="1.0.0")

tokenizer: Any | None = None
model: Any | None = None
model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generation_lock = threading.Lock()


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = SERVED_MODEL_NAME
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    n: int = Field(default=1, gt=0, le=16)
    seed: int | None = None
    stream: bool = False


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content)


def _normalize_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        normalized.append(
            {
                "role": message.role,
                "content": _message_content_to_text(message.content),
            }
        )
    return normalized


def _load_model() -> None:
    global tokenizer, model, model_device

    if not MODEL_PATH.exists():
        raise RuntimeError(f"model path does not exist: {MODEL_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; local service requires the V100 GPU")

    print(
        "local_model_load_start "
        f"model_path={MODEL_PATH} served_model={SERVED_MODEL_NAME} "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        trust_remote_code=False,
    )

    load_kwargs = {
        "torch_dtype": torch.float16,
        "local_files_only": True,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            device_map={"": "cuda:0"},
            **load_kwargs,
        )
        model_device = next(model.parameters()).device
        load_mode = "device_map_cuda"
    except ImportError as exc:
        # `device_map` requires accelerate. Keep the serving environment minimal by
        # falling back to a direct CUDA move when accelerate is not installed.
        print(
            "local_model_device_map_unavailable "
            f"reason={exc.__class__.__name__}: {exc}",
            flush=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            **load_kwargs,
        ).to(model_device)
        load_mode = "manual_to_cuda"

    model.eval()
    torch.cuda.empty_cache()
    reserved_mib = torch.cuda.memory_reserved(0) // (1024 * 1024)
    allocated_mib = torch.cuda.memory_allocated(0) // (1024 * 1024)
    print(
        "local_model_load_complete "
        f"mode={load_mode} device={model_device} "
        f"cuda_reserved_mib={reserved_mib} cuda_allocated_mib={allocated_mib}",
        flush=True,
    )


def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda oom" in message


def _generate_batch(
    inputs: dict[str, Any],
    generation_kwargs: dict[str, Any],
    batch_n: int,
) -> Any:
    return model.generate(
        **inputs,
        **generation_kwargs,
        num_return_sequences=batch_n,
    )


def _generate_with_oom_split(
    inputs: dict[str, Any],
    generation_kwargs: dict[str, Any],
    n: int,
    seed: int | None,
) -> Any:
    _apply_seed(seed)
    try:
        return _generate_batch(inputs, generation_kwargs, n)
    except RuntimeError as exc:
        if n <= 1 or not _is_oom(exc):
            raise
        torch.cuda.empty_cache()
        _apply_seed(seed)
        chunks = []
        remaining = n
        while remaining:
            chunk_n = min(4, remaining)
            chunks.append(_generate_batch(inputs, generation_kwargs, chunk_n))
            remaining -= chunk_n
        return torch.cat(chunks, dim=0)


@app.on_event("startup")
def startup() -> None:
    _load_model()


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.model != SERVED_MODEL_NAME:
        raise HTTPException(
            status_code=404,
            detail=f"model {request.model!r} is not served by this process",
        )
    if request.stream:
        raise HTTPException(status_code=400, detail="streaming is not implemented")
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="model is still loading")

    messages = _normalize_messages(request.messages)
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:  # noqa: BLE001 - return a clean OpenAI-style HTTP error.
        raise HTTPException(
            status_code=400,
            detail=f"failed to apply chat template: {exc}",
        ) from exc

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    max_new_tokens = (
        request.max_completion_tokens
        or request.max_tokens
        or DEFAULT_MAX_TOKENS
    )
    n = int(request.n)
    temperature = DEFAULT_TEMPERATURE if request.temperature is None else request.temperature
    top_p = DEFAULT_TOP_P if request.top_p is None else request.top_p
    do_sample = temperature > 0

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(temperature)
        generation_kwargs["top_p"] = float(top_p)
    else:
        generation_kwargs["temperature"] = None
        generation_kwargs["top_p"] = None
        generation_kwargs["top_k"] = None
        if n > 1:
            n = 1

    with generation_lock:
        with torch.inference_mode():
            output_ids = _generate_with_oom_split(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                n=n,
                seed=request.seed,
            )

    choices = []
    completion_tokens = 0
    for index, sequence in enumerate(output_ids):
        generated_ids = sequence[prompt_tokens:]
        content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completion_tokens += int(generated_ids.shape[-1])
        choices.append(
            {
                "index": index,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        )
    now = int(time.time())

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": SERVED_MODEL_NAME,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
