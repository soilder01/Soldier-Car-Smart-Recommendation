"""OpenAI-compatible service using the NF4 base plus checkpoint-150 adapter."""

from __future__ import annotations

import os
import random
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from training.grpo.formal_training import rollout_generation_cache_context


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Qwen2.5-7B-Instruct"
SFT_ADAPTER = ROOT / "checkpoints" / "sft" / "best_adapter"
ADAPTER_PATH = Path(
    os.environ.get(
        "LOCAL_ADAPTER_PATH",
        ROOT / "checkpoints" / "grpo" / "formal_v4" / "restart_1" / "sales_dense_v2" / "checkpoint-150",
    )
).resolve()
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "car-7b")
DEFAULT_MAX_TOKENS = int(os.getenv("LOCAL_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("LOCAL_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("LOCAL_TOP_P", "0.95"))

app = FastAPI(title="Local car-7b adapter OpenAI-compatible service", version="1.0.0")

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


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    return [
        {
            "role": message.role,
            "content": _message_content_to_text(message.content),
        }
        for message in messages
    ]


def _load_model() -> None:
    global tokenizer, model, model_device

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; adapter service requires the V100 GPU")
    if not (ADAPTER_PATH / "adapter_model.safetensors").exists():
        raise RuntimeError(f"missing adapter_model.safetensors: {ADAPTER_PATH}")

    print(
        "adapter_model_load_start "
        f"base={MODEL_PATH} adapter={ADAPTER_PATH} served_model={SERVED_MODEL_NAME} "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}",
        flush=True,
    )

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=False,
    )
    base_model.config.use_cache = False
    adapter_config = json.loads((SFT_ADAPTER / "adapter_config.json").read_text(encoding="utf-8"))
    peft_config = LoraConfig(
        r=adapter_config["r"],
        lora_alpha=adapter_config["lora_alpha"],
        lora_dropout=adapter_config["lora_dropout"],
        target_modules=adapter_config["target_modules"],
        bias=adapter_config["bias"],
        task_type=adapter_config["task_type"],
        inference_mode=False,
    )
    model = get_peft_model(base_model, peft_config)
    state = load_file(str(ADAPTER_PATH / "adapter_model.safetensors"), device="cpu")
    result = set_peft_model_state_dict(model, state, adapter_name="default")
    if list(getattr(result, "unexpected_keys", ())):
        raise RuntimeError("unexpected adapter keys while loading checkpoint-150")
    model.eval()
    model_device = model.device
    torch.cuda.empty_cache()
    print(
        "adapter_model_load_complete "
        f"device={model_device} adapter_sha256={sha256_file(ADAPTER_PATH / 'adapter_model.safetensors')} "
        f"cuda_reserved_mib={torch.cuda.memory_reserved(0) // (1024 * 1024)} "
        f"cuda_allocated_mib={torch.cuda.memory_allocated(0) // (1024 * 1024)}",
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


def _generate_batch(inputs: dict[str, Any], generation_kwargs: dict[str, Any], batch_n: int) -> Any:
    with rollout_generation_cache_context(model, model.generation_config):
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
                "owned_by": "local-adapter",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.model != SERVED_MODEL_NAME:
        raise HTTPException(status_code=404, detail=f"model {request.model!r} is not served by this process")
    if request.stream:
        raise HTTPException(status_code=400, detail="streaming is not implemented")
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="model is still loading")

    prompt = tokenizer.apply_chat_template(
        _normalize_messages(request.messages),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    max_new_tokens = request.max_completion_tokens or request.max_tokens or DEFAULT_MAX_TOKENS
    n = int(request.n)
    temperature = DEFAULT_TEMPERATURE if request.temperature is None else request.temperature
    top_p = DEFAULT_TOP_P if request.top_p is None else request.top_p
    do_sample = temperature > 0

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": model.generation_config.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(temperature)
        generation_kwargs["top_p"] = float(top_p)
    else:
        if n > 1:
            n = 1

    with generation_lock:
        with torch.inference_mode():
            output_ids = _generate_with_oom_split(inputs, generation_kwargs, n, request.seed)

    choices = []
    completion_tokens = 0
    for index, sequence in enumerate(output_ids):
        generated_ids = sequence[prompt_tokens:]
        content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completion_tokens += int(generated_ids.shape[-1])
        choices.append(
            {
                "index": index,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": SERVED_MODEL_NAME,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
