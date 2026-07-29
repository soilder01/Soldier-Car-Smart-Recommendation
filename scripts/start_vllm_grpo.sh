#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/Qwen2.5-7B-Instruct}"
LORA_PATH="${LORA_PATH:-${ROOT_DIR}/checkpoints/grpo}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen7b-grpo}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8002}"
VLLM_BIN="${VLLM_BIN:-vllm}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "本地模型路径不存在：${MODEL_PATH}" >&2
  echo "请先确认 models/Qwen2.5-7B-Instruct 已存在；脚本不会静默下载模型。" >&2
  exit 1
fi

if [[ ! -d "${LORA_PATH}" ]]; then
  echo "GRPO adapter 不存在：${LORA_PATH}" >&2
  echo "请先完成真实 GRPO 并验证 adapter reload；脚本不会静默下载模型。" >&2
  exit 1
fi

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --enable-lora \
  --lora-modules "${SERVED_MODEL_NAME}=${LORA_PATH}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 8192 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
