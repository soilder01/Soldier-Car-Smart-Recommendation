#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen7b-nev}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
VLLM_BIN="${VLLM_BIN:-vllm}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "本地模型路径不存在：${MODEL_PATH}" >&2
  echo "请先确认 models/Qwen2.5-7B-Instruct 已存在；脚本不会静默下载模型。" >&2
  exit 1
fi

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
