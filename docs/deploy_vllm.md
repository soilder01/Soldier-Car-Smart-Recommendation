# vLLM 原生 Qwen2.5-7B 接入说明

## 范围

本任务只提供启动脚本、配置样例和 tool-call smoke 检查器。实际 vLLM
安装、CUDA 兼容性验证、29GB 模型启动和环境门禁均留给 Task 13。

V100 主机当前为 driver 470 / CUDA 11.4，存在 vLLM 兼容风险。不要在后端
服务 `.venv` 中安装 vLLM、PyTorch CUDA 或其他 CUDA 包；应使用经 Task 13
验证的独立运行环境。

## 启动

默认只读取本地模型，不会从远端静默下载：

```bash
scripts/start_vllm_qwen7b.sh
```

默认模型路径为 `models/Qwen2.5-7B-Instruct`。路径不存在时脚本立即以状态
码 1 退出。可覆盖以下环境变量：

```bash
MODEL_PATH=/local/Qwen2.5-7B-Instruct \
SERVED_MODEL_NAME=qwen7b-nev \
VLLM_HOST=127.0.0.1 \
VLLM_PORT=8000 \
VLLM_BIN=vllm \
scripts/start_vllm_qwen7b.sh
```

脚本固定启用：

```text
--enable-auto-tool-choice
--tool-call-parser hermes
--max-model-len 8192
--gpu-memory-utilization 0.90
--served-model-name qwen7b-nev
```

## 后端配置

参考 `backend/config/config.vllm.example.yaml`，不要提交真实
`backend/config/config.yaml`。当前配置加载器仍读取 YAML 的
`llm.api_key`、`llm.base_url`、`llm.chat_model`；`CHAT_API_KEY`、
`CHAT_BASE_URL`、`CHAT_MODEL` 是对应的高优先级环境变量。

`CHAT_API_KEY`、`CHAT_BASE_URL`、`CHAT_MODEL` 必须来自同一 provider，
并且三项整组设置。任一项缺失或仅设置部分字段时，该组配置不会生效，
配置加载器也不会跨 `CHAT_*`、`ARK_*` 或 legacy `OPENAI_*` 拼接凭据。

`embedding_*` 字段仅为独立 embedding 服务预留。当前 RAG 使用本地
TF-IDF/BM25-like 检索，不调用 embedding API。

## Tool-call smoke

服务可用后运行：

```bash
CHAT_BASE_URL=http://127.0.0.1:8000/v1 \
CHAT_MODEL=qwen7b-nev \
CHAT_API_KEY=dummy \
.venv/bin/python scripts/check_vllm_tool_call.py
```

默认 schema 只有 `search_and_rank_vehicles`，仅用于连通性 smoke，不是最终
五工具评测。Task 5 建立精确 schema 后，应从
`data_synth.tool_schemas` 导出 JSON 并显式传入：

```bash
.venv/bin/python scripts/check_vllm_tool_call.py \
  --schema-file /path/to/task5-tools.json
```

JSON 可以是 OpenAI tools 数组，也可以是 `{"tools": [...]}`。最终评测必须
复用 Task 5 schema，不得继续以内置单工具 smoke 代替。
