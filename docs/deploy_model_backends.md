# 模型后端切换说明

`scripts/switch_llm_backend.py` 生成当前 shell 可执行的环境变量导出命令。
它只切换完整的 `CHAT_API_KEY`、`CHAT_BASE_URL`、`CHAT_MODEL` 三元组，
不会输出或修改任何 `EMBEDDING_*` 变量。

## 后端列表

| 后端 | Base URL | Chat model | API key |
| --- | --- | --- | --- |
| `ark` | 当前 shell 的 `ARK_BASE_URL` | 当前 shell 的 `ARK_CHAT_MODEL` | 当前 shell 的 `ARK_API_KEY` |
| `qwen_base_vllm` | `http://127.0.0.1:8000/v1` | `qwen7b-nev` | `dummy` |
| `qwen_sft_vllm` | `http://127.0.0.1:8001/v1` | `qwen7b-sft` | `dummy` |
| `qwen_grpo_vllm` | `http://127.0.0.1:8002/v1` | `qwen7b-grpo` | `dummy` |

本地 profile 假定对应 vLLM 服务已在表中的端口启动。切换脚本本身不启动、
探测或安装 vLLM。

## Python API 与安全边界

`build_backend_env("ark")` 不读取当前进程的 Ark 配置，也不返回真实密钥、
endpoint 或模型名。无论当前环境是否配置 Ark，它始终返回以下稳定符号映射：

```python
{
    "CHAT_API_KEY": "${ARK_API_KEY}",
    "CHAT_BASE_URL": "${ARK_BASE_URL}",
    "CHAT_MODEL": "${ARK_CHAT_MODEL}",
}
```

`render_exports(build_backend_env("ark"))` 只把这个精确、固定的三元组识别为受信
Ark profile，并生成 eval-time 校验与导出 block。其他任意字符串和非精确匹配的
Ark-like 映射仍逐值经过 `shlex.quote()`，不会获得 shell expansion 能力。
CLI 对所有 profile 均调用同一 `build_backend_env()` + `render_exports()` 路径，
因此 Python API 与 CLI 的输出语义一致。

## 本地 vLLM

查看将要执行的导出命令：

```bash
.venv/bin/python scripts/switch_llm_backend.py qwen_base_vllm
```

应用到当前 shell：

```bash
eval "$(.venv/bin/python scripts/switch_llm_backend.py qwen_base_vllm)"
```

将参数替换为 `qwen_sft_vllm` 或 `qwen_grpo_vllm` 即可切换到对应服务。

## Ark

先在当前 shell 提供完整 Ark 三元组：

```bash
export ARK_API_KEY='<secret>'
export ARK_BASE_URL='https://ark.example/api/v3'
export ARK_CHAT_MODEL='<endpoint-or-model-id>'
```

再应用 Ark profile：

```bash
eval "$(.venv/bin/python scripts/switch_llm_backend.py ark)"
```

Ark render/CLI 输出不包含三个变量的实际值，而是在 `eval` 时从当前 shell
读取它们。普通 Bash 和 `set -u` 下，任一 `ARK_*` 未设置、为空字符串或只含
空白字符时，block 都会向 stderr 明确指出变量名、返回非零，并且三个既有
`CHAT_*` 全部保持不变。三个值均有效时才通过一个 `export` 命令统一提交。

变量值始终在双引号内展开，因此空格、引号、美元符号、分号、换行及
`$(...)` 文本均可无损传递；变量内容不会被当作 shell 源码再次执行，也不会把
`${ARK_*}` 模板字面量写入 `CHAT_*`。

## 验证

```bash
printf 'CHAT_BASE_URL=%s\nCHAT_MODEL=%s\n' "$CHAT_BASE_URL" "$CHAT_MODEL"
```

不要打印 `CHAT_API_KEY`。`eval` 只应执行本仓库可信脚本生成的输出。当前 RAG
继续使用本地检索逻辑；Chat profile 切换不会清空、覆盖或创建
`EMBEDDING_*` 配置。
