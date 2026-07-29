# V100 训练与 vLLM 运行时兼容门禁

## 当前事实

探针命令：

```bash
PYTHONPATH=. .venv/bin/python scripts/check_model_runtime.py
```

2026-07-23 本机探针结果：

| 项目 | 结果 |
| --- | --- |
| GPU | `Tesla V100-SXM2-32GB` |
| Compute capability | `7.0`（V100 硬件规格） |
| 显存 | `32510 MiB` |
| Driver | `470.129.06` |
| `nvidia-smi` 报告 CUDA | `11.4` |
| `nvcc` | 不存在 |
| 服务 `.venv` | 无 `torch`、`vllm`、`bitsandbytes` |
| 模型路径 | `models/Qwen2.5-7B-Instruct` |
| 模型文件总大小 | `30,477,887,814` bytes |
| 模型架构 | `Qwen2ForCausalLM` |

`config.json`、`tokenizer_config.json` 和
`model.safetensors.index.json` 均存在、非空且可解析。index 引用四个
safetensors 分片，实际检查结果为 `missing_shards=[]`、
`empty_shards=[]`：

| 分片 | 大小（bytes） |
| --- | ---: |
| `model-00001-of-00004.safetensors` | 3,945,441,440 |
| `model-00002-of-00004.safetensors` | 3,864,726,352 |
| `model-00003-of-00004.safetensors` | 3,864,726,424 |
| `model-00004-of-00004.safetensors` | 3,556,377,672 |

目录大小不是模型完整性的判据；只有元数据可解析且 index 的全部引用分片
存在并且 `size > 0`，模型门禁才通过。

index 的每个分片引用必须是模型目录直接包含的简单文件名：

- 类型必须是非空字符串，不允许首尾空白，UTF-8 编码后不得超过 255 bytes。
- 必须以小写 `.safetensors` 结尾，不允许绝对路径、`..`、`/`、`\`、
  drive prefix 或任何子目录。
- 分片检查使用不跟随 symlink 的 `stat`；symlink、非普通文件和 `stat`
  `OSError` 都产生结构化错误并关闭模型门禁。
- 非法引用在任何文件系统访问之前拒绝，绝不读取或 `stat` 模型目录外文件。

## 门禁结论

- `hardware_support_status=supported_fp16`：V100/SM 7.0 支持 FP16 训练与推理。
- `training_compute_dtype=float16`，`bf16_allowed=false`：V100 不以 BF16
  作为训练计算 dtype。
- `binary_runtime_compatibility=unverified`：硬件支持不等于具体
  PyTorch/vLLM wheel 与 driver 兼容。
- `vllm_status=requires_compatible_isolated_runtime`。
- `runtime_gate=blocked_until_verified`。
- `model_gate=ready`。
- `overall_gate=blocked_until_verified`：模型完整不代表整体运行时已验证。
- `service_env_can_install_vllm=false`：禁止向服务 `.venv` 安装 vLLM、
  torch CUDA 或 bitsandbytes。

当前预编译 vLLM 运行时通常基于 CUDA 12.x；driver `470.129.06` 不能据此
直接判定兼容。必须选择与该 driver 明确兼容的独立环境，或先升级 driver，
再针对具体 PyTorch/vLLM wheel 组合验证。`nvidia-smi` 中的 CUDA `11.4`
是 driver 能力报告，不代表本机存在 CUDA toolkit，也不证明任一 wheel
可加载。

## 放行条件

在隔离的训练/vLLM 环境中，以下四项必须全部通过：

1. PyTorch 检测到 CUDA 和 `Tesla V100-SXM2-32GB`，并报告所用 wheel
   版本、wheel CUDA 版本和 driver。
2. bitsandbytes 成功加载 4-bit CUDA kernels，而不是 CPU fallback。
3. vLLM 成功加载本地 Qwen2.5-7B，使用
   `--enable-auto-tool-choice --tool-call-parser hermes` 完成一次 tool-call
   smoke。
4. 服务 `.venv` 继续不包含 vLLM、训练用 torch CUDA 和 bitsandbytes。

任一项缺少版本证据或运行证据时，保持 `blocked_until_verified`，不得将
V100 硬件支持表述为 “vLLM ready”。

## Runtime Evidence Schema

`cuda_driver_version` 仅表示 `nvidia-smi` 报告的 driver CUDA 能力。
隔离环境实际使用的 CUDA 来自
`runtime_evidence.torch_cuda_version`，两者不得混用。版本字段只用于报告
和识别显式不兼容，不能单独放行。

只有 GPU 探针、硬件、版本和 `runtime_evidence` 同时满足下列条件时，
binary runtime 才可变为 `verified`：

```json
{
  "gpu_probe": {
    "status": "ok"
  },
  "runtime_evidence": {
    "isolated_environment": true,
    "torch_cuda_available": true,
    "torch_gpu_name": "Tesla V100-SXM2-32GB",
    "torch_cuda_version": "11.3",
    "bitsandbytes_4bit_cuda_smoke": true,
    "vllm_model_load_smoke": true,
    "vllm_tool_call_smoke": true,
    "service_env_clean": true,
    "vllm_version": "<non-empty>",
    "torch_version": "<non-empty>",
    "bitsandbytes_version": "<non-empty>"
  }
}
```

具体绑定规则：

1. `gpu_probe.status` 必须严格为 `ok`，当前硬件必须为
   `hardware_support_status=supported_fp16`。
2. `driver_version` 与 `runtime_evidence.torch_cuda_version` 必须是可解析的
   点分数字版本；非空但畸形同样 blocked。
3. `torch_gpu_name.casefold()` 必须与探针
   `gpu_name.casefold()` 完全相等。仅包含 `V100`、A100/V100 不一致或其他
   子串匹配均不接受。
4. 任一布尔 evidence 缺失/非 `true`，或任一文本 evidence 为空，均保持
   `blocked_until_verified`。

`overall_gate` 只有在 GPU probe ok、硬件 `supported_fp16`、binary runtime
`verified`、`model_gate=ready` 四项全部成立时才是 `ready`。binary evidence
已验证但模型或 overall 仍 blocked 时，`vllm_status` 使用
`runtime_evidence_verified_overall_blocked`，不得声称整体 verified ready。
旧的 `binary_runtime_verified=true` 单布尔字段不会被接受。
