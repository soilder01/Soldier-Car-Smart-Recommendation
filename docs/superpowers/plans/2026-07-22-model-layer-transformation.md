# 模型层改造 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不重写现有推荐业务的前提下，为当前 LangGraph 汽车推荐系统增设可复现的本地 Qwen2.5-7B 模型层，覆盖 vLLM 接入、数据合成、QLoRA SFT、GRPO/RLVR、后端切换和 held-out 对比评估。

**Architecture:** 保留现有 FastAPI/LangGraph/RAG/Obsidian 应用层，新增模型层工具、训练层脚手架和评估记录。Phase 0-1 先锁定基线与原生 Qwen vLLM 可演示闭环；Phase 2-6 再推进数据、训练、强化、部署切换和最终对比。

**Tech Stack:** FastAPI, LangGraph, OpenAI-compatible API, vLLM, Qwen2.5-7B-Instruct, Python, SQLite, scikit-learn, QLoRA, GRPO/RLVR, Markdown reports.

## Global Constraints

- 禁止提交私密文件 `backend/config/config.yaml`。
- 优先使用本地模型路径 `models/Qwen2.5-7B-Instruct`，不静默下载模型。
- 所有 chat 模型后端统一使用 OpenAI-compatible API。
- 服务依赖与训练依赖必须分离；训练依赖进入 `requirements-train.txt`。
- 除非提供明确兼容层，否则保持现有应用 API 稳定。
- 工具 schema 必须与 `backend/app/services/agent_graph.py` 中真实 5 个工具逐字对齐。
- 训练前必须记录基线指标。
- Phase 1 原生 Qwen 基线未记录前，不开始 SFT。
- SFT 未产生可度量工具调用改进或明确失败报告前，不开始 GRPO。
- 训练奖励与最终评估必须隔离；Phase 6 必须使用 held-out 测试集。
- 不允许将同一批 `evaluate_*.py` 用例同时作为 GRPO answer reward 和最终项目证明。
- 现有 `scripts/evaluate_*.py` 主要用于应用层和工程门禁；模型质量证明使用独立的 `data/model_training/eval/held_out.jsonl` 与 `scripts/evaluate_model_outputs.py`。
- vLLM tool calling 启动参数必须包含 `--enable-auto-tool-choice --tool-call-parser hermes`。
- 当前硬件为 Tesla V100-SXM2-32GB；QLoRA compute dtype 必须使用 `float16`，不得默认使用 `bfloat16`。
- 当前 RAG 使用本地 TF-IDF/BM25-like，不调用 embedding API；`EMBEDDING_*` 配置是预留项。
- 当前工作区不是 Git 仓库时，所有 “Commit” 步骤改为更新对应 progress log，并在最终报告中记录未 commit 原因。

---

## 文件结构规划

新增文件：

- `docs/model_layer_baseline.md`：Phase 0 基线、依赖状态、评估脚本分类、teacher 可用性和 release gate 阻断记录。
- `docs/deploy_vllm.md`：Phase 1 vLLM 原生 Qwen 启动与验证说明。
- `docs/model_layer_phase1_vllm_baseline.md`：原生 Qwen vLLM 接入基线报告。
- `docs/deploy_model_backends.md`：ark/base/sft/grpo 后端切换说明。
- `docs/model_layer_phase3_sft_report.md`：SFT 训练与应用层评估报告模板。
- `docs/model_layer_phase4_grpo_report.md`：GRPO 训练、reward、显存与 held-out 隔离报告模板。
- `docs/model_layer_report.md`：Phase 6 四列 held-out 对比报告。
- `docs/model_layer_progress_log.md`：非 Git 环境下的持续推进记录。
- `backend/config/config.vllm.example.yaml`：vLLM 配置样例，不含真实密钥。
- `scripts/start_vllm_qwen7b.sh`：原生 Qwen vLLM 启动脚本。
- `scripts/start_vllm_sft.sh`：SFT adapter vLLM 启动脚本。
- `scripts/start_vllm_grpo.sh`：GRPO adapter vLLM 启动脚本。
- `scripts/switch_llm_backend.py`：安全切换 chat 后端配置，保留 embedding 配置。
- `scripts/evaluate_model_layer_baseline.py`：汇总 Phase 0 工程基线并写报告，不将工程门禁误当模型质量 held-out。
- `scripts/check_vllm_tool_call.py`：直接调用 vLLM `/v1/chat/completions` 验证 tools 返回。
- `data_synth/__init__.py`：数据合成包入口。
- `data_synth/tool_schemas.py`：5 个工具 OpenAI function schema 导出。
- `data_synth/validate_tool_data.py`：SFT/GRPO 数据 schema 与 held-out 泄漏校验。
- `data_synth/generate_sft_data.py`：教师模型数据合成入口。
- `data/model_training/README.md`：训练数据目录约束说明。
- `data/model_training/eval/reward_visible.jsonl`：训练期 reward adapter 可见的模型质量评测集。
- `data/model_training/eval/held_out.jsonl`：只用于 Phase 6 的模型质量 held-out 集。
- `scripts/evaluate_model_outputs.py`：统一评估工具选择、参数合法率、推荐命中率和幻觉率。
- `scripts/run_model_layer_eval.py`：对指定 OpenAI-compatible 后端运行评测集并保存原始输出。
- `training/sft/README.md`：SFT 训练说明。
- `training/sft/qlora_sft_config.yaml`：SFT 配置模板。
- `training/grpo/README.md`：GRPO 训练说明。
- `training/grpo/reward_fn.py`：奖励函数结构与缓存接口。
- `training/grpo/train_grpo.py`：GRPO 训练入口骨架。
- `training/grpo/grpo_config.yaml`：GRPO 配置模板。
- `requirements-train.txt`：训练依赖。
- `tests/model_layer/test_config_chat_embedding.py`：chat/embedding 配置兼容测试。
- `tests/model_layer/test_tool_schemas.py`：工具 schema 对齐测试。
- `tests/model_layer/test_switch_llm_backend.py`：后端切换配置测试。
- `tests/model_layer/test_validate_tool_data.py`：训练数据校验测试。
- `tests/model_layer/test_reward_split.py`：reward-visible 与 held-out 隔离测试。
- `tests/model_layer/test_reward_fn.py`：奖励函数缓存与轻量校验测试。

修改文件：

- `backend/app/config.py`：增加 chat/embedding 分离配置，同时保留 legacy fallback。
- `backend/app/services/llm_client.py`：改用 chat 配置构造 client，保留 `openai_client()` 对外接口。
- `backend/app/main.py`：公开配置诊断中展示 chat/embedding 后端状态，不暴露密钥。
- `backend/requirements.txt`：仅维护服务依赖，不加入训练依赖。
- `.gitignore`：忽略 `checkpoints/`、大模型训练输出和生成数据中间产物；保留小型 README/report 可跟踪。
- `README.md`：Phase 6 后更新模型层说明与真实数字。

目录约束：

- `models/`、`checkpoints/`、大体积 `data/model_training/*.jsonl` 默认不进入 Git。
- 结构性代码、配置样例、报告模板和小型统计报告可以进入仓库。
- 所有生成报告必须记录命令、环境、通过/失败和下一步。

---

### Task 1: Phase 0 基线与评估脚本分类

**Files:**

- Create: `scripts/evaluate_model_layer_baseline.py`
- Create: `docs/model_layer_baseline.md`
- Create: `docs/model_layer_progress_log.md`
- Test: `tests/model_layer/test_reward_split.py`

**Interfaces:**

- Consumes: existing `scripts/evaluate_agent_regression.py`, `scripts/evaluate_knowledge_fusion.py`, `scripts/evaluate_release_gate.py`
- Produces: `build_baseline_report() -> dict`, `docs/model_layer_baseline.md`, reward split metadata section

- [ ] **Step 1: 创建测试文件，定义 reward/held-out 分离的最小要求**

Create `tests/model_layer/test_reward_split.py`:

```python
from scripts.evaluate_model_layer_baseline import classify_evaluation_outputs


def test_classify_evaluation_outputs_keeps_engineering_gates_separate():
    scripts = {
        "evaluate_agent_regression.py": {"numeric": True, "deterministic": True, "engineering_gate": True},
        "evaluate_knowledge_fusion.py": {"numeric": True, "deterministic": True, "engineering_gate": True},
        "evaluate_release_gate.py": {"numeric": True, "deterministic": True, "engineering_gate": True},
    }

    result = classify_evaluation_outputs(scripts)

    assert "reward_compatible" in result
    assert "report_only" in result
    assert "engineering_gate" in result
    assert "evaluate_agent_regression.py" in result["engineering_gate"]
    assert "evaluate_knowledge_fusion.py" in result["engineering_gate"]
    assert "evaluate_knowledge_fusion.py" not in result["reward_compatible"]
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer/test_reward_split.py -v
```

Expected: FAIL，提示 `scripts.evaluate_model_layer_baseline` 不存在。

- [ ] **Step 3: 实现基线脚本的分类函数与报告骨架**

Create `scripts/evaluate_model_layer_baseline.py`:

```python
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "docs" / "model_layer_baseline.md"
PROGRESS = ROOT / "docs" / "model_layer_progress_log.md"


def classify_evaluation_outputs(scripts: Dict[str, Dict[str, Any]]) -> Dict[str, list[str]]:
    result = {"reward_compatible": [], "report_only": [], "engineering_gate": []}
    for name, meta in scripts.items():
        if meta.get("engineering_gate"):
            result["engineering_gate"].append(name)
        elif meta.get("numeric") and meta.get("deterministic"):
            result["reward_compatible"].append(name)
        else:
            result["report_only"].append(name)
    return {key: sorted(value) for key, value in result.items()}


def run_command(cmd: list[str]) -> Dict[str, Any]:
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    return {
        "cmd": " ".join(cmd),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "passed": completed.returncode == 0,
    }


def detect_teacher_config() -> Dict[str, Any]:
    import os

    configured = bool(os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY"))
    return {
        "configured": configured,
        "status": "available_by_env" if configured else "missing",
        "note": "Phase 2 requires Ark or an equivalent OpenAI-compatible teacher endpoint.",
    }


def build_baseline_report() -> Dict[str, Any]:
    checks = {
        "backend_import": run_command(["bash", "-lc", "PYTHONPATH=backend .venv/bin/python - <<'PY'\nimport app.main\nprint('app.main OK')\nPY"]),
        "health_testclient": run_command(["bash", "-lc", "PYTHONPATH=backend .venv/bin/python - <<'PY'\nfrom fastapi.testclient import TestClient\nfrom app.main import app\nwith TestClient(app) as client:\n    response = client.get('/api/health')\n    print(response.status_code)\n    print(response.json())\n    raise SystemExit(0 if response.status_code == 200 else 1)\nPY"]),
        "agent_regression": run_command(["bash", "-lc", "PYTHONPATH=backend .venv/bin/python scripts/evaluate_agent_regression.py"]),
        "knowledge_fusion": run_command(["bash", "-lc", "PYTHONPATH=backend .venv/bin/python scripts/evaluate_knowledge_fusion.py"]),
        "release_gate": run_command(["bash", "-lc", "PYTHONPATH=backend .venv/bin/python scripts/evaluate_release_gate.py"]),
    }
    split = classify_evaluation_outputs(
        {
            "evaluate_agent_regression.py": {"numeric": True, "deterministic": True, "engineering_gate": True},
            "evaluate_release_gate.py": {"numeric": True, "deterministic": True, "engineering_gate": True},
            "evaluate_knowledge_fusion.py": {"numeric": True, "deterministic": True, "engineering_gate": True},
        }
    )
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checks": checks,
        "reward_split": split,
        "teacher": detect_teacher_config(),
        "notes": [
            "Reward-compatible cases are allowed for GRPO reward adapters.",
            "Engineering-gate scripts do not prove model quality.",
            "Phase 6 uses a separate model-output held-out dataset.",
        ],
    }


def write_markdown(report: Dict[str, Any]) -> None:
    lines = [
        "# 模型层改造 Phase 0 基线报告",
        "",
        f"生成时间：{report['generated_at']}",
        "",
        "## 评估脚本划分",
        "",
        f"- reward-compatible：{', '.join(report['reward_split']['reward_compatible']) or '无'}",
        f"- report-only：{', '.join(report['reward_split']['report_only']) or '无'}",
        f"- engineering-gate：{', '.join(report['reward_split']['engineering_gate']) or '无'}",
        "",
        "## 教师模型可用性",
        "",
        f"- 状态：{report['teacher']['status']}",
        f"- 说明：{report['teacher']['note']}",
        "",
        "## 检查结果",
        "",
    ]
    for name, check in report["checks"].items():
        lines.append(f"- {name}：{'通过' if check['passed'] else '失败'}，命令 `{check['cmd']}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_progress(report: Dict[str, Any]) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(f"\n## {report['generated_at']} Phase 0\n\n")
        f.write("- 已生成基线报告。\n")
        f.write("- 当前工作区不是 Git 仓库时，以本文件记录进度。\n")


def main() -> None:
    report = build_baseline_report()
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report)
    append_progress(report)
    print(json.dumps({"generated_at": report["generated_at"], "reward_split": report["reward_split"], "teacher": report["teacher"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试与基线脚本**

Run:

```bash
PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer/test_reward_split.py -v
PYTHONPATH=backend .venv/bin/python scripts/evaluate_model_layer_baseline.py
```

Expected:

- pytest PASS。
- `docs/model_layer_baseline.md` 被创建。
- 输出包含 `reward_compatible` 和 `engineering_gate`。

- [ ] **Step 5: 检查记录文件**

Run:

```bash
test -s docs/model_layer_baseline.md && test -s docs/model_layer_progress_log.md
```

Expected: exit code 0。

- [ ] **Step 6: Commit 或记录非 Git 状态**

If Git works:

```bash
git add scripts/evaluate_model_layer_baseline.py tests/model_layer/test_reward_split.py docs/model_layer_baseline.md docs/model_layer_progress_log.md
git commit -m "docs: record model layer baseline"
```

If Git is unavailable, append to `docs/model_layer_progress_log.md`:

```markdown
Git unavailable: root workspace is not a git repository. Phase 0 artifacts written without commit.
```

---

### Task 2: Chat/Embedding 配置分离

**Files:**

- Modify: `backend/app/config.py`
- Modify: `backend/app/services/llm_client.py`
- Modify: `backend/app/main.py`
- Create: `tests/model_layer/test_config_chat_embedding.py`

**Interfaces:**

- Consumes: existing `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `CHAT_MODEL`, `EMBEDDING_MODEL`
- Produces: `CHAT_API_KEY`, `CHAT_BASE_URL`, `CHAT_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `chat_client()`, legacy-compatible `openai_client()`

- [ ] **Step 1: 写配置兼容测试**

Create `tests/model_layer/test_config_chat_embedding.py`:

```python
import importlib


def reload_config(monkeypatch, **env):
    for key in [
        "CHAT_API_KEY", "CHAT_BASE_URL", "CHAT_MODEL",
        "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL",
        "ARK_API_KEY", "ARK_BASE_URL", "ARK_CHAT_MODEL", "ARK_EMBEDDING_MODEL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_CHAT_MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import app.config as config
    return importlib.reload(config)


def test_chat_config_prefers_chat_env(monkeypatch):
    config = reload_config(
        monkeypatch,
        CHAT_API_KEY="chat-key",
        CHAT_BASE_URL="http://chat/v1",
        CHAT_MODEL="chat-model",
        ARK_API_KEY="ark-key",
        ARK_BASE_URL="http://ark/v1",
        ARK_CHAT_MODEL="ark-model",
    )

    assert config.CHAT_API_KEY == "chat-key"
    assert config.CHAT_BASE_URL == "http://chat/v1"
    assert config.CHAT_MODEL == "chat-model"
    assert config.OPENAI_API_KEY == "chat-key"
    assert config.OPENAI_BASE_URL == "http://chat/v1"


def test_embedding_config_is_separate_and_optional(monkeypatch):
    config = reload_config(
        monkeypatch,
        CHAT_BASE_URL="http://chat/v1",
        EMBEDDING_BASE_URL="http://embedding/v1",
        EMBEDDING_MODEL="bge-test",
    )

    assert config.CHAT_BASE_URL == "http://chat/v1"
    assert config.EMBEDDING_BASE_URL == "http://embedding/v1"
    assert config.EMBEDDING_MODEL == "bge-test"
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
PYTHONPATH=backend .venv/bin/python -m pytest tests/model_layer/test_config_chat_embedding.py -v
```

Expected: FAIL，提示 `CHAT_API_KEY` 等变量不存在。

- [ ] **Step 3: 修改 `backend/app/config.py`**

Add after `LLM = SETTINGS.get("llm", {})`:

```python
CHAT_API_KEY = os.getenv("CHAT_API_KEY") or os.getenv("ARK_API_KEY") or LLM.get("api_key", "") or os.getenv("OPENAI_API_KEY", "")
CHAT_BASE_URL = (
    os.getenv("CHAT_BASE_URL")
    or os.getenv("ARK_BASE_URL")
    or LLM.get("base_url", "")
    or os.getenv("OPENAI_BASE_URL", "")
).strip()
CHAT_MODEL = (
    os.getenv("CHAT_MODEL")
    or os.getenv("ARK_CHAT_MODEL")
    or LLM.get("chat_model", "")
    or os.getenv("OPENAI_CHAT_MODEL", "")
).strip()

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("ARK_EMBEDDING_API_KEY") or CHAT_API_KEY
EMBEDDING_BASE_URL = (
    os.getenv("EMBEDDING_BASE_URL")
    or os.getenv("ARK_EMBEDDING_BASE_URL")
    or LLM.get("embedding_base_url", "")
    or CHAT_BASE_URL
).strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or os.getenv("ARK_EMBEDDING_MODEL") or LLM.get("embedding_model", "")

# Legacy aliases. Existing application code imports these names.
OPENAI_API_KEY = CHAT_API_KEY
OPENAI_BASE_URL = CHAT_BASE_URL
```

Replace the old `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `CHAT_MODEL`, `EMBEDDING_MODEL` assignments with the block above.

- [ ] **Step 4: 修改 `backend/app/services/llm_client.py`**

Replace imports and add `chat_client()`:

```python
import httpx
from openai import OpenAI

from app.config import CHAT_API_KEY, CHAT_BASE_URL, TIMEOUT


def chat_client() -> OpenAI:
    return OpenAI(
        api_key=CHAT_API_KEY,
        base_url=CHAT_BASE_URL,
        http_client=httpx.Client(trust_env=False, timeout=TIMEOUT),
    )


def openai_client() -> OpenAI:
    return chat_client()
```

Keep existing `mask_secret()` and `check_chat_model()` but update references inside `check_chat_model()` from `OPENAI_API_KEY`/`OPENAI_BASE_URL` to `CHAT_API_KEY`/`CHAT_BASE_URL`.

- [ ] **Step 5: 修改 `/api/config/public` 诊断输出**

In `backend/app/main.py`, update `public_config()` to include non-secret backend status:

```python
"chat": {
    "base_url_configured": bool(OPENAI_BASE_URL),
    "model_configured": bool(CHAT_MODEL),
    "model": "已配置（已隔离）" if CHAT_MODEL else "未配置",
},
"embedding": {
    "base_url_configured": bool(getattr(__import__("app.config", fromlist=["EMBEDDING_BASE_URL"]), "EMBEDDING_BASE_URL", "")),
    "model_configured": bool(EMBEDDING_MODEL),
    "model": "已配置（预留）" if EMBEDDING_MODEL else "未配置（当前RAG未使用embedding API）",
},
```

Also import `EMBEDDING_BASE_URL` from `app.config` at the top of `main.py`.

- [ ] **Step 6: 跑测试与导入检查**

Run:

```bash
PYTHONPATH=backend .venv/bin/python -m pytest tests/model_layer/test_config_chat_embedding.py -v
PYTHONPATH=backend .venv/bin/python - <<'PY'
import app.main
from app.services.llm_client import openai_client, chat_client
print("imports ok")
PY
```

Expected: tests PASS, imports ok。

- [ ] **Step 7: 更新进度记录**

Append to `docs/model_layer_progress_log.md`:

```markdown
## Phase 1 config split

- 已增加 chat/embedding 配置分离。
- 当前 RAG 仍使用本地 TF-IDF/BM25-like，embedding 配置为预留项。
```

---

### Task 3: vLLM 原生 Qwen 启动脚本与配置样例

**Files:**

- Create: `scripts/start_vllm_qwen7b.sh`
- Create: `backend/config/config.vllm.example.yaml`
- Create: `docs/deploy_vllm.md`
- Create: `scripts/check_vllm_tool_call.py`
- Test: `tests/model_layer/test_vllm_scripts.py`

**Interfaces:**

- Consumes: local model path `models/Qwen2.5-7B-Instruct`
- Produces: reproducible vLLM start command with hermes tool parser

- [ ] **Step 1: 写脚本内容测试**

Create `tests/model_layer/test_vllm_scripts.py`:

```python
from pathlib import Path


def test_start_vllm_script_contains_required_tool_call_flags():
    script = Path("scripts/start_vllm_qwen7b.sh").read_text(encoding="utf-8")

    assert "--enable-auto-tool-choice" in script
    assert "--tool-call-parser hermes" in script
    assert "--max-model-len 8192" in script
    assert "--gpu-memory-utilization 0.90" in script
    assert "--served-model-name qwen7b-nev" in script
    assert "models/Qwen2.5-7B-Instruct" in script
    assert "exit 1" in script
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
.venv/bin/python -m pytest tests/model_layer/test_vllm_scripts.py -v
```

Expected: FAIL，脚本不存在。

- [ ] **Step 3: 创建 `scripts/start_vllm_qwen7b.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen7b-nev}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "本地模型路径不存在：${MODEL_PATH}" >&2
  echo "请先确认 models/Qwen2.5-7B-Instruct 已存在；脚本不会静默下载模型。" >&2
  exit 1
fi

exec vllm serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

Run:

```bash
chmod +x scripts/start_vllm_qwen7b.sh
```

- [ ] **Step 4: 创建 `backend/config/config.vllm.example.yaml`**

```yaml
llm:
  # Chat 后端：本地 vLLM OpenAI-compatible endpoint
  api_key: "dummy"
  base_url: "http://127.0.0.1:8000/v1"
  chat_model: "qwen7b-nev"
  temperature: 0.2
  timeout: 60

  # 当前 RAG 使用本地 TF-IDF/BM25-like，不调用 embedding API。
  # 以下字段为未来 embedding 服务预留；Phase 1 不需要启动 embedding 服务。
  embedding_base_url: ""
  embedding_model: ""

app:
  tavily_api_key: ""
```

- [ ] **Step 5: 创建 `scripts/check_vllm_tool_call.py`**

```python
import json
import os

from openai import OpenAI


def main() -> None:
    base_url = os.getenv("CHAT_BASE_URL", "http://127.0.0.1:8000/v1")
    model = os.getenv("CHAT_MODEL", "qwen7b-nev")
    client = OpenAI(api_key=os.getenv("CHAT_API_KEY", "dummy"), base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "预算25万，三口之家，推荐新能源SUV"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_and_rank_vehicles",
                    "description": "Search and rank vehicles for a user profile.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "budget_max": {"type": "integer"},
                            "preferred_type": {"type": "string"},
                            "preferred_energy": {"type": "string"},
                            "concerns": {"type": "string"},
                            "top_k": {"type": "integer"},
                        },
                    },
                },
            }
        ],
        tool_choice="auto",
        temperature=0,
    )
    message = response.choices[0].message
    print(json.dumps(message.model_dump(), ensure_ascii=False, indent=2))
    if not message.tool_calls:
        raise SystemExit("vLLM did not return tool_calls")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 创建 `docs/deploy_vllm.md`**

```markdown
# vLLM 原生 Qwen2.5-7B 接入说明

## 目标

使用本地 `models/Qwen2.5-7B-Instruct` 启动 OpenAI-compatible vLLM chat 服务，并验证 tool calling。

## 启动

```bash
scripts/start_vllm_qwen7b.sh
```

必须包含：

```bash
--enable-auto-tool-choice
--tool-call-parser hermes
--max-model-len 8192
--gpu-memory-utilization 0.90
```

## 配置

参考 `backend/config/config.vllm.example.yaml`。不要提交真实 `backend/config/config.yaml`。

## 验证

```bash
CHAT_BASE_URL=http://127.0.0.1:8000/v1 CHAT_MODEL=qwen7b-nev CHAT_API_KEY=dummy \
  PYTHONPATH=backend .venv/bin/python scripts/check_vllm_tool_call.py
```

随后启动后端并验证 `/api/agent/recommend` 与 `/api/recommend-stream`。

```

- [ ] **Step 7: 跑测试**

Run:

```bash
.venv/bin/python -m pytest tests/model_layer/test_vllm_scripts.py -v
```

Expected: PASS。

- [ ] **Step 8: 记录 Phase 1 未启动或启动结果**

If GPU/vLLM is unavailable, append:

```markdown
## Phase 1 vLLM status

- vLLM 启动脚本、配置样例和 tool-call 检查脚本已完成。
- 当前未启动 vLLM 的原因：记录实际环境原因。
```

If vLLM runs, record command output and API results in `docs/model_layer_phase1_vllm_baseline.md`。

---

### Task 4: 后端切换脚本

**Files:**

- Create: `scripts/switch_llm_backend.py`
- Create: `tests/model_layer/test_switch_llm_backend.py`
- Modify: `docs/deploy_model_backends.md`

**Interfaces:**

- Consumes: backend name `ark | qwen_base_vllm | qwen_sft_vllm | qwen_grpo_vllm`
- Produces: environment export text or YAML profile without writing secrets

- [ ] **Step 1: 写切换脚本测试**

Create `tests/model_layer/test_switch_llm_backend.py`:

```python
from scripts.switch_llm_backend import build_backend_env


def test_qwen_base_backend_preserves_embedding_when_not_requested():
    env = build_backend_env("qwen_base_vllm")

    assert env["CHAT_BASE_URL"] == "http://127.0.0.1:8000/v1"
    assert env["CHAT_MODEL"] == "qwen7b-nev"
    assert env["CHAT_API_KEY"] == "dummy"
    assert "EMBEDDING_BASE_URL" not in env
    assert "EMBEDDING_MODEL" not in env


def test_unknown_backend_raises():
    try:
        build_backend_env("unknown")
    except ValueError as exc:
        assert "unknown backend" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/model_layer/test_switch_llm_backend.py -v
```

Expected: FAIL，脚本不存在。

- [ ] **Step 3: 实现 `scripts/switch_llm_backend.py`**

```python
import argparse
from typing import Dict


BACKENDS = {
    "ark": {
        "CHAT_API_KEY": "${ARK_API_KEY}",
        "CHAT_BASE_URL": "${ARK_BASE_URL}",
        "CHAT_MODEL": "${ARK_CHAT_MODEL}",
    },
    "qwen_base_vllm": {
        "CHAT_API_KEY": "dummy",
        "CHAT_BASE_URL": "http://127.0.0.1:8000/v1",
        "CHAT_MODEL": "qwen7b-nev",
    },
    "qwen_sft_vllm": {
        "CHAT_API_KEY": "dummy",
        "CHAT_BASE_URL": "http://127.0.0.1:8001/v1",
        "CHAT_MODEL": "qwen7b-sft",
    },
    "qwen_grpo_vllm": {
        "CHAT_API_KEY": "dummy",
        "CHAT_BASE_URL": "http://127.0.0.1:8002/v1",
        "CHAT_MODEL": "qwen7b-grpo",
    },
}


def build_backend_env(name: str) -> Dict[str, str]:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend: {name}")
    return dict(BACKENDS[name])


def render_exports(env: Dict[str, str]) -> str:
    return "\n".join(f"export {key}={value!r}" for key, value in env.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=sorted(BACKENDS))
    args = parser.parse_args()
    print(render_exports(build_backend_env(args.backend)))
    print("# embedding settings are intentionally not changed by this script")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 创建 `docs/deploy_model_backends.md`**

```markdown
# 模型后端切换说明

支持后端：

- `ark`
- `qwen_base_vllm`
- `qwen_sft_vllm`
- `qwen_grpo_vllm`

生成环境变量：

```bash
.venv/bin/python scripts/switch_llm_backend.py qwen_base_vllm
```

应用到当前 shell：

```bash
eval "$(.venv/bin/python scripts/switch_llm_backend.py qwen_base_vllm)"
```

脚本只切换 chat 后端，不改 `EMBEDDING_*`。当前 RAG 不调用 embedding API。

```

- [ ] **Step 5: 跑测试**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/model_layer/test_switch_llm_backend.py -v
```

Expected: PASS。

---

### Task 5: 工具 Schema 导出与校验

**Files:**

- Create: `data_synth/__init__.py`
- Create: `data_synth/tool_schemas.py`
- Create: `tests/model_layer/test_tool_schemas.py`

**Interfaces:**

- Consumes: real tool signatures from `backend/app/services/agent_graph.py`
- Produces: `build_tool_schemas() -> list[dict]`, `TOOL_SCHEMAS`

- [ ] **Step 1: 写 schema 对齐测试**

Create `tests/model_layer/test_tool_schemas.py`:

```python
from data_synth.tool_schemas import TOOL_SCHEMAS, build_tool_schemas


def test_tool_schema_names_match_real_five_tools():
    names = [item["function"]["name"] for item in build_tool_schemas()]

    assert names == [
        "extract_user_profile",
        "search_and_rank_vehicles",
        "retrieve_knowledge_base",
        "search_web_info",
        "generate_sales_talk",
    ]


def test_search_and_rank_vehicle_schema_has_exact_parameters():
    schemas = {item["function"]["name"]: item for item in TOOL_SCHEMAS}
    props = schemas["search_and_rank_vehicles"]["function"]["parameters"]["properties"]

    assert set(props) == {"budget_max", "preferred_type", "preferred_energy", "concerns", "top_k"}
    assert props["budget_max"]["type"] == "integer"
    assert props["top_k"]["type"] == "integer"
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer/test_tool_schemas.py -v
```

Expected: FAIL，`data_synth.tool_schemas` 不存在。

- [ ] **Step 3: 创建 `data_synth/tool_schemas.py`**

```python
from copy import deepcopy
from typing import Dict, List


TOOL_SCHEMAS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "extract_user_profile",
            "description": "Extract a structured vehicle purchase profile from user query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "budget_max": {"type": "integer", "default": 0},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_and_rank_vehicles",
            "description": "Search and rank vehicles from the local vehicle database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget_max": {"type": "integer", "default": 0},
                    "preferred_type": {"type": "string", "default": ""},
                    "preferred_energy": {"type": "string", "default": ""},
                    "concerns": {"type": "string", "default": ""},
                    "top_k": {"type": "integer", "default": 5},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge_base",
            "description": "Retrieve local knowledge base evidence.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web_info",
            "description": "Search public web information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_sales_talk",
            "description": "Generate sales talk for a recommended vehicle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget_max": {"type": "integer", "default": 0},
                    "concerns": {"type": "string", "default": ""},
                    "top_model": {"type": "string", "default": ""},
                },
            },
        },
    },
]


def build_tool_schemas() -> List[Dict]:
    return deepcopy(TOOL_SCHEMAS)
```

Create `data_synth/__init__.py` as an empty file.

- [ ] **Step 4: 跑测试**

Run:

```bash
PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer/test_tool_schemas.py -v
```

Expected: PASS。

---

### Task 6: 训练数据校验器与 held-out 泄漏防线

**Files:**

- Create: `data_synth/validate_tool_data.py`
- Create: `tests/model_layer/test_validate_tool_data.py`
- Create: `data/model_training/README.md`

**Interfaces:**

- Consumes: JSONL messages with `tool_calls`, `held_out_ids`
- Produces: `validate_record(record: dict, held_out_ids: set[str]) -> list[str]`

- [ ] **Step 1: 写校验器测试**

Create `tests/model_layer/test_validate_tool_data.py`:

```python
from data_synth.validate_tool_data import validate_record


def test_validate_record_rejects_unknown_tool():
    record = {
        "id": "sample-1",
        "messages": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "unknown_tool", "arguments": "{}"}}]},
        ],
    }

    errors = validate_record(record, held_out_ids=set())

    assert any("unknown tool" in error for error in errors)


def test_validate_record_rejects_held_out_leakage():
    record = {"id": "heldout-1", "messages": []}

    errors = validate_record(record, held_out_ids={"heldout-1"})

    assert any("held-out" in error for error in errors)
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer/test_validate_tool_data.py -v
```

Expected: FAIL，校验器不存在。

- [ ] **Step 3: 实现校验器**

Create `data_synth/validate_tool_data.py`:

```python
import json
from typing import Dict, List, Set

from data_synth.tool_schemas import TOOL_SCHEMAS

ALLOWED_TOOLS = {item["function"]["name"] for item in TOOL_SCHEMAS}


def _iter_tool_calls(record: Dict):
    for message in record.get("messages", []):
        for call in message.get("tool_calls", []) or []:
            yield call


def validate_record(record: Dict, held_out_ids: Set[str]) -> List[str]:
    errors: List[str] = []
    record_id = str(record.get("id", ""))
    if record_id in held_out_ids:
        errors.append(f"record {record_id} leaks held-out case into training data")
    for call in _iter_tool_calls(record):
        function = call.get("function", {})
        name = function.get("name", "")
        if name not in ALLOWED_TOOLS:
            errors.append(f"unknown tool: {name}")
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                json.loads(arguments or "{}")
            except json.JSONDecodeError:
                errors.append(f"invalid JSON arguments for tool {name}")
        elif not isinstance(arguments, dict):
            errors.append(f"invalid argument type for tool {name}: {type(arguments).__name__}")
    return errors
```

- [ ] **Step 4: 创建训练数据目录说明**

Create `data/model_training/README.md`:

```markdown
# 模型训练数据目录

本目录用于模型层改造的数据产物。

- `sft_train.jsonl`：SFT 训练集，禁止包含 held-out 用例。
- `sft_val.jsonl`：SFT 验证集，禁止包含 Phase 6 held-out-only 用例。
- `grpo_prompts.jsonl`：GRPO prompt 集，禁止包含 held-out 用例。
- `data_synth_report.md`：数据统计、schema 校验、负样本类型、教师模型状态。

大体积 JSONL 默认不进入 Git；小型统计报告可以保留。
```

- [ ] **Step 5: 跑测试**

Run:

```bash
PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer/test_validate_tool_data.py -v
```

Expected: PASS。

---

### Task 7: 教师模型数据合成入口骨架

**Files:**

- Create: `data_synth/generate_sft_data.py`
- Test: extend `tests/model_layer/test_validate_tool_data.py`

**Interfaces:**

- Consumes: `TOOL_SCHEMAS`, teacher env `CHAT_BASE_URL`/`CHAT_MODEL`/`CHAT_API_KEY`
- Produces: `check_teacher_available() -> dict`, no full dataset claim without teacher

- [ ] **Step 1: 添加 teacher availability 测试**

Append to `tests/model_layer/test_validate_tool_data.py`:

```python
def test_teacher_unavailable_without_chat_config(monkeypatch):
    from data_synth.generate_sft_data import check_teacher_available

    monkeypatch.delenv("CHAT_BASE_URL", raising=False)
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    monkeypatch.delenv("ARK_BASE_URL", raising=False)
    monkeypatch.delenv("ARK_CHAT_MODEL", raising=False)
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    result = check_teacher_available()

    assert result["available"] is False
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer/test_validate_tool_data.py::test_teacher_unavailable_without_chat_config -v
```

Expected: FAIL，`generate_sft_data` 不存在。

- [ ] **Step 3: 创建 `data_synth/generate_sft_data.py`**

```python
import json
import os
from pathlib import Path
from typing import Dict

from data_synth.tool_schemas import build_tool_schemas

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "model_training"


def check_teacher_available() -> Dict[str, object]:
    base_url = os.getenv("CHAT_BASE_URL") or os.getenv("ARK_BASE_URL")
    model = os.getenv("CHAT_MODEL") or os.getenv("ARK_CHAT_MODEL")
    api_key = os.getenv("CHAT_API_KEY") or os.getenv("ARK_API_KEY")
    available = bool(base_url and model and api_key)
    return {
        "available": available,
        "base_url_configured": bool(base_url),
        "model_configured": bool(model),
        "api_key_configured": bool(api_key),
    }


def write_pilot_report() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    teacher = check_teacher_available()
    path = OUT_DIR / "data_synth_report.md"
    lines = [
        "# 数据合成状态报告",
        "",
        f"- 教师模型可用：{teacher['available']}",
        f"- base_url 配置：{teacher['base_url_configured']}",
        f"- model 配置：{teacher['model_configured']}",
        f"- api_key 配置：{teacher['api_key_configured']}",
        f"- 工具 schema 数：{len(build_tool_schemas())}",
        "",
        "没有可用教师模型时，本阶段只完成 schema 和 validator，不声明数据合成就绪。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    path = write_pilot_report()
    print(json.dumps({"report": str(path), "teacher": check_teacher_available()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试和报告生成**

Run:

```bash
PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer/test_validate_tool_data.py -v
PYTHONPATH=backend:. .venv/bin/python data_synth/generate_sft_data.py
```

Expected: tests PASS；生成 `data/model_training/data_synth_report.md`。

---

### Task 8: SFT 训练依赖与配置模板

**Files:**

- Create: `requirements-train.txt`
- Create: `training/sft/README.md`
- Create: `training/sft/qlora_sft_config.yaml`
- Create: `docs/model_layer_phase3_sft_report.md`

**Interfaces:**

- Consumes: `data/model_training/sft_train.jsonl`, `data/model_training/sft_val.jsonl`
- Produces: documented QLoRA SFT configuration

- [ ] **Step 1: 创建训练依赖文件**

Create `requirements-train.txt`:

```txt
torch
transformers
datasets
accelerate
peft
trl
bitsandbytes
sentencepiece
protobuf
```

- [ ] **Step 2: 创建 SFT 配置**

Create `training/sft/qlora_sft_config.yaml`:

```yaml
model:
  base_model_path: "models/Qwen2.5-7B-Instruct"
  output_dir: "checkpoints/sft"

data:
  train_file: "data/model_training/sft_train.jsonl"
  val_file: "data/model_training/sft_val.jsonl"
  max_seq_len: 4096

qlora:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

train:
  learning_rate: 0.0002
  epochs: 3
  warmup_ratio: 0.03
  gradient_checkpointing: true
```

- [ ] **Step 3: 创建 SFT README**

Create `training/sft/README.md`:

```markdown
# QLoRA SFT

SFT 只能在 Phase 1 原生 Qwen vLLM 基线完成后启动。

输入：

- `data/model_training/sft_train.jsonl`
- `data/model_training/sft_val.jsonl`

输出：

- `checkpoints/sft/`
- `docs/model_layer_phase3_sft_report.md`

约束：

- 4-bit NF4 QLoRA
- V100 compute dtype 使用 float16，不使用 bfloat16
- LoRA r=16, alpha=32, dropout=0.05
- max_seq_len 初始为 4096
- gradient_checkpointing=true
```

- [ ] **Step 4: 创建 SFT 报告模板**

Create `docs/model_layer_phase3_sft_report.md`:

```markdown
# Phase 3 QLoRA SFT 报告

## 前置条件

- Phase 1 原生 Qwen 基线：未填写前不得启动正式 SFT。
- SFT 数据集规模：记录 train/val 条数和每 intent 数量。

## 训练配置

- base model：`models/Qwen2.5-7B-Instruct`
- LoRA：r=16, alpha=32, dropout=0.05
- max_seq_len=4096

## 评估

- 工具选择准确率：
- 参数合法率：
- 应用评估相对 Phase 1 是否回退：

## 结论

未训练时不得填写模型提升结论。
```

- [ ] **Step 5: 验证文件存在**

Run:

```bash
test -s requirements-train.txt
test -s training/sft/qlora_sft_config.yaml
test -s training/sft/README.md
test -s docs/model_layer_phase3_sft_report.md
```

Expected: all exit code 0。

---

### Task 9: GRPO 奖励函数骨架、缓存与 held-out 防线

**Files:**

- Create: `training/grpo/reward_fn.py`
- Create: `training/grpo/train_grpo.py`
- Create: `training/grpo/grpo_config.yaml`
- Create: `training/grpo/README.md`
- Create: `tests/model_layer/test_reward_fn.py`
- Modify: `docs/model_layer_phase4_grpo_report.md`

**Interfaces:**

- Consumes: Phase 0 reward split, model completions
- Produces: `RewardContext`, `compute_format_reward()`, `compute_tool_execution_reward()`, `RewardCache`

- [ ] **Step 1: 写 reward 基础测试**

Create `tests/model_layer/test_reward_fn.py`:

```python
from training.grpo.reward_fn import RewardCache, compute_format_reward


def test_format_reward_rejects_unknown_tool():
    completion = {"tool_calls": [{"function": {"name": "unknown_tool", "arguments": "{}"}}]}

    assert compute_format_reward(completion) == 0.0


def test_reward_cache_uses_stable_key():
    cache = RewardCache()
    cache.set("prompt", "completion", {"score": 0.5})

    assert cache.get("prompt", "completion") == {"score": 0.5}
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer/test_reward_fn.py -v
```

Expected: FAIL，`training.grpo.reward_fn` 不存在。

- [ ] **Step 3: 实现 `training/grpo/reward_fn.py`**

```python
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from data_synth.tool_schemas import TOOL_SCHEMAS

ALLOWED_TOOLS = {item["function"]["name"] for item in TOOL_SCHEMAS}


@dataclass
class RewardCache:
    values: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def key(self, prompt: str, completion: str) -> str:
        payload = json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, prompt: str, completion: str) -> Optional[Dict[str, Any]]:
        return self.values.get(self.key(prompt, completion))

    def set(self, prompt: str, completion: str, value: Dict[str, Any]) -> None:
        self.values[self.key(prompt, completion)] = value


def compute_format_reward(completion: Dict[str, Any]) -> float:
    calls = completion.get("tool_calls") or []
    if not calls:
        return 0.0
    for call in calls:
        function = call.get("function", {})
        if function.get("name") not in ALLOWED_TOOLS:
            return 0.0
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return 0.0
        elif not isinstance(arguments, dict):
            return 0.0
    return 1.0


def compute_tool_execution_reward(completion: Dict[str, Any]) -> float:
    if compute_format_reward(completion) == 0.0:
        return 0.0
    return 0.5


def compute_answer_reward(prompt_id: str, completion: Dict[str, Any], reward_visible_ids: set[str]) -> float:
    if prompt_id not in reward_visible_ids:
        return 0.0
    return 0.0
```

- [ ] **Step 4: 创建 GRPO 配置和 README**

Create `training/grpo/grpo_config.yaml`:

```yaml
model:
  sft_adapter_path: "checkpoints/sft"
  output_dir: "checkpoints/grpo"

rollout:
  backend: "vllm"
  base_url: "http://127.0.0.1:8003/v1"
  model: "qwen7b-sft"
  num_generations: 8
  temperature: 0.8

kl:
  beta: 0.0
  strategy: "remove_kl_initially_enable_small_beta_if_needed"
  variance_control: "clip-higher"

memory:
  single_gpu_vram_gb: 32
  vllm_gpu_memory_utilization: "0.4-0.5"
  fallback: "generation_training_alternating"

reward:
  format_reward_cap: 1.0
  answer_reward_primary: true
  cache_enabled: true
```

Create `training/grpo/README.md`:

```markdown
# GRPO/RLVR

GRPO 只能在 SFT 产生可度量改进后启动。

硬约束：

- rollout 使用 vLLM backend
- num_generations >= 8
- temperature 0.7-1.0
- KL 使用小 beta 或去 KL
- 使用 clip-higher 或等价高方差控制
- held-out cases 不得进入 reward、调参或 early stopping

如果 reward 计算慢于 rollout，必须启用缓存、批处理或离线 reward scoring。
```

Create `training/grpo/train_grpo.py`:

```python
def main() -> None:
    raise SystemExit(
        "GRPO training is gated. Complete Phase 0 reward split and Phase 3 SFT report before enabling training."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 创建 Phase 4 报告模板**

Create `docs/model_layer_phase4_grpo_report.md`:

```markdown
# Phase 4 GRPO/RLVR 报告

## 前置门禁

- Phase 0 reward-compatible / held-out 划分：
- Phase 3 SFT 改进证据：

## Rollout

- backend：vLLM
- num_generations：
- temperature：

## Reward

- answer reward：
- format reward：
- tool execution reward：
- reward cache：
- reward 吞吐：

## 显存策略

- 同卡并发 / 分时执行：
- OOM 处理：

## Held-out 隔离确认

held-out cases 未用于 reward、调参、early stopping。

## 结论

未完成 held-out 对比前，不得声明最终模型提升。
```

- [ ] **Step 6: 跑测试**

Run:

```bash
PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer/test_reward_fn.py -v
```

Expected: PASS。

---

### Task 10: SFT/GRPO vLLM 启动脚本

**Files:**

- Create: `scripts/start_vllm_sft.sh`
- Create: `scripts/start_vllm_grpo.sh`
- Extend: `tests/model_layer/test_vllm_scripts.py`

**Interfaces:**

- Consumes: `checkpoints/sft`, `checkpoints/grpo`
- Produces: reproducible vLLM startup scripts for later adapters

- [ ] **Step 1: 扩展测试**

Append to `tests/model_layer/test_vllm_scripts.py`:

```python
def test_adapter_vllm_scripts_do_not_silently_download():
    for path in ["scripts/start_vllm_sft.sh", "scripts/start_vllm_grpo.sh"]:
        script = Path(path).read_text(encoding="utf-8")
        assert "--enable-auto-tool-choice" in script
        assert "--tool-call-parser hermes" in script
        assert "exit 1" in script
        assert "checkpoints/" in script
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
.venv/bin/python -m pytest tests/model_layer/test_vllm_scripts.py::test_adapter_vllm_scripts_do_not_silently_download -v
```

Expected: FAIL，脚本不存在。

- [ ] **Step 3: 创建 `scripts/start_vllm_sft.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/Qwen2.5-7B-Instruct}"
LORA_PATH="${LORA_PATH:-${ROOT_DIR}/checkpoints/sft}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "本地模型路径不存在：${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -d "${LORA_PATH}" ]]; then
  echo "SFT adapter 不存在：${LORA_PATH}" >&2
  exit 1
fi

exec vllm serve "${MODEL_PATH}" \
  --host "${VLLM_HOST:-127.0.0.1}" \
  --port "${VLLM_PORT:-8001}" \
  --served-model-name qwen7b-sft \
  --enable-lora \
  --lora-modules "qwen7b-sft=${LORA_PATH}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 8192 \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
```

- [ ] **Step 4: 创建 `scripts/start_vllm_grpo.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/Qwen2.5-7B-Instruct}"
LORA_PATH="${LORA_PATH:-${ROOT_DIR}/checkpoints/grpo}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "本地模型路径不存在：${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -d "${LORA_PATH}" ]]; then
  echo "GRPO adapter 不存在：${LORA_PATH}" >&2
  exit 1
fi

exec vllm serve "${MODEL_PATH}" \
  --host "${VLLM_HOST:-127.0.0.1}" \
  --port "${VLLM_PORT:-8002}" \
  --served-model-name qwen7b-grpo \
  --enable-lora \
  --lora-modules "qwen7b-grpo=${LORA_PATH}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 8192 \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
```

Run:

```bash
chmod +x scripts/start_vllm_sft.sh scripts/start_vllm_grpo.sh
```

- [ ] **Step 5: 跑测试**

Run:

```bash
.venv/bin/python -m pytest tests/model_layer/test_vllm_scripts.py -v
```

Expected: PASS。

---

### Task 11: Phase 6 对比报告模板与 README 更新门禁

**Files:**

- Create: `docs/model_layer_report.md`
- Modify: `README.md`

**Interfaces:**

- Consumes: Phase 0 held-out split, Phase 1/3/4 reports
- Produces: final four-column report template and README model-layer section

- [ ] **Step 1: 创建最终报告模板**

Create `docs/model_layer_report.md`:

```markdown
# 模型层最终对比报告

## 评估隔离

- reward-visible cases：
- held-out cases：
- 本报告主要结论只基于 held-out cases。

## 四列对比

| 后端 | 工具选择准确率 | 参数合法率 | 推荐命中率 | 幻觉率 | 端到端时延 | agent regression | release gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Ark baseline | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 |
| 原生 Qwen2.5-7B via vLLM | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 |
| QLoRA SFT Qwen2.5-7B | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 |
| GRPO Qwen2.5-7B | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 |

## 结论规则

未完成 held-out 四列对比前，不声明本地模型优于云端或 SFT/GRPO 有最终提升。
```

- [ ] **Step 2: 更新 README 的模型层规划小节**

Add a concise section to `README.md`:

```markdown
## 模型层改造规划

项目当前应用层已经具备 LangGraph 多智能体、5 个真实业务工具、本地 RAG、Obsidian 长期知识库和评估脚本。后续模型层改造按 `docs/superpowers/specs/2026-07-22-model-layer-transformation-design.md` 推进：

1. 先记录当前基线。
2. 再用本地 `Qwen2.5-7B-Instruct` 通过 vLLM 接入现有 Agent。
3. 之后进行工具调用数据合成、QLoRA SFT 和 GRPO/RLVR。
4. 最终只用 held-out 评估集声明模型提升。

当前真实数字以代码为准：5 个工具、实际知识文档数量、实际车辆库数量。不得为了展示虚报。
```

- [ ] **Step 3: 检查 README 中没有新增虚假数字**

Run:

```bash
rg -n "7 个工具|50 篇|虚报" README.md docs/model_layer_report.md || true
```

Expected: 不应出现 “7 个工具” 或 “50 篇”。

---

### Task 12: 脚手架阶段验证与记录

**Files:**

- Modify: `docs/model_layer_progress_log.md`

**Interfaces:**

- Consumes: Tasks 1-11
- Produces: scaffold verification summary; this task does not complete the model-layer project

- [ ] **Step 1: 运行 Python 单测**

Run:

```bash
PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer -v
```

Expected: all PASS。

- [ ] **Step 2: 运行现有应用层核心评估**

Run:

```bash
PYTHONPATH=backend .venv/bin/python scripts/evaluate_agent_regression.py
PYTHONPATH=backend .venv/bin/python scripts/evaluate_knowledge_fusion.py
PYTHONPATH=backend .venv/bin/python scripts/evaluate_release_gate.py
```

Expected:

- `evaluate_agent_regression.py` PASS。
- `evaluate_knowledge_fusion.py` PASS。
- `evaluate_release_gate.py` 可以 blocked，但必须记录 blocker 原因。

- [ ] **Step 3: 后端 import 与 health smoke**

Run:

```bash
PYTHONPATH=backend .venv/bin/python - <<'PY'
from fastapi.testclient import TestClient
from app.main import app
with TestClient(app) as client:
    response = client.get("/api/health")
    print(response.status_code)
    print(response.json())
    raise SystemExit(0 if response.status_code == 200 else 1)
PY
```

Expected: status code 200。

- [ ] **Step 4: 检查大文件与私密配置未被要求提交**

Run:

```bash
test ! -f backend/config/config.yaml || echo "config.yaml exists locally; do not commit"
test -d models/Qwen2.5-7B-Instruct && echo "local model exists; do not commit models/"
```

Expected: local warnings only。

- [ ] **Step 5: 更新进度日志**

Append:

```markdown
## Model layer scaffold verification

- tests/model_layer：
- agent_regression：
- knowledge_fusion：
- release_gate：
- health smoke：
- Git 状态：
- 下一步：
```

- [ ] **Step 6: Commit 或记录无法 commit**

If Git works:

```bash
git add backend/app/config.py backend/app/services/llm_client.py backend/app/main.py backend/config/config.vllm.example.yaml scripts data_synth training tests docs README.md requirements-train.txt
git commit -m "feat: scaffold model layer transformation"
```

If Git is unavailable, append:

```markdown
无法 commit：当前 `/home/yanhongliang/workspace/Agentic_Agent` 不是 Git 仓库。所有改动已保留在工作区并记录于本日志。
```

Task 12 只验证应用/config/数据/训练脚手架，不得将 Phase 2、Phase 3、Phase 4 或 Phase 6 标记为完成。

---

### Task 13: V100 训练与 vLLM 环境兼容门禁

**Files:**

- Create: `scripts/check_model_runtime.py`
- Create: `docs/model_runtime_compatibility.md`
- Create: `tests/model_layer/test_model_runtime.py`

**Interfaces:**

- Consumes: local GPU/driver/Python/model path state
- Produces: `collect_runtime_facts() -> dict`, `evaluate_runtime_gates(facts: dict) -> dict`

- [ ] **Step 1: 写失败测试**

Create `tests/model_layer/test_model_runtime.py`:

```python
from scripts.check_model_runtime import evaluate_runtime_gates


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

    assert result["training_compute_dtype"] == "float16"
    assert result["bf16_allowed"] is False
    assert result["service_env_can_install_vllm"] is False
    assert result["vllm_status"] == "requires_compatible_isolated_runtime"
```

- [ ] **Step 2: 运行 RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/model_layer/test_model_runtime.py -v
```

Expected: FAIL because `scripts.check_model_runtime` does not exist.

- [ ] **Step 3: 实现运行时事实与门禁**

`scripts/check_model_runtime.py` must collect GPU name, driver, reported CUDA driver version, memory, local model path, and model size. `evaluate_runtime_gates()` must:

- select `float16` for V100;
- reject BF16 as default;
- require a separate vLLM/training environment;
- prohibit installing CUDA training packages into service `.venv`;
- report compatibility as a gate, not silently install packages.

- [ ] **Step 4: 运行 GREEN 和环境探针**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/model_layer/test_model_runtime.py -v
PYTHONPATH=. .venv/bin/python scripts/check_model_runtime.py
```

Expected: test PASS; output records V100, driver, CUDA driver compatibility, model path, and FP16 gate.

- [ ] **Step 5: 写兼容性报告**

`docs/model_runtime_compatibility.md` must record exact facts and require all of:

1. PyTorch detects CUDA and V100.
2. bitsandbytes loads 4-bit CUDA kernels.
3. vLLM loads local Qwen2.5-7B and completes a tool-call smoke.
4. Service `.venv` remains free of vLLM/training CUDA dependencies.

---

### Task 14: 独立模型质量评测协议

**Files:**

- Create: `data/model_training/eval/reward_visible.jsonl`
- Create: `data/model_training/eval/held_out.jsonl`
- Create: `scripts/evaluate_model_outputs.py`
- Create: `scripts/run_model_layer_eval.py`
- Create: `tests/model_layer/test_model_output_evaluation.py`

**Interfaces:**

- Consumes: JSONL model cases and model output JSONL
- Produces: deterministic metrics: tool selection accuracy, argument validity, recommendation hit rate, hallucination rate

- [ ] **Step 1: 写评测器失败测试**

The test must prove one exact-match case scores:

- tool selection accuracy `1.0`;
- argument validity `1.0`;
- recommendation hit rate `1.0`;
- hallucination rate `0.0`.

- [ ] **Step 2: 运行 RED**

Run:

```bash
PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer/test_model_output_evaluation.py -v
```

Expected: FAIL because evaluator does not exist.

- [ ] **Step 3: 实现确定性评测器**

`scripts/evaluate_model_outputs.py` must:

- join cases and outputs by exact ID;
- reject missing/extra/mismatched IDs;
- validate tool names and argument object types against the five schemas;
- compute exact numerators, denominators, and percentages;
- preserve per-case failures for audit.

- [ ] **Step 4: 创建互斥数据集**

Create at least 20 reward-visible and 40 held-out cases. Every row must include:

```json
{
  "id": "heldout-family-001",
  "query": "预算25万，无家充，三口之家，需要SUV",
  "intent": "recommend",
  "expected_tools": ["extract_user_profile", "search_and_rank_vehicles", "retrieve_knowledge_base"],
  "allowed_models": ["比亚迪 宋PLUS DM-i", "理想 L6", "问界 M7"]
}
```

IDs and normalized queries must be disjoint between files.

- [ ] **Step 5: 实现 OpenAI-compatible 推理采集**

`scripts/run_model_layer_eval.py` must load one evaluation JSONL, pass the exact five tool schemas, save raw assistant text/tool calls/latency/recommended models, resume by case ID, and never modify source cases.

- [ ] **Step 6: 运行 GREEN、隔离和 CLI 检查**

Run:

```bash
PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer/test_model_output_evaluation.py -v
PYTHONPATH=backend:. .venv/bin/python scripts/evaluate_model_outputs.py --help
PYTHONPATH=backend:. .venv/bin/python scripts/run_model_layer_eval.py --help
```

Expected: PASS and both CLIs return 0.

---

### Task 15: 全量教师数据合成与审计

**Files:**

- Modify: `data_synth/generate_sft_data.py`
- Modify: `data_synth/validate_tool_data.py`
- Create: `tests/model_layer/test_sft_generation.py`
- Create: `data/model_training/data_synth_report.md`

**Interfaces:**

- Consumes: teacher OpenAI-compatible endpoint, real five tools, local vehicle database
- Produces: validated `sft_train.jsonl`, `sft_val.jsonl`, `grpo_prompts.jsonl`

- [ ] **Step 1: TDD 定义可恢复、分层和去重行为**

Tests must cover stable IDs, resume without duplicates, held-out leakage rejection, per-intent quotas, required real tool results, and hallucinated vehicle rejection.

- [ ] **Step 2: 实现 pilot 生成并人工审计 50 条**

Run teacher synthesis for 20 samples per intent first. Validate and manually audit 50 total rows before scaling.

- [ ] **Step 3: 扩展到每个主要 intent 至少 500 条正样本**

Generate `recommend`, `compare`, `knowledge`, and `sales` strata with template/temperature variation. Invalid or duplicate rows do not count.

- [ ] **Step 4: 写数据报告**

Report exact numerator/denominator/percentage for generated/accepted/rejected, schema-valid, tool-executable, hallucination-free, quotas, negative categories, and held-out leakage.

- [ ] **Step 5: 门禁**

Complete only when each intent has at least 500 accepted positive rows, parameter validity is 100%, held-out leakage is 0, and 50/50 manual audit rows are grounded.

---

### Task 16: V100 FP16 QLoRA SFT 真实训练

**Files:**

- Create: `training/sft/train_sft.py`
- Modify: `training/sft/qlora_sft_config.yaml`
- Modify: `requirements-train.txt`
- Create: `tests/model_layer/test_sft_config.py`
- Modify: `docs/model_layer_phase3_sft_report.md`

**Interfaces:**

- Consumes: validated SFT train/val JSONL and local Qwen base
- Produces: real PEFT adapter under `checkpoints/sft`

- [ ] **Step 1: TDD 锁定 V100 配置**

Tests must assert FP16, BF16 disabled, NF4 double quant, exact target modules, output path, and validated data files.

- [ ] **Step 2: 在独立训练环境安装并验证 CUDA 栈**

Do not use service `.venv`. Record exact Python/PyTorch/CUDA/bitsandbytes/transformers/peft/trl versions. Verify CUDA sees Tesla V100 and bitsandbytes can load a 4-bit model.

- [ ] **Step 3: 运行小规模过拟合试验**

Use 32-64 rows to prove loss decreases, adapter saves/reloads, and generated tool calls parse.

- [ ] **Step 4: 运行全量 SFT**

Use all accepted training data. Save adapter, tokenizer/config, trainer state, loss/eval history, exact command, and environment manifest.

- [ ] **Step 5: 真实 adapter reload 与验证推理**

Load base + adapter through PEFT and generate on validation prompts. No mock path is allowed when CUDA is available.

---

### Task 17: SFT 晋级门禁

**Files:**

- Create: `scripts/evaluate_sft_gate.py`
- Create: `tests/model_layer/test_sft_gate.py`
- Modify: `docs/model_layer_phase3_sft_report.md`

**Interfaces:**

- Consumes: native Qwen and SFT outputs on reward-visible and held-out datasets
- Produces: `promote_to_grpo: bool`

- [ ] **Step 1: TDD 定义门禁**

Require all:

- parse validity >= 98%;
- parameter validity >= native Qwen and >= 98%;
- held-out tool selection improves by at least 5 percentage points;
- held-out recommendation hit rate does not regress;
- held-out hallucination rate does not increase;
- adapter reload and CUDA generation succeeded.

- [ ] **Step 2: 运行原生 Qwen 与 SFT 的相同 held-out 推理**

Use `scripts/run_model_layer_eval.py` and preserve raw outputs.

- [ ] **Step 3: 生成门禁报告**

Show numerator/denominator/percentage for every metric. Any failure sets `promote_to_grpo=false` and stops GRPO.

---

### Task 18: 条件式 GRPO/RLVR 真实训练

**Files:**

- Modify: `training/grpo/reward_fn.py`
- Replace gated stub: `training/grpo/train_grpo.py`
- Modify: `training/grpo/grpo_config.yaml`
- Create: `tests/model_layer/test_grpo_gate.py`
- Modify: `docs/model_layer_phase4_grpo_report.md`

**Interfaces:**

- Consumes: passing SFT gate, reward-visible prompts, SFT adapter
- Produces: real GRPO adapter only when gate passes

- [ ] **Step 1: 强制读取 SFT 门禁**

Training must refuse to start unless `promote_to_grpo=true`.

- [ ] **Step 2: 实现确定性 reward 与缓存**

Held-out IDs and normalized queries must be rejected. Track answer/format/tool-execution reward separately.

- [ ] **Step 3: 先做 8-generation rollout smoke**

Use vLLM only if runtime compatibility passes. On one V100, use a verified colocate/sleep or generation-training alternating strategy and record it.

- [ ] **Step 4: 运行 GRPO**

Use reward-visible prompts only. Record reward curves, KL/beta, clip strategy, rollout/reward throughput, GPU memory, and adapter output.

- [ ] **Step 5: 独立 held-out 验证**

Complete only when a real adapter reloads and Phase 6 held-out inference succeeds. Reward-set gains alone do not count.

---

### Task 19: 四后端 held-out 对比与最终报告

**Files:**

- Modify: `docs/model_layer_report.md`
- Create: `scripts/compare_model_backends.py`
- Create: `tests/model_layer/test_compare_model_backends.py`
- Modify: `README.md`

**Interfaces:**

- Consumes: Ark, native Qwen, SFT, optional promoted GRPO raw held-out outputs
- Produces: four-column comparison with exact counts and percentages

- [ ] **Step 1: TDD 定义统一汇总**

Reject mismatched case IDs and reward-visible outputs in the final table.

- [ ] **Step 2: 采集所有可用后端输出**

Use the exact same held-out file. If GRPO was not promoted, show `未执行（SFT门禁未通过）`, never fabricate a score.

- [ ] **Step 3: 生成最终报告**

For every metric show numerator, denominator, percentage, latency distribution, environment, model identity, adapter checksum, and raw output path.

- [ ] **Step 4: 更新 README**

State only measured facts: five tools, actual knowledge documents, actual vehicles, and actual model results.

---

### Task 20: 全量验证、最终代码审查与交付记录

**Files:**

- Modify: `docs/model_layer_progress_log.md`
- Modify: `.superpowers/sdd/progress.md`

**Interfaces:**

- Consumes: all implementation and review artifacts
- Produces: final verified state and explicit blockers

- [ ] **Step 1: 运行完整单测、应用评估和健康检查**

- [ ] **Step 2: 校验所有报告的分子/分母/百分比和 held-out 隔离声明**

- [ ] **Step 3: 对全部快照 diff 做最终广域代码审查**

- [ ] **Step 4: 修复 Critical/Important findings 并复审**

- [ ] **Step 5: 记录已完成项、未执行项、外部阻断和复现命令**

## Self-Review

Spec coverage:

Spec coverage:

- Phase 0 基线、reward/held-out 划分和教师模型检查由 Task 1 覆盖。
- Phase 1 vLLM tool calling、配置样例和 tool-call 检查由 Task 2-4 覆盖。
- Phase 2 工具 schema、数据校验、负样本和教师模型依赖由 Task 5-7 覆盖。
- Phase 3 SFT 训练依赖与配置由 Task 8 覆盖。
- Phase 4 GRPO rollout/reward/KL/显存/held-out 防线由 Task 9-10 覆盖。
- Phase 5 后端切换由 Task 4 和 Task 10 覆盖。
- Task 12 只验证脚手架，不代表训练或最终交付完成。
- V100 FP16 与独立 vLLM/训练环境门禁由 Task 13 覆盖。
- 独立模型质量 reward-visible/held-out 协议由 Task 14 覆盖。
- 全量教师数据合成与审计由 Task 15 覆盖。
- 真实 QLoRA SFT 与晋级门禁由 Task 16-17 覆盖。
- GRPO 仅在 SFT 门禁通过时由 Task 18 执行。
- Phase 6 四后端 held-out 对比和文档真实性由 Task 19 覆盖。
- 全量验证、最终审查和交付记录由 Task 20 覆盖。

Placeholder scan:

- 本计划不使用 TBD/TODO/以后补等占位。
- 训练脚本明确保留门禁退出，不伪装成可训练完成状态。

Type consistency:

- `build_backend_env(name: str) -> Dict[str, str]` 在测试和实现中一致。
- `build_tool_schemas() -> List[Dict]` 在测试和实现中一致。
- `validate_record(record: Dict, held_out_ids: Set[str]) -> List[str]` 在测试和实现中一致。
- `RewardCache.get/set` 在测试和实现中一致。
