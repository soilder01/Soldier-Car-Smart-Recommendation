# 模型层改造 Phase 0 基线报告

生成时间：2026-07-23 09:49:38

## 评估脚本划分

- reward-compatible：无
- report-only：无
- engineering-gate：evaluate_agent_regression.py, evaluate_knowledge_fusion.py, evaluate_release_gate.py

## 教师模型可用性

- 状态：missing
- 说明：Phase 2 requires a complete CHAT, ARK, or legacy OPENAI teacher configuration.

## 检查结果

- backend_import：command_status=success，命令 `bash -lc PYTHONPATH=backend .venv/bin/python - <<'PY'
import app.main
print('app.main OK')
PY`
- health_testclient：command_status=success，命令 `bash -lc PYTHONPATH=backend .venv/bin/python - <<'PY'
from fastapi.testclient import TestClient
from app.main import app
with TestClient(app) as client:
    response = client.get('/api/health')
    print(response.status_code)
    print(response.json())
    raise SystemExit(0 if response.status_code == 200 else 1)
PY`
- agent_regression：command_status=success，命令 `bash -lc PYTHONPATH=backend .venv/bin/python scripts/evaluate_agent_regression.py`
- knowledge_fusion：command_status=success，命令 `bash -lc PYTHONPATH=backend .venv/bin/python scripts/evaluate_knowledge_fusion.py`
- release_gate：command_status=success，business_status=blocked，release_allowed=false，门禁：4/6，命令 `bash -lc PYTHONPATH=backend .venv/bin/python scripts/evaluate_release_gate.py`

## 评估脚本副作用

### overwritten

- `data/real_world/agent_regression_evaluation.json`
- `data/real_world/agent_regression_evaluation.md`
- `data/real_world/knowledge_fusion_evaluation.json`
- `data/real_world/knowledge_fusion_evaluation.md`
- `data/real_world/real_data_governance_report.json`
- `data/real_world/release_gate_evaluation.json`
- `data/real_world/release_gate_evaluation.md`

### created

- `obsidian-vault/07-测试样例/Agent端到端回归评估-20260723-094933.md`
- `obsidian-vault/08-推荐案例/推荐案例-20260723-094933-Hyundai-IONIQ-5-Base.md`
- `obsidian-vault/08-推荐案例/推荐案例-20260723-094933-智界-R7.md`
- `obsidian-vault/08-推荐案例/推荐案例-20260723-094933-比亚迪-宋PLUS-DM-i.md`
- `obsidian-vault/08-推荐案例/推荐案例-20260723-094933-理想-L6.md`
- `obsidian-vault/08-推荐案例/推荐案例-20260723-094938-Hyundai-IONIQ-5-Base.md`
- `obsidian-vault/08-推荐案例/推荐案例-20260723-094938-智界-R7.md`
- `obsidian-vault/08-推荐案例/推荐案例-20260723-094938-比亚迪-宋PLUS-DM-i.md`
- `obsidian-vault/08-推荐案例/推荐案例-20260723-094938-理想-L6.md`
