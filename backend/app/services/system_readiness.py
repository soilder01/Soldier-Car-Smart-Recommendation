from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from app.config import CONFIG_DIR, DATA_DIR, DB_PATH, OPENAI_API_KEY, PROJECT_DIR, VEHICLE_CSV, VECTOR_DIR
from app.database import list_vehicles
from app.services.rag import rag_service


def _file_check(name: str, path: Path, required: bool = True) -> Dict[str, Any]:
    exists = path.exists()
    size = path.stat().st_size if exists and path.is_file() else 0
    passed = exists if required else True
    return {"name": name, "path": str(path.relative_to(PROJECT_DIR)), "exists": exists, "size": size, "passed": passed}


def system_readiness() -> Dict[str, Any]:
    files = [
        _file_check("车型主库", VEHICLE_CSV),
        _file_check("SQLite运行库", DB_PATH),
        _file_check("配置样例", CONFIG_DIR / "config.example.yaml"),
        _file_check("真实数据候选库", DATA_DIR / "real_world" / "real_ev_specs.csv"),
        _file_check("真实数据治理报告", DATA_DIR / "real_world" / "real_data_governance_report.json", False),
        _file_check("前端构建入口", PROJECT_DIR / "frontend" / "dist" / "index.html", False),
    ]
    vehicle_count = len(list_vehicles())
    rag_stats = rag_service.stats()
    checks = [
        {"name": "后端数据目录", "passed": DATA_DIR.exists(), "detail": str(DATA_DIR.relative_to(PROJECT_DIR))},
        {"name": "车型库可读取", "passed": vehicle_count > 0, "detail": f"{vehicle_count} 条车型"},
        {"name": "RAG索引可用", "passed": rag_stats.get("chunks", 0) > 0, "detail": f"{rag_stats.get('chunks', 0)} chunks"},
        {"name": "LLM密钥配置", "passed": bool(OPENAI_API_KEY), "detail": "已配置" if OPENAI_API_KEY else "未配置，保留规则兜底"},
        {"name": "向量目录存在", "passed": VECTOR_DIR.exists(), "detail": str(VECTOR_DIR.relative_to(PROJECT_DIR))},
    ] + [{"name": item["name"], "passed": item["passed"], "detail": item["path"]} for item in files]
    passed = sum(1 for item in checks if item["passed"])
    readiness_score = round(passed / len(checks) * 100, 1)
    risks = []
    if not OPENAI_API_KEY:
        risks.append({"level": "P1", "title": "LLM密钥未配置", "action": "生产部署前通过 backend/config/config.yaml 或环境变量配置 ARK_API_KEY"})
    if not (PROJECT_DIR / "frontend" / "dist" / "index.html").exists():
        risks.append({"level": "P2", "title": "前端未构建", "action": "部署前执行 npm run build"})
    if readiness_score < 100:
        risks.append({"level": "P2", "title": "工程化检查未满分", "action": "优先修复未通过检查项后再发布"})
    return {"summary": {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "check_count": len(checks), "passed_count": passed, "readiness_score": readiness_score, "vehicle_count": vehicle_count, "rag_chunks": rag_stats.get("chunks", 0)}, "checks": checks, "files": files, "risks": risks}
