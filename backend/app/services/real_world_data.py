import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from app.config import DATA_DIR
from app.services.real_data_governance import load_real_data_governance

REAL_DIR = DATA_DIR / "real_world"
QUALITY_REPORT = REAL_DIR / "real_data_quality_report.json"
EVALUATION_REPORT = REAL_DIR / "real_recommendation_evaluation.json"
ENRICHMENT_REPORT = REAL_DIR / "real_field_enrichment_report.json"
REAL_SPECS_CSV = REAL_DIR / "real_ev_specs.csv"
RECOMMENDER_CSV = REAL_DIR / "real_ev_specs_for_recommender.csv"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, limit: int = 500) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))[:limit]


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0


def real_world_overview(limit: int = 30) -> Dict[str, Any]:
    quality = _read_json(QUALITY_REPORT)
    enrichment = _read_json(ENRICHMENT_REPORT)
    evaluation = _read_json(EVALUATION_REPORT)
    rows = _read_csv(REAL_SPECS_CSV, limit=500)
    sample_rows = rows[:limit]
    range_values = [_number(row.get("range_km")) for row in rows if _number(row.get("range_km")) > 0]
    battery_values = [_number(row.get("battery_kwh")) for row in rows if _number(row.get("battery_kwh")) > 0]
    dc_values = [_number(row.get("dc_charge_kw")) for row in rows if _number(row.get("dc_charge_kw")) > 0]
    return {
        "quality": quality,
        "enrichment": enrichment,
        "governance": load_real_data_governance(),
        "evaluation": {
            "generated_at": evaluation.get("generated_at"),
            "case_count": evaluation.get("case_count", 0),
            "passed_count": evaluation.get("passed_count", 0),
            "pass_rate": evaluation.get("pass_rate", 0),
            "cases": evaluation.get("cases", []),
        },
        "stats": {
            "record_count": len(rows) or quality.get("record_count", 0),
            "avg_range_km": round(sum(range_values) / len(range_values), 1) if range_values else 0,
            "max_range_km": max(range_values) if range_values else 0,
            "avg_battery_kwh": round(sum(battery_values) / len(battery_values), 1) if battery_values else 0,
            "avg_dc_charge_kw": round(sum(dc_values) / len(dc_values), 1) if dc_values else 0,
            "brand_distribution": Counter(row.get("brand", "未知") for row in rows).most_common(12),
            "vehicle_type_distribution": Counter(row.get("vehicle_type", "未知") for row in rows).most_common(),
            "year_distribution": Counter(row.get("model_year", "未知") for row in rows).most_common(12),
        },
        "samples": sample_rows,
        "files": quality.get("files", {}),
    }
