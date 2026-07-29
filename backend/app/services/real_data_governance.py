import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from app.config import DATA_DIR

REAL_DIR = DATA_DIR / "real_world"
REAL_SPECS_CSV = REAL_DIR / "real_ev_specs.csv"
GOVERNANCE_REPORT = REAL_DIR / "real_data_governance_report.json"

TRUSTED_DOMAINS = {
    "audi.com", "bmw.com", "hyundai.com", "kia.com", "mercedes-benz.com", "porsche.com",
    "tesla.com", "volkswagen.com", "volvo.com", "polestar.com", "ford.com", "nissan-global.com",
}
CORE_FIELDS = ["brand", "model", "model_year", "vehicle_type", "energy_type", "battery_kwh", "range_km", "source_url", "raw_path"]


def _read_csv(path: Path = REAL_SPECS_CSV, limit: int = 1000) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))[:limit]


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _source_domain(url: str) -> str:
    host = urlparse(url or "").netloc.lower().replace("www.", "")
    return host or "unknown"


def _source_trust(row: Dict[str, Any]) -> Dict[str, Any]:
    domain = _source_domain(row.get("source_url", ""))
    raw = row.get("raw_path", "")
    if any(domain.endswith(item) for item in TRUSTED_DOMAINS):
        return {"level": "high", "score": 100, "domain": domain, "reason": "官方或主机厂域名"}
    if "open-ev-data" in raw or "github" in domain:
        return {"level": "medium", "score": 82, "domain": domain, "reason": "开放数据集或代码仓来源"}
    if domain != "unknown":
        return {"level": "medium", "score": 72, "domain": domain, "reason": "可追溯外部域名"}
    return {"level": "low", "score": 45, "domain": domain, "reason": "缺少可追溯来源"}


def _duplicate_key(row: Dict[str, Any]) -> str:
    return "|".join([row.get("brand", "").strip().lower(), row.get("model", "").strip().lower(), row.get("model_year", "").strip(), row.get("trim", "").strip().lower()])


def _detect_anomalies(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues = []
    year = int(_number(row.get("model_year")))
    battery = _number(row.get("battery_kwh"))
    range_km = _number(row.get("range_km"))
    dc = _number(row.get("dc_charge_kw"))
    price = _number(row.get("price_min"))
    if year and (year < 2008 or year > datetime.now().year + 2):
        issues.append({"field": "model_year", "level": "high", "message": f"年份 {year} 超出合理范围"})
    if battery and (battery < 15 or battery > 160):
        issues.append({"field": "battery_kwh", "level": "medium", "message": f"电池容量 {battery}kWh 异常"})
    if range_km and (range_km < 80 or range_km > 1000):
        issues.append({"field": "range_km", "level": "medium", "message": f"续航 {range_km}km 异常"})
    if dc and dc > 400:
        issues.append({"field": "dc_charge_kw", "level": "low", "message": f"直流快充功率 {dc}kW 需核验"})
    if price and (price < 50000 or price > 2000000):
        issues.append({"field": "price_min", "level": "medium", "message": f"价格 {price} 元异常"})
    return issues


def generate_real_data_governance(persist: bool = True) -> Dict[str, Any]:
    rows = _read_csv()
    duplicate_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    missing_counter = Counter()
    trust_counter = Counter()
    source_counter = Counter()
    anomaly_rows = []
    trusted_scores = []
    for row in rows:
        duplicate_groups[_duplicate_key(row)].append(row)
        for field in CORE_FIELDS:
            if not str(row.get(field, "")).strip():
                missing_counter[field] += 1
        trust = _source_trust(row)
        trust_counter[trust["level"]] += 1
        source_counter[trust["domain"]] += 1
        trusted_scores.append(trust["score"])
        issues = _detect_anomalies(row)
        if issues:
            anomaly_rows.append({"brand": row.get("brand"), "model": row.get("model"), "model_year": row.get("model_year"), "issues": issues})
    duplicate_items = [
        {"key": key, "count": len(items), "brand": items[0].get("brand"), "model": items[0].get("model"), "model_year": items[0].get("model_year"), "trim": items[0].get("trim")}
        for key, items in duplicate_groups.items() if len(items) > 1
    ]
    duplicate_items.sort(key=lambda item: item["count"], reverse=True)
    record_count = len(rows)
    quality_score = 100
    quality_score -= min(25, len(duplicate_items) / max(record_count, 1) * 100)
    quality_score -= min(25, sum(missing_counter.values()) / max(record_count * len(CORE_FIELDS), 1) * 100)
    quality_score -= min(20, len(anomaly_rows) / max(record_count, 1) * 100)
    quality_score += min(10, (sum(trusted_scores) / max(len(trusted_scores), 1) - 70) / 3) if trusted_scores else 0
    quality_score = round(max(0, min(100, quality_score)), 1)
    actions = []
    if duplicate_items:
        actions.append({"priority": "P1", "title": "合并重复车型版本", "evidence": f"发现 {len(duplicate_items)} 组重复 key", "action": "按 brand/model/year/trim 聚合，保留来源可信度最高记录"})
    if missing_counter:
        field, count = missing_counter.most_common(1)[0]
        actions.append({"priority": "P1", "title": f"补齐核心字段 {field}", "evidence": f"缺失 {count} 条", "action": "优先从官方来源或开放数据 raw_path 回补"})
    if anomaly_rows:
        actions.append({"priority": "P2", "title": "核验异常参数", "evidence": f"发现 {len(anomaly_rows)} 条疑似异常记录", "action": "对续航、电池、快充、价格异常值加人工核验标记"})
    result = {
        "summary": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "record_count": record_count,
            "quality_score": quality_score,
            "duplicate_group_count": len(duplicate_items),
            "anomaly_record_count": len(anomaly_rows),
            "trusted_source_rate": round((trust_counter.get("high", 0) + trust_counter.get("medium", 0)) / max(record_count, 1) * 100, 1),
            "missing_field_count": sum(missing_counter.values()),
        },
        "duplicates": duplicate_items[:20],
        "missing_fields": [{"field": key, "count": value} for key, value in missing_counter.most_common()],
        "anomalies": anomaly_rows[:30],
        "source_trust": {"levels": dict(trust_counter), "top_domains": source_counter.most_common(12)},
        "actions": actions[:8],
    }
    if persist:
        GOVERNANCE_REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def load_real_data_governance() -> Dict[str, Any]:
    if GOVERNANCE_REPORT.exists():
        return json.loads(GOVERNANCE_REPORT.read_text(encoding="utf-8"))
    return generate_real_data_governance(True)
