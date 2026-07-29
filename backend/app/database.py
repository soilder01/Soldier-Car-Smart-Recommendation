import csv
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List

from app.config import DB_PATH, VEHICLE_CSV


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand TEXT,
                model TEXT,
                vehicle_type TEXT,
                energy_type TEXT,
                price_min INTEGER,
                price_max INTEGER,
                cltc_range INTEGER,
                battery_kwh REAL,
                fast_charge_minutes INTEGER,
                seats INTEGER,
                drive_type TEXT,
                adas_level TEXT,
                smart_cockpit TEXT,
                wheelbase INTEGER,
                trunk_volume INTEGER,
                safety_score REAL,
                monthly_sales INTEGER,
                suitable_scenarios TEXT,
                highlights TEXT,
                weaknesses TEXT
            );
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                name TEXT,
                phone_masked TEXT,
                profile_json TEXT,
                budget INTEGER,
                city TEXT,
                concerns TEXT,
                intent_level TEXT,
                recommended_models TEXT,
                next_action TEXT
            );
            CREATE TABLE IF NOT EXISTS recommendation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                user_query TEXT,
                profile_json TEXT,
                result_json TEXT,
                top_model TEXT,
                confidence REAL
            );
            CREATE TABLE IF NOT EXISTS recommendation_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                query TEXT,
                model_name TEXT,
                rating TEXT,
                reason TEXT,
                profile_json TEXT,
                recommendation_json TEXT,
                candidate_pool TEXT DEFAULT '',
                scenario_tags TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                updated_at TEXT,
                summary TEXT
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                role TEXT,
                content TEXT,
                created_at TEXT
            );
            """
        )
        db_count = conn.execute("SELECT COUNT(*) AS c FROM vehicles").fetchone()["c"]
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(recommendation_feedback)").fetchall()}
        if "candidate_pool" not in columns:
            conn.execute("ALTER TABLE recommendation_feedback ADD COLUMN candidate_pool TEXT DEFAULT ''")
        if "scenario_tags" not in columns:
            conn.execute("ALTER TABLE recommendation_feedback ADD COLUMN scenario_tags TEXT DEFAULT ''")
        with VEHICLE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            csv_count = sum(1 for _ in csv.DictReader(f))
        if db_count != csv_count:
            conn.execute("DELETE FROM vehicles")
            seed_vehicles(conn)


def seed_vehicles(conn):
    with VEHICLE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        conn.execute(
            """
            INSERT INTO vehicles (
                brand, model, vehicle_type, energy_type, price_min, price_max, cltc_range,
                battery_kwh, fast_charge_minutes, seats, drive_type, adas_level, smart_cockpit,
                wheelbase, trunk_volume, safety_score, monthly_sales, suitable_scenarios,
                highlights, weaknesses
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["brand"], row["model"], row["vehicle_type"], row["energy_type"],
                int(row["price_min"]), int(row["price_max"]), int(row["cltc_range"]),
                float(row["battery_kwh"]), int(row["fast_charge_minutes"]), int(row["seats"]),
                row["drive_type"], row["adas_level"], row["smart_cockpit"], int(row["wheelbase"]),
                int(row["trunk_volume"]), float(row["safety_score"]), int(row["monthly_sales"]),
                row["suitable_scenarios"], row["highlights"], row["weaknesses"],
            ),
        )


def rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def list_vehicles() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        return rows_to_dicts(conn.execute("SELECT * FROM vehicles ORDER BY monthly_sales DESC").fetchall())


def find_vehicles_by_models(models: List[str]) -> List[Dict[str, Any]]:
    vehicles = list_vehicles()
    result = []
    for name in models:
        for vehicle in vehicles:
            full_name = f"{vehicle['brand']} {vehicle['model']}"
            if name.lower() in full_name.lower() or vehicle["model"].lower() in name.lower():
                if vehicle not in result:
                    result.append(vehicle)
    return result


def match_mentioned_vehicles(query: str) -> List[Dict[str, Any]]:
    text = (query or "").lower().replace(" ", "")
    vehicles = list_vehicles()
    result = []

    def add_if(predicate):
        for item in vehicles:
            if predicate(item) and item not in result:
                result.append(item)

    if any(alias in text for alias in ["e300l", "e300", "奔驰e", "奔驰e级"]):
        add_if(lambda v: v["brand"] == "奔驰" and v["model"] == "E300L")
    if any(alias in text for alias in ["宝马5", "5系", "宝马五系", "530li"]):
        add_if(lambda v: v["brand"] == "宝马" and v["model"] == "530Li")
    if any(alias in text for alias in ["model3", "model 3", "特斯拉3"]):
        add_if(lambda v: v["brand"] == "特斯拉" and v["model"] == "Model 3")
    if any(alias in text for alias in ["modely", "model y"]):
        add_if(lambda v: v["brand"] == "特斯拉" and v["model"] == "Model Y")
    if "享界s9增程" in text:
        add_if(lambda v: v["brand"] == "享界" and v["model"] == "S9增程")
    elif "享界s9" in text:
        add_if(lambda v: v["brand"] == "享界" and v["model"] == "S9")
    if "尊界s800" in text:
        add_if(lambda v: v["brand"] == "尊界" and v["model"] == "S800")
    for brand, models in {"问界": ["M5", "M7", "M8", "M9"], "智界": ["S7", "R7"]}.items():
        for model in models:
            if f"{brand}{model}".lower() in text:
                add_if(lambda v, b=brand, m=model: v["brand"] == b and v["model"] == m)

    for vehicle in vehicles:
        model_key = vehicle["model"].lower().replace(" ", "")
        names = [f"{vehicle['brand']}{vehicle['model']}".lower().replace(" ", "")]
        if len(model_key) >= 2:
            names.append(model_key)
        if any(alias.lower().replace(" ", "") in text for alias in names):
            if vehicle not in result:
                result.append(vehicle)
    return result


def save_recommendation_log(query: str, profile: Dict, result: List[Dict]):
    top_model = f"{result[0]['brand']} {result[0]['model']}" if result else ""
    confidence = float(result[0].get("score", 0)) if result else 0
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO recommendation_logs (created_at, user_query, profile_json, result_json, top_model, confidence) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), query, json.dumps(profile, ensure_ascii=False), json.dumps(result, ensure_ascii=False), top_model, confidence),
        )


def create_lead(data: Dict):
    profile = data.get("profile", {})
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO leads (created_at, name, phone_masked, profile_json, budget, city, concerns, intent_level, recommended_models, next_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.get("name", "匿名客户"),
                data.get("phone_masked", ""),
                json.dumps(profile, ensure_ascii=False),
                profile.get("budget_max") or 0,
                profile.get("city", ""),
                ",".join(profile.get("concerns", [])),
                profile.get("intent_level", "了解中"),
                ",".join(data.get("recommended_models", [])),
                data.get("next_action", ""),
            ),
        )
        return {"id": cur.lastrowid, **data}


def list_leads() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        return rows_to_dicts(conn.execute("SELECT * FROM leads ORDER BY id DESC").fetchall())


def list_recommendation_logs() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        return rows_to_dicts(conn.execute("SELECT * FROM recommendation_logs ORDER BY id DESC LIMIT 200").fetchall())


def save_recommendation_feedback(data: Dict[str, Any]) -> Dict[str, Any]:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = data.get("model_name") or ""
    scenario_tags = data.get("scenario_tags") or []
    if isinstance(scenario_tags, str):
        scenario_tags = [item.strip() for item in scenario_tags.split(",") if item.strip()]
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO recommendation_feedback (created_at, query, model_name, rating, reason, profile_json, recommendation_json, candidate_pool, scenario_tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                data.get("query", ""),
                model_name,
                data.get("rating", "neutral"),
                data.get("reason", ""),
                json.dumps(data.get("profile", {}), ensure_ascii=False),
                json.dumps(data.get("recommendation", {}), ensure_ascii=False),
                data.get("candidate_pool", "") or "unknown",
                json.dumps(scenario_tags, ensure_ascii=False),
            ),
        )
    return {"id": cur.lastrowid, "created_at": created_at, **data, "model_name": model_name, "scenario_tags": scenario_tags}


def list_recommendation_feedback(limit: int = 100) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = rows_to_dicts(conn.execute(
            "SELECT * FROM recommendation_feedback ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 500)),),
        ).fetchall())
    for row in rows:
        row["profile"] = json.loads(row.pop("profile_json") or "{}")
        row["recommendation"] = json.loads(row.pop("recommendation_json") or "{}")
        try:
            row["scenario_tags"] = json.loads(row.get("scenario_tags") or "[]")
        except json.JSONDecodeError:
            row["scenario_tags"] = [item.strip() for item in (row.get("scenario_tags") or "").split(",") if item.strip()]
        row["candidate_pool"] = row.get("candidate_pool") or "unknown"
    return rows


def recommendation_feedback_summary() -> Dict[str, Any]:
    rows = list_recommendation_feedback(500)
    total = len(rows)
    positive = sum(1 for item in rows if item.get("rating") == "positive")
    negative = sum(1 for item in rows if item.get("rating") == "negative")
    neutral = total - positive - negative
    model_counter: Dict[str, Dict[str, Any]] = {}
    pool_counter: Dict[str, Dict[str, Any]] = {}
    scene_counter: Dict[str, Dict[str, Any]] = {}
    reason_counter: Dict[str, int] = {}
    for item in rows:
        rating_key = item.get("rating") if item.get("rating") in {"positive", "negative"} else "neutral"
        model = item.get("model_name") or "未命名车型"
        bucket = model_counter.setdefault(model, {"model_name": model, "total": 0, "positive": 0, "negative": 0, "neutral": 0})
        bucket["total"] += 1
        bucket[rating_key] += 1
        pool = item.get("candidate_pool") or "unknown"
        pool_bucket = pool_counter.setdefault(pool, {"candidate_pool": pool, "total": 0, "positive": 0, "negative": 0, "neutral": 0, "positive_rate": 0})
        pool_bucket["total"] += 1
        pool_bucket[rating_key] += 1
        for scene in item.get("scenario_tags") or ["未标注场景"]:
            scene_bucket = scene_counter.setdefault(scene, {"scenario": scene, "total": 0, "positive": 0, "negative": 0, "neutral": 0, "negative_rate": 0})
            scene_bucket["total"] += 1
            scene_bucket[rating_key] += 1
        reason = item.get("reason") or "未填写原因"
        reason_counter[reason] = reason_counter.get(reason, 0) + 1
    model_rows = sorted(model_counter.values(), key=lambda item: item["total"], reverse=True)[:10]
    for row in pool_counter.values():
        row["positive_rate"] = round(row["positive"] / max(row["total"], 1) * 100, 1)
    for row in scene_counter.values():
        row["negative_rate"] = round(row["negative"] / max(row["total"], 1) * 100, 1)
    pool_rows = sorted(pool_counter.values(), key=lambda item: (item["positive_rate"], item["total"]), reverse=True)
    scene_rows = sorted(scene_counter.values(), key=lambda item: (item["negative_rate"], item["total"]), reverse=True)[:10]
    reasons = [{"reason": key, "count": value} for key, value in sorted(reason_counter.items(), key=lambda item: item[1], reverse=True)[:10]]
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "positive_rate": round(positive / max(total, 1) * 100, 1),
        "model_rows": model_rows,
        "pool_rows": pool_rows,
        "scene_rows": scene_rows,
        "reasons": reasons,
        "recent": rows[:20],
    }


def clear_runtime_data():
    with get_conn() as conn:
        conn.execute("DELETE FROM leads")
        conn.execute("DELETE FROM recommendation_logs")
        conn.execute("DELETE FROM recommendation_feedback")
        conn.execute("DELETE FROM chat_sessions")
        conn.execute("DELETE FROM chat_messages")
    return {"status": "ok"}
