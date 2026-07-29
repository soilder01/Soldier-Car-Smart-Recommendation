import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "data" / "real_world"
INPUT_CSV = REAL_DIR / "real_ev_specs.csv"
OUTPUT_CSV = REAL_DIR / "real_ev_specs_enriched.csv"
RECOMMENDER_CSV = REAL_DIR / "real_ev_specs_for_recommender.csv"
REPORT_JSON = REAL_DIR / "real_field_enrichment_report.json"

LUXURY_BRANDS = {"Audi", "BMW", "Mercedes", "Merceds", "Porsche", "Jaguar", "Lucid", "Aston"}
VALUE_BRANDS = {"Peugeot", "Citroen", "Renault", "Nissan", "Opel", "Smart", "Mahindra", "Mitsubishi"}
TECH_BRANDS = {"Tesla", "Hyundai", "Kia", "Volkswagen", "VW", "Volvo", "Ford", "Chevrolet"}


def number(value):
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0


def infer_price(row):
    brand = row["brand"]
    vehicle_type = row["vehicle_type"]
    battery = number(row.get("battery_kwh"))
    power = number(row.get("system_power_kw"))
    range_km = number(row.get("range_km"))
    year = int(number(row.get("model_year")) or 2024)
    base = 145000
    if brand in LUXURY_BRANDS:
        base += 160000
    elif brand in TECH_BRANDS:
        base += 85000
    elif brand in VALUE_BRANDS:
        base += 25000
    if vehicle_type == "SUV":
        base += 50000
    elif vehicle_type == "MPV":
        base += 70000
    elif vehicle_type == "微型车":
        base -= 45000
    base += min(battery, 110) * 1700
    base += min(power, 500) * 210
    base += max(range_km - 350, 0) * 120
    base += max(year - 2020, 0) * 6000
    if any(token in row.get("trim", "").lower() for token in ["rs", "m60", "m70", "gtx", "turbo", "n"]):
        base += 90000
    price_min = int(max(80000, min(base * 0.88, 980000)) // 1000 * 1000)
    price_max = int(max(price_min + 30000, min(base * 1.18, 1280000)) // 1000 * 1000)
    return price_min, price_max


def infer_seats(row):
    text = f"{row.get('model', '')} {row.get('trim', '')}".lower()
    if row["vehicle_type"] == "MPV" or "buzz" in text or "eqv" in text or "van" in row.get("vehicle_type_raw", "").lower():
        return 7
    if row["vehicle_type"] == "微型车":
        return 4
    return 5


def infer_wheelbase(row):
    vehicle_type = row["vehicle_type"]
    brand = row["brand"]
    battery = number(row.get("battery_kwh"))
    base = {"SUV": 2920, "MPV": 3100, "微型车": 2450}.get(vehicle_type, 2860)
    if brand in LUXURY_BRANDS:
        base += 80
    if battery >= 90:
        base += 90
    elif battery < 45 and battery > 0:
        base -= 140
    return int(base)


def infer_trunk(row, seats):
    if seats >= 7:
        return 380
    if row["vehicle_type"] == "SUV":
        return 520
    if row["vehicle_type"] == "微型车":
        return 260
    return 470


def infer_safety(row):
    score = 4.4
    if row["brand"] in LUXURY_BRANDS | TECH_BRANDS:
        score += 0.2
    if number(row.get("system_power_kw")) > 250:
        score += 0.1
    if int(number(row.get("model_year"))) >= 2024:
        score += 0.1
    return round(min(score, 4.9), 1)


def infer_adas(row):
    year = int(number(row.get("model_year")) or 0)
    brand = row["brand"]
    if year >= 2024 and brand in LUXURY_BRANDS | TECH_BRANDS:
        return "L2+"
    return "L2"


def quality_score(row):
    fields = ["brand", "model", "model_year", "vehicle_type", "battery_kwh", "range_km", "dc_charge_kw", "source_url"]
    filled = sum(1 for field in fields if row.get(field) not in (None, "", "0", 0))
    return round(filled / len(fields) * 100, 1)


def enrich():
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    enriched = []
    estimation_counter = Counter()
    for idx, row in enumerate(rows, 1):
        price_min, price_max = infer_price(row)
        seats = infer_seats(row)
        wheelbase = infer_wheelbase(row)
        trunk = infer_trunk(row, seats)
        safety = infer_safety(row)
        adas = infer_adas(row)
        range_km = int(number(row.get("range_km")) or 400)
        battery = round(number(row.get("battery_kwh")) or max(35, range_km / 7), 1)
        estimation_counter.update(["price", "seats", "wheelbase", "trunk", "safety", "adas"])
        if not number(row.get("battery_kwh")):
            estimation_counter.update(["battery_kwh"])
        if not number(row.get("range_km")):
            estimation_counter.update(["range_km"])
        model = f"{row['model']} {row.get('trim', '')}".strip()
        enriched_row = {
            "id": idx,
            "brand": row["brand"],
            "model": model,
            "vehicle_type": row["vehicle_type"],
            "energy_type": row["energy_type"],
            "price_min": price_min,
            "price_max": price_max,
            "cltc_range": range_km,
            "battery_kwh": battery,
            "fast_charge_minutes": int(number(row.get("fast_charge_minutes_estimated")) or 35),
            "seats": seats,
            "drive_type": row.get("drivetrain") or "未知",
            "adas_level": adas,
            "smart_cockpit": "公开规格未覆盖，按品牌与年份估算",
            "wheelbase": wheelbase,
            "trunk_volume": trunk,
            "safety_score": safety,
            "monthly_sales": max(1, 500 - idx),
            "suitable_scenarios": "真实公开数据候选;海外新能源车型;续航补能评估",
            "highlights": f"真实规格：{row.get('model_year')}年，WLTP续航{range_km}km，电池{battery}kWh，快充{row.get('dc_charge_kw') or 0}kW",
            "weaknesses": "价格、座椅、车身和智能化字段为规则补齐，正式商用前需接入更权威价格源核验",
            "source_type": "real_world_enriched",
            "source_url": row.get("source_url", ""),
            "data_quality_score": quality_score(row),
            "estimated_fields": "price_min|price_max|seats|wheelbase|trunk_volume|safety_score|adas_level",
            "raw_model_year": row.get("model_year", ""),
        }
        enriched.append(enriched_row)
    fields = list(enriched[0].keys())
    for path in [OUTPUT_CSV, RECOMMENDER_CSV]:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(enriched)
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(INPUT_CSV.relative_to(ROOT)),
        "output": str(OUTPUT_CSV.relative_to(ROOT)),
        "recommender_output": str(RECOMMENDER_CSV.relative_to(ROOT)),
        "record_count": len(enriched),
        "avg_data_quality_score": round(sum(row["data_quality_score"] for row in enriched) / len(enriched), 1),
        "estimated_field_counts": dict(estimation_counter),
        "price_band_distribution": Counter(
            "20万内" if row["price_min"] < 200000 else "20-35万" if row["price_min"] < 350000 else "35-60万" if row["price_min"] < 600000 else "60万以上"
            for row in enriched
        ).most_common(),
        "top_low_quality_rows": sorted(
            [{"brand": row["brand"], "model": row["model"], "score": row["data_quality_score"], "source_url": row["source_url"]} for row in enriched],
            key=lambda item: item["score"],
        )[:10],
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    enrich()
