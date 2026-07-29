import csv
import json
import ssl
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "real_world"
RAW_DIR = OUT_DIR / "raw"
RAW_ZIP = RAW_DIR / "open_ev_data_dataset_main.zip"
REAL_SPECS_CSV = OUT_DIR / "real_ev_specs.csv"
RECOMMENDER_CSV = OUT_DIR / "real_ev_specs_for_recommender.csv"
REPORT_JSON = OUT_DIR / "real_data_quality_report.json"
SOURCE_ARCHIVE_URL = "https://github.com/open-ev-data/open-ev-data-dataset/archive/refs/heads/main.zip"
SUPPLEMENTAL_CSV_URL = "https://raw.githubusercontent.com/OSkrk/Electric-vehicles-EV-Database/master/Data/EVs_data_base.csv"
SUPPLEMENTAL_CSV = RAW_DIR / "oskrk_ev_database.csv"
LIMIT = 500


def fetch_url(url, target, timeout=120):
    request = urllib.request.Request(url, headers={"User-Agent": "Soldier-Car-Recommendation-Research/1.0"})
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        target.write_bytes(response.read())


def download_sources():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetch_url(SOURCE_ARCHIVE_URL, RAW_ZIP)
    fetch_url(SUPPLEMENTAL_CSV_URL, SUPPLEMENTAL_CSV)


def nested(data, *keys, default=""):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default


def first_range(data):
    rated = nested(data, "range", "rated", default=[])
    if isinstance(rated, list):
        preferred = sorted(rated, key=lambda item: 0 if item.get("cycle") in {"cltc", "wltp", "epa"} else 1)
        for item in preferred:
            if item.get("range_km"):
                return int(float(item["range_km"])), item.get("cycle", "")
    return 0, ""


def first_source(data):
    sources = data.get("sources") or []
    if sources and isinstance(sources, list):
        return sources[0].get("url", "")
    return ""


def normalize_vehicle_type(value):
    mapping = {
        "passenger_car": "轿车",
        "suv": "SUV",
        "van": "MPV",
        "truck": "皮卡",
        "micro": "微型车",
    }
    return mapping.get((value or "").lower(), "轿车")


def normalize_energy_type(data):
    return "纯电" if nested(data, "battery", "pack_capacity_kwh_net", default=0) else "纯电"


def infer_fast_charge_minutes(dc_kw):
    if dc_kw >= 250:
        return 18
    if dc_kw >= 150:
        return 25
    if dc_kw >= 100:
        return 32
    if dc_kw > 0:
        return 45
    return 60


def parse_archive():
    rows = []
    with zipfile.ZipFile(RAW_ZIP) as zf:
        names = sorted(name for name in zf.namelist() if "/src/" in name and name.endswith(".json") and not name.endswith("base.json"))
        for name in names:
            if len(rows) >= LIMIT:
                break
            data = json.loads(zf.read(name).decode("utf-8"))
            if not data.get("year") or not nested(data, "make", "name") or not nested(data, "model", "name"):
                continue
            range_km, range_cycle = first_range(data)
            dc_kw = int(float(nested(data, "charging", "dc", "max_power_kw", default=0) or 0))
            ac_kw = float(nested(data, "charging", "ac", "max_power_kw", default=0) or 0)
            battery_kwh = float(nested(data, "battery", "pack_capacity_kwh_net", default=0) or nested(data, "battery", "pack_capacity_kwh_gross", default=0) or 0)
            rows.append({
                "brand": nested(data, "make", "name"),
                "model": nested(data, "model", "name"),
                "model_year": data.get("year", ""),
                "trim": nested(data, "trim", "name"),
                "vehicle_type_raw": data.get("vehicle_type", ""),
                "vehicle_type": normalize_vehicle_type(data.get("vehicle_type", "")),
                "energy_type": normalize_energy_type(data),
                "markets": "|".join(data.get("markets") or []),
                "drivetrain": nested(data, "powertrain", "drivetrain"),
                "system_power_kw": nested(data, "powertrain", "system_power_kw", default=0),
                "battery_kwh": battery_kwh,
                "range_km": range_km,
                "range_cycle": range_cycle,
                "ac_charge_kw": ac_kw,
                "dc_charge_kw": dc_kw,
                "fast_charge_minutes_estimated": infer_fast_charge_minutes(dc_kw),
                "source_url": first_source(data),
                "raw_path": name,
            })
    return rows


def parse_number(value):
    text = str(value or "").strip()
    if not text or text.lower() in {"empty", "na", "none"}:
        return 0
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return 0


def parse_supplemental_csv(existing_rows):
    rows = list(existing_rows)
    with SUPPLEMENTAL_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f):
            if len(rows) >= LIMIT:
                break
            car_model = (item.get("Car model") or "").strip()
            if not car_model:
                continue
            parts = car_model.split(" ", 1)
            brand = parts[0]
            model = parts[1] if len(parts) > 1 else car_model
            battery = parse_number(item.get("Battery_Capacity_kWh"))
            rows.append({
                "brand": brand,
                "model": model,
                "model_year": int(parse_number(item.get("year")) or 0),
                "trim": "",
                "vehicle_type_raw": item.get("Car_type", ""),
                "vehicle_type": normalize_vehicle_type((item.get("Car_type") or "passenger_car").lower()),
                "energy_type": "纯电",
                "markets": "EU|Global",
                "drivetrain": "未知",
                "system_power_kw": 0,
                "battery_kwh": battery,
                "range_km": int(parse_number(item.get("Autonomy_WLTP_Km")) or 0),
                "range_cycle": "wltp",
                "ac_charge_kw": parse_number(item.get("AC_nominal_charging")),
                "dc_charge_kw": int(parse_number(item.get("DC_nominal_charge_power_KW")) or 0),
                "fast_charge_minutes_estimated": infer_fast_charge_minutes(int(parse_number(item.get("DC_nominal_charge_power_KW")) or 0)),
                "source_url": item.get("Source", "") or SUPPLEMENTAL_CSV_URL,
                "raw_path": "OSkrk/Electric-vehicles-EV-Database/Data/EVs_data_base.csv",
            })
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_recommender_rows(rows):
    result = []
    for idx, row in enumerate(rows):
        brand = row["brand"]
        model = f"{row['model']} {row['trim']}".strip() if row.get("trim") else row["model"]
        result.append({
            "brand": brand,
            "model": model,
            "vehicle_type": row["vehicle_type"],
            "energy_type": row["energy_type"],
            "price_min": 150000,
            "price_max": 450000,
            "cltc_range": row["range_km"] or 400,
            "battery_kwh": row["battery_kwh"] or 60,
            "fast_charge_minutes": row["fast_charge_minutes_estimated"],
            "seats": 5 if row["vehicle_type"] != "MPV" else 7,
            "drive_type": row["drivetrain"] or "未知",
            "adas_level": "L2",
            "smart_cockpit": "公开数据未覆盖",
            "wheelbase": 2850 if row["vehicle_type"] == "SUV" else 2750,
            "trunk_volume": 480,
            "safety_score": 4.5,
            "monthly_sales": max(1, LIMIT - idx),
            "suitable_scenarios": "真实公开数据扩展样本;海外新能源车型;续航补能评估",
            "highlights": f"Open EV Data真实规格：{row['model_year']}年，{row['range_cycle'].upper() or '公开'}续航{row['range_km'] or '未知'}km，电池{row['battery_kwh'] or '未知'}kWh",
            "weaknesses": "价格、座椅、智能座舱等字段非该公开数据集核心字段，已用保守估计值补齐用于评估",
        })
    return result


def build_report(rows):
    fields = list(rows[0].keys()) if rows else []
    missing_counts = {field: sum(1 for row in rows if row.get(field) in (None, "", 0) and field not in {"dc_charge_kw", "ac_charge_kw", "system_power_kw"}) for field in fields}
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "open-ev-data/open-ev-data-dataset + OSkrk/Electric-vehicles-EV-Database",
        "source_url": [SOURCE_ARCHIVE_URL, SUPPLEMENTAL_CSV_URL],
        "license_file": "https://github.com/open-ev-data/open-ev-data-dataset/blob/main/LICENCE",
        "record_count": len(rows),
        "unique_brand_count": len({row["brand"] for row in rows}),
        "unique_model_count": len({(row["brand"], row["model"]) for row in rows}),
        "year_range": [min(int(row["model_year"]) for row in rows), max(int(row["model_year"]) for row in rows)] if rows else [],
        "brand_distribution_top20": Counter(row["brand"] for row in rows).most_common(20),
        "vehicle_type_distribution": Counter(row["vehicle_type"] for row in rows).most_common(),
        "range_cycle_distribution": Counter(row["range_cycle"] or "unknown" for row in rows).most_common(),
        "missing_counts": missing_counts,
        "files": {
            "raw_archive": str(RAW_ZIP.relative_to(ROOT)),
            "normalized_csv": str(REAL_SPECS_CSV.relative_to(ROOT)),
            "recommender_csv": str(RECOMMENDER_CSV.relative_to(ROOT)),
        },
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    download_sources()
    rows = parse_supplemental_csv(parse_archive())
    if len(rows) < 200:
        raise RuntimeError(f"真实数据不足 200 条：{len(rows)}")
    write_csv(REAL_SPECS_CSV, rows, list(rows[0].keys()))
    recommender_rows = build_recommender_rows(rows)
    write_csv(RECOMMENDER_CSV, recommender_rows, list(recommender_rows[0].keys()))
    print(json.dumps(build_report(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
