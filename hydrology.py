"""
Modul inti kalkulasi debit andalan
Metode: F.J. Mock Water Balance
Referensi: SNI 6738:2015
"""

import numpy as np
import pandas as pd
from scipy import stats
import math


# ─────────────────────────────────────────────────────────────
# 1. BOBOT THIESSEN
# ─────────────────────────────────────────────────────────────
def thiessen_weights(stations: list[dict], das_area_km2: float) -> list[dict]:
    """
    Hitung bobot Thiessen sederhana berbasis jarak terbalik (IDW).
    stations: [{'id', 'name', 'lat', 'lon'}, ...]
    Mengembalikan list dengan field tambahan 'weight' (0–1).
    """
    n = len(stations)
    if n == 1:
        stations[0]["weight"] = 1.0
        return stations

    coords = [(s["lat"], s["lon"]) for s in stations]

    # Hitung jarak antar stasiun (km) ─ formula Haversine
    def haversine(c1, c2):
        R = 6371.0
        lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
        lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    # Gunakan IDW (1/d^2) berbasis koordinat stasiun terhadap centroid semu
    c_lat = sum(c[0] for c in coords) / n
    c_lon = sum(c[1] for c in coords) / n
    centroid = (c_lat, c_lon)

    dists = []
    for c in coords:
        d = haversine(c, centroid)
        dists.append(max(d, 0.001))  # hindari bagi nol

    # Bobot = 1/d² dinormalisasi
    inv_d2 = [1 / (d ** 2) for d in dists]
    total = sum(inv_d2)
    weights = [v / total for v in inv_d2]

    for i, s in enumerate(stations):
        s["weight"] = round(weights[i], 4)

    return stations


# ─────────────────────────────────────────────────────────────
# 2. EVAPOTRANSPIRASI POTENSIAL (Penman-Monteith Sederhana)
# ─────────────────────────────────────────────────────────────
def etp_penman_monteith(T_mean: float, RH: float, Rs: float, uz: float, month: int) -> float:
    """
    Hitung ETP bulanan (mm/bulan) – metode Penman-Monteith FAO-56.
    T_mean  : suhu rata-rata (°C)
    RH      : kelembaban relatif (%)
    Rs      : radiasi matahari (MJ/m²/hari)
    uz      : kecepatan angin pd ketinggian z (m/s), diasumsikan z=2m
    month   : bulan (1-12) untuk koreksi lama hari
    """
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    n_days = days_in_month[month - 1]

    # Tekanan uap jenuh (kPa)
    es = 0.6108 * math.exp(17.27 * T_mean / (T_mean + 237.3))
    # Tekanan uap aktual
    ea = es * RH / 100
    # Slope kurva tekanan uap (kPa/°C)
    delta = 4098 * es / ((T_mean + 237.3) ** 2)
    # Konstanta psikrometrik (kPa/°C) - asumsi P = 101.3 kPa
    gamma = 0.000665 * 101.3
    # Radiasi neto (estimasi: Rn ≈ 0.77 * Rs - 2.0 Langley)
    Rn = 0.77 * Rs - 2.0
    if Rn < 0:
        Rn = 0.0
    # Konversi ke mm/hari
    lambda_ = 2.45  # MJ/kg
    Rn_mm = Rn / lambda_
    # Flux panas tanah G ≈ 0 (bulanan)
    G = 0

    numerator = (0.408 * delta * (Rn_mm - G) + gamma * (900 / (T_mean + 273)) * uz * (es - ea))
    denominator = delta + gamma * (1 + 0.34 * uz)

    ETP_day = numerator / denominator if denominator != 0 else 0
    ETP_month = max(ETP_day * n_days, 0)
    return round(ETP_month, 2)


# ─────────────────────────────────────────────────────────────
# 3. MODEL F.J. MOCK (NERACA AIR BULANAN)
# ─────────────────────────────────────────────────────────────
def mock_water_balance(
    rain_monthly: list[float],   # 12 nilai rata-rata hujan bulanan kawasan (mm)
    etp_monthly: list[float],    # 12 nilai ETP bulanan (mm)
    das_area_km2: float,
    sm_cap: float = 200.0,       # Kapasitas lengas tanah (mm)
    i: float = 0.5,              # koef infiltrasi
    k: float = 0.5,              # koef resesi air tanah
    gws_init: float = 0.0,       # tampungan air tanah awal (mm)
    sm_init: float = 100.0,      # lengas tanah awal (mm)
    n_years: int = 10,           # ulang simulasi untuk kondisi tunak
) -> dict:
    """
    Simulasi Mock bulanan selama n_years × 12 bulan.
    Mengembalikan dict dengan debit bulanan (m³/s) dan parameter antara.
    """
    results = {
        "month": [],
        "rain": [],
        "etp": [],
        "aet": [],
        "water_surplus": [],
        "direct_runoff": [],
        "infiltration": [],
        "gws": [],
        "baseflow": [],
        "total_flow_mm": [],
        "debit_m3s": [],
    }

    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    A_m2 = das_area_km2 * 1e6

    sm = sm_init
    gws = gws_init

    for yr in range(n_years):
        for m in range(12):
            P = rain_monthly[m]
            ETP = etp_monthly[m]
            n_days = days_per_month[m]

            # 1. Evapotranspirasi aktual
            delta_sm = P - ETP
            if delta_sm >= 0:
                AET = ETP
                sm_new = min(sm + delta_sm, sm_cap)
            else:
                # Defisit → ambil dari lengas tanah
                sm_reduction = min(abs(delta_sm), sm)
                AET = P + sm_reduction
                sm_new = sm - sm_reduction

            water_surplus = max(P - AET - (sm_new - sm), 0)

            # 2. Limpasan & infiltrasi
            direct_runoff = (1 - i) * water_surplus
            inf = i * water_surplus

            # 3. Perubahan tampungan air tanah
            gws_new = k * (gws + inf)
            baseflow = (1 - k) * (gws + inf) + k * gws - gws_new
            # Sederhananya:
            baseflow = (gws + inf) * (1 - k)
            gws_new = (gws + inf) * k

            total_flow_mm = direct_runoff + baseflow

            # 4. Debit (m³/s)
            Q_m3s = (total_flow_mm / 1000 * A_m2) / (n_days * 86400)

            # Simpan hasil tahun terakhir saja
            if yr == n_years - 1:
                results["month"].append(m + 1)
                results["rain"].append(round(P, 2))
                results["etp"].append(round(ETP, 2))
                results["aet"].append(round(AET, 2))
                results["water_surplus"].append(round(water_surplus, 2))
                results["direct_runoff"].append(round(direct_runoff, 2))
                results["infiltration"].append(round(inf, 2))
                results["gws"].append(round(gws_new, 2))
                results["baseflow"].append(round(baseflow, 2))
                results["total_flow_mm"].append(round(total_flow_mm, 2))
                results["debit_m3s"].append(round(Q_m3s, 4))

            sm = sm_new
            gws = gws_new

    return results


# ─────────────────────────────────────────────────────────────
# 4. KURVA DURASI ALIRAN (FDC)
# ─────────────────────────────────────────────────────────────
def flow_duration_curve(debit_list: list[float]) -> dict:
    """
    Hitung FDC dari deret debit bulanan.
    Mengembalikan {'exceedance': [...], 'debit': [...]}
    """
    sorted_q = sorted(debit_list, reverse=True)
    n = len(sorted_q)
    exceedance = [round((i + 1) / (n + 1) * 100, 2) for i in range(n)]
    return {"exceedance": exceedance, "debit": [round(q, 4) for q in sorted_q]}


def get_q_at_exceedance(fdc: dict, pct: float) -> float:
    """Interpolasi debit pada persentase exceedance tertentu."""
    exc = fdc["exceedance"]
    deb = fdc["debit"]
    if pct <= exc[0]:
        return deb[0]
    if pct >= exc[-1]:
        return deb[-1]
    for i in range(len(exc) - 1):
        if exc[i] <= pct <= exc[i + 1]:
            # Interpolasi linier
            t = (pct - exc[i]) / (exc[i + 1] - exc[i])
            return round(deb[i] + t * (deb[i + 1] - deb[i]), 4)
    return deb[-1]


# ─────────────────────────────────────────────────────────────
# 5. ANALISIS FREKUENSI (Log Pearson III)
# ─────────────────────────────────────────────────────────────
def log_pearson_iii_params(data: list[float]) -> dict:
    log_data = [math.log10(x) for x in data if x > 0]
    n = len(log_data)
    mean = np.mean(log_data)
    std = np.std(log_data, ddof=1)
    skew = (n / ((n - 1) * (n - 2))) * sum((x - mean) ** 3 for x in log_data) / (std ** 3) if std > 0 else 0
    return {"mean": round(mean, 4), "std": round(std, 4), "skew": round(skew, 4), "n": n}


def log_pearson_iii_quantile(params: dict, return_period: float) -> float:
    """Debit pada kala ulang T tahun (Log Pearson III)."""
    p_exceed = 1 / return_period
    # Nilai k dari distribusi normal standar
    z = stats.norm.ppf(1 - p_exceed)
    Cs = params["skew"]
    # Koefisien frekuensi Kite
    k = z + (z ** 2 - 1) * Cs / 6 + (z ** 3 - 6 * z) * (Cs ** 2) / 36 - (z ** 2 - 1) * (Cs ** 3) / 216
    log_Q = params["mean"] + k * params["std"]
    return round(10 ** log_Q, 4)


# ─────────────────────────────────────────────────────────────
# 6. HITUNG HUJAN KAWASAN (RATA-RATA TERBOBOTI)
# ─────────────────────────────────────────────────────────────
def weighted_areal_rainfall(station_rain: dict, weights: dict) -> list[float]:
    """
    station_rain: {station_id: [12 nilai bulanan], ...}
    weights     : {station_id: weight, ...}
    Return: [12 nilai hujan kawasan]
    """
    n_months = 12
    result = [0.0] * n_months
    total_w = sum(weights.values())

    for sid, rain_list in station_rain.items():
        w = weights.get(sid, 0)
        for m in range(n_months):
            result[m] += (w / total_w) * rain_list[m]

    return [round(v, 2) for v in result]


# ─────────────────────────────────────────────────────────────
# 7. WRAPPER UTAMA
# ─────────────────────────────────────────────────────────────
def run_analysis(payload: dict) -> dict:
    """
    Jalankan analisis debit andalan lengkap dari satu payload JSON.
    """
    # ── Data stasiun & bobot Thiessen
    stations = payload["stations"]  # [{id, name, lat, lon}]
    stations = thiessen_weights(stations, payload["das_area_km2"])
    weights = {s["id"]: s["weight"] for s in stations}

    # ── Hujan kawasan bulanan
    station_rain = payload["station_rain"]  # {id: [12 nilai]}
    areal_rain = weighted_areal_rainfall(station_rain, weights)

    # ── ETP bulanan
    climate = payload["climate"]  # [{month,T,RH,Rs,uz}, ...]
    etp_monthly = []
    for c in sorted(climate, key=lambda x: x["month"]):
        etp = etp_penman_monteith(c["T"], c["RH"], c["Rs"], c["uz"], c["month"])
        etp_monthly.append(etp)

    # ── Parameter DAS & Mock
    das = payload["das_params"]
    mock_res = mock_water_balance(
        rain_monthly=areal_rain,
        etp_monthly=etp_monthly,
        das_area_km2=payload["das_area_km2"],
        sm_cap=das.get("sm_cap", 200),
        i=das.get("i", 0.5),
        k=das.get("k", 0.5),
        gws_init=das.get("gws_init", 0),
        sm_init=das.get("sm_init", 100),
        n_years=20,
    )

    debit_list = mock_res["debit_m3s"]

    # ── FDC & debit andalan
    fdc = flow_duration_curve(debit_list)
    q80 = get_q_at_exceedance(fdc, 80)
    q90 = get_q_at_exceedance(fdc, 90)
    q95 = get_q_at_exceedance(fdc, 95)

    # ── Log Pearson III (opsional, dari debit bulanan)
    lp3 = log_pearson_iii_params(debit_list)
    lp3_results = {}
    for T in [2, 5, 10, 25, 50, 100]:
        lp3_results[f"Q{T}"] = log_pearson_iii_quantile(lp3, T)

    month_names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                   "Jul", "Agt", "Sep", "Okt", "Nov", "Des"]

    return {
        "stations": stations,
        "areal_rain": areal_rain,
        "etp_monthly": etp_monthly,
        "mock_table": {
            "month_name": month_names,
            **mock_res,
        },
        "fdc": fdc,
        "q_andalan": {"Q80": q80, "Q90": q90, "Q95": q95},
        "lp3_params": lp3,
        "lp3_kala_ulang": lp3_results,
    }
