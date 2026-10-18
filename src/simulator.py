"""Data-generating process (simulator) for the delivery-time problem.

IMPORTANT: This dataset is SIMULATED. It is NOT real company data. The
formulas below encode commonly observed logistics relationships (traffic
slows vehicles non-linearly, rain hurts bikes more than vans, each extra stop
adds waiting time, ...). The numeric constants are assumptions chosen for
realism and are not measured values.

The deterministic part of the process, `expected_time()`, is also used as a
"digital twin" to evaluate optimisation decisions independently of the ML
model (so the optimisation results are not circular).
"""
import numpy as np
import pandas as pd

VEHICLES = ["bike", "van", "truck"]
WEATHERS = ["clear", "rain", "fog", "storm"]
ZONES = ["Central", "North", "South", "East", "West"]

SPEED_KMPH = {"bike": 16.0, "van": 28.0, "truck": 22.0}
TRAFFIC_SENS = {"bike": 0.6, "van": 1.0, "truck": 1.1}
WEATHER_FACTOR = {
    "bike":  {"clear": 1.00, "rain": 1.35, "fog": 1.20, "storm": 1.70},
    "van":   {"clear": 1.00, "rain": 1.15, "fog": 1.12, "storm": 1.40},
    "truck": {"clear": 1.00, "rain": 1.12, "fog": 1.10, "storm": 1.35},
}
ZONE_TRAFFIC_ADJ = {"Central": 1.2, "North": 0.0, "South": 0.2, "East": -0.4, "West": -0.2}
WEATHER_TRAFFIC_ADJ = {"clear": 0.0, "rain": 0.8, "fog": 0.3, "storm": 1.5}
HOURS = np.arange(7, 21)  # operating hours 07:00 - 20:59


def hourly_traffic_profile(h):
    """Average congestion index (0-10) by hour: morning and evening peaks."""
    h = np.asarray(h, dtype=float)
    return (2.0 + 3.2 * np.exp(-(h - 9) ** 2 / (2 * 1.3 ** 2))
            + 3.6 * np.exp(-(h - 18) ** 2 / (2 * 1.5 ** 2))
            + 0.8 * np.exp(-(h - 13) ** 2 / (2 * 1.5 ** 2)))


def traffic_factor(t, vehicle):
    """Multiplicative slow-down caused by congestion index t (non-linear)."""
    if isinstance(vehicle, str):
        sens = TRAFFIC_SENS[vehicle]
    else:
        sens = pd.Series(vehicle).map(TRAFFIC_SENS).to_numpy()
    t = np.asarray(t, dtype=float)
    return 1.0 + sens * (0.04 * t + 0.006 * t ** 2)


def expected_time(df: pd.DataFrame) -> np.ndarray:
    """Noise-free expected dispatch-to-delivery time (minutes). Needs complete rows."""
    v = df["vehicle_type"].to_numpy()
    speed = pd.Series(v).map(SPEED_KMPH).to_numpy()
    wf = np.array([WEATHER_FACTOR[a][b] for a, b in zip(v, df["weather"])])
    tf = traffic_factor(df["traffic_index"].to_numpy(), v)
    ef = 1.0 - 0.008 * np.minimum(df["driver_experience_years"].to_numpy(), 10)
    lf = 1.0 + 0.0015 * (df["vehicle_load_pct"].to_numpy() - 60)
    w = np.minimum(df["package_weight_kg"].to_numpy(), 150)
    drive = df["distance_km"].to_numpy() / speed * 60.0 * tf * wf * ef * lf
    loading = 3.0 + 0.04 * w
    handling = 1.5 + 0.03 * w
    express = (df["priority"].to_numpy() == "express")
    queue = (df["stops_on_route"].to_numpy() - 1) * 2.2 * np.where(express, 0.7, 1.0)
    return loading + drive + queue + handling


def promised_eta(df: pd.DataFrame) -> np.ndarray:
    """Customer-quoted ETA from today's rule-of-thumb (ignores traffic/weather)."""
    return 14.0 + 3.0 * df["distance_km"].to_numpy() + 2.0 * (df["stops_on_route"].to_numpy() - 1)


def generate_deliveries(n: int, seed: int) -> pd.DataFrame:
    """Return the COMPLETE simulated dataset (no missing values)."""
    rng = np.random.default_rng(seed)

    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    demand = 1 + 1.2 * np.exp(-(HOURS - 10) ** 2 / (2 * 1.8 ** 2)) \
        + 1.4 * np.exp(-(HOURS - 17) ** 2 / (2 * 2.0 ** 2))
    hour = rng.choice(HOURS, n, p=demand / demand.sum())
    minute = rng.integers(0, 60, n)
    dispatch_time = dates + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
    dow = dispatch_time.dayofweek.to_numpy()

    zone = rng.choice(ZONES, n, p=[0.25, 0.20, 0.20, 0.20, 0.15])
    weather = rng.choice(WEATHERS, n, p=[0.70, 0.22, 0.05, 0.03])
    priority = rng.choice(["standard", "express"], n, p=[0.80, 0.20])

    distance = np.clip(rng.gamma(2.2, 3.6, n), 0.5, 40.0).round(2)
    weight = np.clip(rng.lognormal(1.6, 0.95, n), 0.1, 300.0).round(2)
    stops = np.clip(rng.poisson(5, n) + 1, 1, 14)
    experience = np.clip(rng.gamma(3.0, 2.5, n), 0.5, 20.0).round(1)
    load_pct = rng.uniform(20, 100, n).round(0)

    # Vehicle choice mimics current dispatch practice (heavy / far -> van or truck)
    vehicle = np.empty(n, dtype=object)
    small = (weight <= 10) & (distance <= 15)
    u = rng.random(n)
    vehicle[small] = np.where(u[small] < 0.45, "bike", "van")
    big = ~small
    vehicle[big] = np.where(((weight[big] > 60) & (u[big] < 0.55)) | (u[big] < 0.05), "truck", "van")

    traffic = (hourly_traffic_profile(hour)
               + pd.Series(zone).map(ZONE_TRAFFIC_ADJ).to_numpy()
               + pd.Series(weather).map(WEATHER_TRAFFIC_ADJ).to_numpy()
               - np.where(dow >= 5, 1.0, 0.0) + rng.normal(0, 0.8, n))
    traffic = np.clip(traffic, 0, 10).round(1)

    df = pd.DataFrame({
        "dispatch_time": dispatch_time, "distance_km": distance,
        "package_weight_kg": weight, "stops_on_route": stops, "hour_of_day": hour,
        "day_of_week": dow, "traffic_index": traffic, "weather": weather,
        "zone": zone, "vehicle_type": vehicle, "driver_experience_years": experience,
        "vehicle_load_pct": load_pct, "priority": priority,
    })
    df = df.sort_values("dispatch_time").reset_index(drop=True)
    df.insert(0, "delivery_id", [f"D{i:05d}" for i in range(1, n + 1)])

    mu = expected_time(df)
    noise = rng.lognormal(0.0, 0.10, n)
    incident = np.where(rng.random(n) < 0.03, rng.uniform(8, 30, n), 0.0)  # unpredictable events
    df["delivery_time_min"] = (mu * noise + incident).round(2)
    df["promised_eta_min"] = promised_eta(df).round(1)
    return df


def inject_missing(df: pd.DataFrame, seed: int, rate: float = 0.015) -> pd.DataFrame:
    """Mimic sensor / data-entry gaps: ~1.5% missing in two columns."""
    rng = np.random.default_rng(seed + 1)
    out = df.copy()
    for col in ["traffic_index", "driver_experience_years"]:
        out.loc[rng.random(len(out)) < rate, col] = np.nan
    return out
