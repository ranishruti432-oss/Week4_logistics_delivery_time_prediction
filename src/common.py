"""Shared configuration, paths and helper functions.

All randomness in this project is controlled by SEED so that every run
reproduces exactly the same dataset, metrics and figures.
"""
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
N_SAMPLES = 6000

ROOT = Path(__file__).resolve().parents[1]
# Output directory can be redirected (used by verify_reproducibility.py)
OUT = Path(os.environ.get("LOGISTICS_OUTPUT_DIR", ROOT))
DATA_DIR = OUT / "data"
RESULTS_DIR = OUT / "results"
FIG_DIR = RESULTS_DIR / "figures"
for _d in (DATA_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "delivery_data.csv"
MODEL_PATH = RESULTS_DIR / "best_model.joblib"

TARGET = "delivery_time_min"
NUMERIC_RAW = ["distance_km", "package_weight_kg", "stops_on_route", "hour_of_day",
               "day_of_week", "traffic_index", "driver_experience_years",
               "vehicle_load_pct"]
CATEGORICAL = ["vehicle_type", "weather", "zone", "priority"]
RAW_FEATURES = NUMERIC_RAW + CATEGORICAL
PEAK_HOURS = {8, 9, 10, 17, 18, 19}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering applied inside the sklearn Pipeline."""
    df = df.copy()
    df["is_peak_hour"] = df["hour_of_day"].isin(PEAK_HOURS).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["dist_x_traffic"] = df["distance_km"] * df["traffic_index"]
    return df


NUMERIC_ALL = NUMERIC_RAW + ["is_peak_hour", "is_weekend", "dist_x_traffic"]


def split_data(df: pd.DataFrame):
    """Deterministic 80/20 random split (returns train_df, test_df)."""
    return train_test_split(df, test_size=0.20, random_state=SEED)


def save_json(obj, path):
    import json
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
