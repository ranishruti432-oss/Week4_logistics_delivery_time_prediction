"""Step 1 - generate the SIMULATED logistics dataset and an EDA figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATA_PATH, FIG_DIR, N_SAMPLES, SEED, RESULTS_DIR, TARGET, save_json
from simulator import generate_deliveries, inject_missing


def main():
    clean = generate_deliveries(N_SAMPLES, SEED)
    df = inject_missing(clean, SEED)
    df.to_csv(DATA_PATH, index=False)

    summary = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "date_range": [str(df.dispatch_time.min()), str(df.dispatch_time.max())],
        "target_mean": round(float(df[TARGET].mean()), 2),
        "target_median": round(float(df[TARGET].median()), 2),
        "target_std": round(float(df[TARGET].std()), 2),
        "target_min": round(float(df[TARGET].min()), 2),
        "target_max": round(float(df[TARGET].max()), 2),
        "missing_traffic_index": int(df.traffic_index.isna().sum()),
        "missing_driver_experience": int(df.driver_experience_years.isna().sum()),
        "late_rate_vs_promised_eta": round(float((df[TARGET] > df.promised_eta_min).mean()), 4),
        "vehicle_share": df.vehicle_type.value_counts(normalize=True).round(3).to_dict(),
        "weather_share": df.weather.value_counts(normalize=True).round(3).to_dict(),
    }
    save_json(summary, RESULTS_DIR / "dataset_summary.json")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(df[TARGET], bins=50, color="#2563eb", edgecolor="white")
    ax[0].set(title="Distribution of delivery time", xlabel="minutes", ylabel="deliveries")
    for v, c in zip(["bike", "van", "truck"], ["#16a34a", "#2563eb", "#dc2626"]):
        g = df[df.vehicle_type == v].groupby("hour_of_day")[TARGET].mean()
        ax[1].plot(g.index, g.values, marker="o", label=v, color=c)
    ax[1].set(title="Mean delivery time by hour and vehicle", xlabel="hour of day", ylabel="minutes")
    ax[1].legend()
    sc = ax[2].scatter(df.distance_km, df[TARGET], c=df.traffic_index, s=6, cmap="viridis", alpha=0.6)
    ax[2].set(title="Time vs distance (colour = traffic index)", xlabel="distance (km)", ylabel="minutes")
    plt.colorbar(sc, ax=ax[2], label="traffic index")
    fig.suptitle("Exploratory view of the SIMULATED dataset", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_overview.png", dpi=150)
    print(f"Dataset written: {DATA_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
