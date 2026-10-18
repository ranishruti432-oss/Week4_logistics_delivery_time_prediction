"""Step 3 - turn predictions into operational decisions.

Four optimisation modules built on the trained model:
  A. Dispatch planning        - shift flexible deliveries out of peak congestion
  B. Traffic-risk alerts      - Green / Amber / Red flags against the promised ETA
  C. Vehicle allocation       - cost-minimising assignment (Hungarian algorithm)
  D. Route planning           - traffic-aware stop sequencing (NN + 2-opt)

HOW RESULTS ARE EVALUATED (avoids circular reasoning)
  Decisions are made with the ML model's predictions. Outcomes are then scored
  with the simulator's noise-free "digital twin" (`simulator.expected_time`),
  not with the model itself. All gains are therefore SIMULATED gains; they show
  that the method works on this synthetic system, not that a specific company
  would save this exact amount.
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.isotonic import IsotonicRegression

from common import (DATA_PATH, FIG_DIR, MODEL_PATH, N_SAMPLES, PEAK_HOURS, RAW_FEATURES,
                    RESULTS_DIR, SEED, TARGET, save_json, split_data)
from simulator import (HOURS, SPEED_KMPH, expected_time, generate_deliveries,
                       hourly_traffic_profile, traffic_factor)

warnings.filterwarnings("ignore")
BLUE, RED, GREEN, AMBER, GREY = "#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#6b7280"


def load_everything():
    model = joblib.load(MODEL_PATH)
    masked = pd.read_csv(DATA_PATH, parse_dates=["dispatch_time"])   # what the model sees
    clean = generate_deliveries(N_SAMPLES, SEED)                      # digital-twin truth
    _, test_m = split_data(masked)
    test_c = clean.loc[test_m.index]
    assert (test_m.delivery_id.values == test_c.delivery_id.values).all()
    with open(RESULTS_DIR / "model_summary.json") as f:
        msum = json.load(f)
    return model, test_m.reset_index(drop=True), test_c.reset_index(drop=True), msum


# ----------------------------------------------------------------------------
# A. Dispatch planning
# ----------------------------------------------------------------------------
def dispatch_planning(model, test_m, test_c):
    # (1) Hourly "best dispatch window" table for a fixed reference delivery
    ref = pd.DataFrame({"distance_km": 8.0, "package_weight_kg": 3.0, "stops_on_route": 5,
                        "hour_of_day": HOURS, "day_of_week": 2,
                        "traffic_index": hourly_traffic_profile(HOURS).round(1),
                        "driver_experience_years": 5.0, "vehicle_load_pct": 60.0,
                        "vehicle_type": "van", "weather": "clear", "zone": "North",
                        "priority": "standard"})
    hourly_pred = model.predict(ref[RAW_FEATURES])
    hourly_true = expected_time(ref)

    # (2) Shift simulation for flexible (standard) deliveries dispatched at peak
    flex = test_m.priority.eq("standard") & test_m.hour_of_day.isin(PEAK_HOURS)
    idx = np.where(flex)[0]
    MAX_SHIFT, MIN_GAIN = 2, 2.0   # at most +2 h; act only if predicted saving >= 2 min

    def variant(frame, s):
        v = frame.iloc[idx].copy()
        h0 = v.hour_of_day.to_numpy()
        h1 = np.minimum(h0 + s, HOURS.max())
        delta = hourly_traffic_profile(h1) - hourly_traffic_profile(h0)
        v["traffic_index"] = np.clip(v.traffic_index + delta, 0, 10)
        v["hour_of_day"] = h1
        return v, h1 - h0

    pred_s, true_s = [], []
    for s in range(MAX_SHIFT + 1):
        vm, _ = variant(test_m, s)
        vc, _ = variant(test_c, s)
        pred_s.append(model.predict(vm[RAW_FEATURES]))
        true_s.append(expected_time(vc))
    pred_s, true_s = np.array(pred_s), np.array(true_s)

    best_pred_s = pred_s.argmin(axis=0)
    gain_pred = pred_s[0] - pred_s.min(axis=0)
    chosen = np.where(gain_pred >= MIN_GAIN, best_pred_s, 0)
    realised = true_s[0] - true_s[chosen, np.arange(len(idx))]
    oracle = true_s[0] - true_s.min(axis=0)
    moved = chosen > 0
    out = {
        "peak_standard_deliveries": int(len(idx)),
        "share_of_all_test_deliveries_pct": round(100 * len(idx) / len(test_m), 1),
        "deliveries_shifted": int(moved.sum()),
        "shifted_share_pct": round(100 * moved.mean(), 1),
        "avg_shift_hours_when_shifted": round(float(chosen[moved].mean()), 2) if moved.any() else 0.0,
        "avg_true_minutes_saved_per_shifted_delivery": round(float(realised[moved].mean()), 2) if moved.any() else 0.0,
        "total_vehicle_minutes_saved": round(float(realised.sum()), 1),
        "avg_minutes_saved_per_peak_standard_delivery": round(float(realised.mean()), 2),
        "oracle_total_vehicle_minutes_saved": round(float(oracle.sum()), 1),
        "pct_of_oracle_captured": round(100 * float(realised.sum()) / float(oracle.sum()), 1),
        "baseline_total_minutes_peak_standard": round(float(true_s[0].sum()), 1),
        "pct_reduction_in_on_road_minutes": round(100 * float(realised.sum()) / float(true_s[0].sum()), 2),
        "hourly_table": [{"hour": int(h), "model_pred_min": round(float(p), 1),
                          "true_expected_min": round(float(t), 1)}
                         for h, p, t in zip(HOURS, hourly_pred, hourly_true)],
        "best_hour": int(HOURS[hourly_pred.argmin()]), "worst_hour": int(HOURS[hourly_pred.argmax()]),
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(HOURS, hourly_pred, marker="o", color=BLUE, label="Model prediction")
    ax.plot(HOURS, hourly_true, ls="--", color=GREY, label="Simulator truth (noise-free)")
    for lo, hi in [(8, 11), (17, 20)]:
        ax.axvspan(lo - 0.5, hi - 0.5, color=RED, alpha=0.08)
    ax.set(xlabel="Dispatch hour", ylabel="Predicted delivery time (min)",
           title="Reference delivery (8 km, 5 stops, van, clear weather): time by dispatch hour")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "dispatch_hourly.png", dpi=150); plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# B. Traffic-risk alerts
# ----------------------------------------------------------------------------
def risk_alerts(model, test_m, buffer_q90):
    pred = model.predict(test_m[RAW_FEATURES])
    promised = test_m.promised_eta_min.to_numpy()
    actual = test_m[TARGET].to_numpy()
    late = actual > promised

    level = np.where(pred > promised, "Red",
                     np.where(pred + buffer_q90 > promised, "Amber", "Green"))
    table = {}
    for lv in ["Red", "Amber", "Green"]:
        m = level == lv
        table[lv] = {"deliveries": int(m.sum()), "share_pct": round(100 * m.mean(), 1),
                     "actual_late_rate_pct": round(100 * float(late[m].mean()), 1) if m.any() else None}

    def prf(flag):
        tp = int((flag & late).sum())
        return {"flagged_pct": round(100 * flag.mean(), 1),
                "precision_pct": round(100 * tp / max(flag.sum(), 1), 1),
                "recall_pct": round(100 * tp / max(late.sum(), 1), 1)}

    flagged = level != "Green"
    res = {
        "overall_late_rate_pct": round(100 * float(late.mean()), 1),
        "alert_levels": table,
        "red_or_amber": prf(flagged),
        "red_only": prf(level == "Red"),
        "baseline_traffic_index_ge_7_rule": prf((test_m.traffic_index.fillna(0) >= 7).to_numpy()),
        "eta_mae_model_min": round(float(np.mean(np.abs(actual - pred))), 2),
        "eta_mae_rule_of_thumb_min": round(float(np.mean(np.abs(actual - promised))), 2),
        "buffer_used_min": round(buffer_q90, 2),
    }
    # Reason tags for flagged deliveries (simple, explainable rules)
    tf = test_m.traffic_index.fillna(0).to_numpy()
    bad_w = test_m.weather.isin(["rain", "fog", "storm"]).to_numpy()
    pk = test_m.hour_of_day.isin(PEAK_HOURS).to_numpy()
    reason = np.where(tf >= 7, "High traffic", np.where(bad_w, "Adverse weather",
                      np.where(pk, "Peak hour", "Long distance / many stops")))
    res["alert_reasons_pct"] = (pd.Series(reason[flagged]).value_counts(normalize=True)
                                .mul(100).round(1).to_dict())

    fig, ax = plt.subplots(figsize=(6.5, 4))
    labs = ["Green", "Amber", "Red"]
    vals = [table[l]["actual_late_rate_pct"] for l in labs]
    bars = ax.bar(labs, vals, color=[GREEN, AMBER, RED])
    ax.axhline(100 * late.mean(), color=GREY, ls="--", label=f"Overall late rate {100*late.mean():.1f}%")
    for b, v, l in zip(bars, vals, labs):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%\n(n={table[l]['deliveries']})", ha="center", fontsize=9)
    ax.set(ylabel="Actual late-delivery rate (%)", ylim=(0, max(vals) * 1.25),
           title="Late rate by alert level (hold-out test set)")
    ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(FIG_DIR / "alert_levels.png", dpi=150); plt.close(fig)
    return res


# ----------------------------------------------------------------------------
# C. Vehicle allocation (assignment problem)
# ----------------------------------------------------------------------------
RATE = {"bike": 4.0, "van": 7.0, "truck": 12.0}       # INR per on-road minute (assumed)
FIXED = {"bike": 20.0, "van": 60.0, "truck": 150.0}   # INR fixed cost per trip (assumed)
LATE_PENALTY = 10.0                                    # INR per minute beyond promised ETA (assumed)
FLEET = ["bike"] * 8 + ["van"] * 12 + ["truck"] * 4    # 24 vehicles for 20 parcels per wave
N_PER_BATCH, N_BATCHES = 20, 200


def feasible(frame, v):
    w, d = frame.package_weight_kg.to_numpy(), frame.distance_km.to_numpy()
    if v == "bike":
        return (w <= 10) & (d <= 15)
    if v == "van":
        return w <= 150
    return np.ones(len(frame), dtype=bool)


def vehicle_allocation(model, test_m, test_c):
    promised = test_m.promised_eta_min.to_numpy()
    pred, true, feas = {}, {}, {}
    for v in ["bike", "van", "truck"]:
        m, c = test_m.copy(), test_c.copy()
        m["vehicle_type"] = v
        c["vehicle_type"] = v
        pred[v] = model.predict(m[RAW_FEATURES])
        true[v] = expected_time(c)
        feas[v] = feasible(test_m, v)

    def cost(t, i, v, use_late=True):
        return FIXED[v] + RATE[v] * t + LATE_PENALTY * np.maximum(0, t - promised[i])

    rng = np.random.default_rng(SEED)
    INF = 1e9
    rows = []
    for _ in range(N_BATCHES):
        b = rng.choice(len(test_m), N_PER_BATCH, replace=False)
        C_pred = np.full((N_PER_BATCH, len(FLEET)), INF)
        C_true = np.full((N_PER_BATCH, len(FLEET)), INF)
        for j, v in enumerate(FLEET):
            ok = feas[v][b]
            C_pred[ok, j] = cost(pred[v][b], b, v)[ok]
            C_true[ok, j] = cost(true[v][b], b, v)[ok]

        r_opt, c_opt = linear_sum_assignment(C_pred)                # model-guided
        r_orc, c_orc = linear_sum_assignment(C_true)                # perfect information (bound)

        # Baseline: today's rule-of-thumb dispatcher (first free vehicle in preference order)
        free = list(range(len(FLEET)))
        base_assign = {}
        for i_loc, i in enumerate(b):
            w, d = test_m.package_weight_kg[i], test_m.distance_km[i]
            pref = ["truck"] if w > 150 else (["van", "truck"] if not (w <= 10 and d <= 15)
                                              else ["bike", "van", "truck"])
            for v in pref:
                cand = [j for j in free if FLEET[j] == v]
                if cand:
                    base_assign[i_loc] = cand[0]
                    free.remove(cand[0])
                    break

        def score(assign):
            tot_cost = tot_min = late_n = 0.0
            for i_loc, j in assign:
                v, i = FLEET[j], b[i_loc]
                t = true[v][i]
                tot_cost += FIXED[v] + RATE[v] * t + LATE_PENALTY * max(0, t - promised[i])
                tot_min += t
                late_n += t > promised[i]
            return tot_cost, tot_min, late_n

        sb = score(base_assign.items())
        so = score(zip(r_opt, c_opt))
        sr = score(zip(r_orc, c_orc))
        rows.append({"base_cost": sb[0], "base_min": sb[1], "base_late": sb[2],
                     "opt_cost": so[0], "opt_min": so[1], "opt_late": so[2],
                     "orc_cost": sr[0], "orc_min": sr[1], "orc_late": sr[2]})
    R = pd.DataFrame(rows)
    res = {
        "batches": N_BATCHES, "parcels_per_batch": N_PER_BATCH, "fleet_size": len(FLEET),
        "fleet_mix": {"bike": 8, "van": 12, "truck": 4},
        "assumed_cost": {"rate_inr_per_min": RATE, "fixed_inr_per_trip": FIXED,
                         "late_penalty_inr_per_min": LATE_PENALTY},
        "avg_cost_per_batch_inr": {"rule_based": round(R.base_cost.mean(), 1),
                                   "model_guided": round(R.opt_cost.mean(), 1),
                                   "perfect_information_bound": round(R.orc_cost.mean(), 1)},
        "avg_vehicle_minutes_per_batch": {"rule_based": round(R.base_min.mean(), 1),
                                          "model_guided": round(R.opt_min.mean(), 1),
                                          "perfect_information_bound": round(R.orc_min.mean(), 1)},
        "avg_late_parcels_per_batch": {"rule_based": round(R.base_late.mean(), 2),
                                       "model_guided": round(R.opt_late.mean(), 2),
                                       "perfect_information_bound": round(R.orc_late.mean(), 2)},
        "cost_reduction_pct": round(100 * (1 - R.opt_cost.mean() / R.base_cost.mean()), 1),
        "late_reduction_pct": round(100 * (1 - R.opt_late.mean() / max(R.base_late.mean(), 1e-9)), 1),
        "pct_of_bound_captured": round(100 * (R.base_cost.mean() - R.opt_cost.mean())
                                       / (R.base_cost.mean() - R.orc_cost.mean()), 1),
        "batches_where_model_guided_cheaper_pct": round(100 * float((R.opt_cost < R.base_cost).mean()), 1),
    }
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    names = ["Rule-based", "Model-guided", "Perfect info"]
    cols = [GREY, BLUE, GREEN]
    ac = res["avg_cost_per_batch_inr"]; al = res["avg_late_parcels_per_batch"]
    ax[0].bar(names, list(ac.values()), color=cols); ax[0].set(ylabel="INR per batch (simulated)", title="Avg cost per dispatch wave")
    ax[1].bar(names, list(al.values()), color=cols); ax[1].set(ylabel="late parcels per batch", title="Avg late parcels per wave")
    for a in ax:
        a.tick_params(axis="x", labelrotation=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "allocation_comparison.png", dpi=150); plt.close(fig)
    return res


# ----------------------------------------------------------------------------
# D. Route planning (traffic-aware stop sequencing)
# ----------------------------------------------------------------------------
def learned_minutes_per_km(model):
    """Marginal minutes per km at traffic index t, learned by the model.

    pk(t) = [pred(15 km, t) - pred(5 km, t)] / 10  - the difference cancels
    loading / queueing time, leaving pure driving time per km.
    """
    grid = np.arange(0, 10.5, 0.5)
    base = dict(package_weight_kg=3.0, stops_on_route=3, hour_of_day=12, day_of_week=2,
                driver_experience_years=5.0, vehicle_load_pct=60.0, vehicle_type="van",
                weather="clear", zone="North", priority="standard")
    def pr(d):
        f = pd.DataFrame({"distance_km": d, "traffic_index": grid, **base})
        return model.predict(f[RAW_FEATURES])
    pk = (pr(15.0) - pr(5.0)) / 10.0
    return grid, np.maximum(pk, 0.3)


def traffic_field(rng):
    centres = np.array([[10, 10]] + [rng.uniform(2, 18, 2) for _ in range(2)])
    amps = np.array([2.5, rng.uniform(4, 6), rng.uniform(4, 6)])   # central core + 2 hotspots
    sig = np.array([4.0, 2.5, 2.5])
    def t(p):
        p = np.atleast_2d(p)
        z = 1.5 + sum(a * np.exp(-((p - c) ** 2).sum(1) / (2 * s ** 2)) for a, c, s in zip(amps, centres, sig))
        return np.clip(z, 0, 10)
    return t


def tour_len(order, M):
    return sum(M[order[i], order[i + 1]] for i in range(len(order) - 1))


def nearest_neighbour(M):
    n, order, left = len(M), [0], set(range(1, len(M)))
    while left:
        nxt = min(left, key=lambda j: M[order[-1], j]); order.append(nxt); left.remove(nxt)
    return order + [0]


def two_opt(order, M):
    best, improved = order[:], True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for k in range(i + 1, len(best) - 1):
                new = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                if tour_len(new, M) < tour_len(best, M) - 1e-9:
                    best, improved = new, True
    return best


def route_planning(model):
    grid, pk_raw = learned_minutes_per_km(model)
    # Congestion can only slow traffic down, so enforce a monotone (non-decreasing) curve
    pk = IsotonicRegression(increasing=True).fit_transform(grid, pk_raw)
    true_pk = 60.0 / SPEED_KMPH["van"] * traffic_factor(grid, "van")
    rng = np.random.default_rng(SEED + 7)
    N_ROUTES, N_STOPS, CIRCUITY = 300, 8, 1.3
    out_rows, store = [], []
    for r in range(N_ROUTES):
        tfun = traffic_field(rng)
        pts = np.vstack([[10, 10], rng.uniform(0, 20, (N_STOPS, 2))])
        n = len(pts)
        D = CIRCUITY * np.linalg.norm(pts[:, None] - pts[None], axis=2)
        t_hat = np.clip(tfun(pts) + rng.normal(0, 0.7, n), 0, 10)       # noisy traffic feed at stops
        M_pred = np.zeros((n, n)); M_true = np.zeros((n, n))   # planner's view / ground truth
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                M_pred[i, j] = D[i, j] * np.interp((t_hat[i] + t_hat[j]) / 2, grid, pk)
                s = np.linspace(0, 1, 7)[:, None]
                line = pts[i] * (1 - s) + pts[j] * s
                M_true[i, j] = D[i, j] * 60.0 / SPEED_KMPH["van"] * float(np.mean(
                    traffic_factor(tfun(line), "van")))
        A = nearest_neighbour(D)                       # distance-only greedy (typical baseline)
        B = two_opt(A, D)                              # better optimiser, still ignores traffic
        C = two_opt(nearest_neighbour(M_pred), M_pred) # traffic-aware (model-derived times)
        Dor = two_opt(nearest_neighbour(M_true), M_true)  # ceiling: perfect traffic knowledge
        ta, tb, tc, td = (tour_len(o, M_true) for o in (A, B, C, Dor))
        out_rows.append({"A": ta, "B": tb, "C": tc, "O": td})
        store.append((pts, tfun, A, C, ta, tc))
    R = pd.DataFrame(out_rows)
    res = {
        "routes_simulated": N_ROUTES, "stops_per_route": N_STOPS,
        "avg_true_route_minutes": {"A_nearest_neighbour_distance": round(R.A.mean(), 1),
                                   "B_2opt_distance": round(R.B.mean(), 1),
                                   "C_2opt_traffic_aware_model": round(R.C.mean(), 1),
                                   "Oracle_2opt_perfect_traffic_knowledge": round(R.O.mean(), 1)},
        "reduction_oracle_vs_A_pct": round(100 * (1 - R.O.mean() / R.A.mean()), 1),
        "extra_reduction_oracle_vs_B_pct": round(100 * (1 - R.O.mean() / R.B.mean()), 2),
        "reduction_C_vs_A_pct": round(100 * (1 - R.C.mean() / R.A.mean()), 1),
        "reduction_B_vs_A_pct": round(100 * (1 - R.B.mean() / R.A.mean()), 1),
        "extra_reduction_C_vs_B_pct": round(100 * (1 - R.C.mean() / R.B.mean()), 2),
        "routes_where_C_faster_than_A_pct": round(100 * float((R.C < R.A - 1e-9).mean()), 1),
        "minutes_saved_per_route_C_vs_A": round(R.A.mean() - R.C.mean(), 1),
        "model_learned_min_per_km_at_t0_t5_t10": [round(float(np.interp(t, grid, pk)), 2) for t in (0, 5, 10)],
        "true_min_per_km_at_t0_t5_t10": [round(float(np.interp(t, grid, true_pk)), 2) for t in (0, 5, 10)],
    }
    # Representative example: the route whose saving is the MEDIAN (not cherry-picked)
    sav = np.array([s[4] - s[5] for s in store]); k = int(np.argsort(sav)[len(sav) // 2])
    pts, tfun, A, C, ta, tc = store[k]
    xs = np.linspace(0, 20, 80); X, Y = np.meshgrid(xs, xs)
    Z = tfun(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, order, ttl, tt in [(ax[0], A, "A: distance-only nearest neighbour", ta),
                              (ax[1], C, "C: 2-opt + model traffic-aware times", tc)]:
        im = a.contourf(X, Y, Z, levels=12, cmap="YlOrRd", alpha=0.75)
        a.plot(pts[order, 0], pts[order, 1], "-o", color="#111827", lw=1.6, ms=5)
        a.scatter(*pts[0], marker="s", s=90, color=BLUE, zorder=5, label="Depot")
        for rank, node in enumerate(order[1:-1], 1):
            a.annotate(str(rank), pts[node], textcoords="offset points", xytext=(4, 4), fontsize=8)
        a.set(title=f"{ttl}\ntrue time = {tt:.0f} min", xlim=(0, 20), ylim=(0, 20)); a.legend(loc="lower left")
    fig.colorbar(im, ax=ax, label="congestion index (0-10)", shrink=0.8)
    fig.suptitle("Median-saving example route (simulated city, 8 stops)", fontsize=11, y=1.04)
    fig.savefig(FIG_DIR / "route_example.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return res


def main():
    model, test_m, test_c, msum = load_everything()
    summary = {
        "note": "All outcomes are SIMULATED (digital-twin evaluation). Not real company savings.",
        "A_dispatch_planning": dispatch_planning(model, test_m, test_c),
        "B_risk_alerts": risk_alerts(model, test_m, msum["residual_buffer_q90_min"]),
        "C_vehicle_allocation": vehicle_allocation(model, test_m, test_c),
        "D_route_planning": route_planning(model),
    }
    save_json(summary, RESULTS_DIR / "optimization_summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "note"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
