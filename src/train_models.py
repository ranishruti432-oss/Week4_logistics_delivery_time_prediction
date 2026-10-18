"""Step 2 - train, cross-validate, tune and evaluate regression models.

Target : delivery_time_min (dispatch-to-delivery time in minutes)
Metrics: MAE, RMSE, R-squared
Flow   : split -> 5-fold CV of 6 default models -> randomized hyper-parameter
         search for 4 tunable models -> model selection by CV RMSE ->
         one-time evaluation on the untouched hold-out test set.
"""
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (KFold, RandomizedSearchCV, cross_val_predict,
                                     cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from common import (CATEGORICAL, DATA_PATH, FIG_DIR, MODEL_PATH, N_SAMPLES, NUMERIC_ALL,
                    RAW_FEATURES, RESULTS_DIR, SEED, TARGET, add_features, save_json, split_data)
from simulator import expected_time, generate_deliveries

warnings.filterwarnings("ignore")


def build_pipeline(estimator):
    """Preprocessing + model in ONE Pipeline (prevents train/test leakage)."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, NUMERIC_ALL),
                              ("cat", categorical, CATEGORICAL)])
    return Pipeline([("features", FunctionTransformer(add_features)),
                     ("prep", prep), ("model", estimator)])


def metrics(y_true, y_pred):
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, parse_dates=["dispatch_time"])
    train_df, test_df = split_data(df)
    X_train, y_train = train_df[RAW_FEATURES], train_df[TARGET]
    X_test, y_test = test_df[RAW_FEATURES], test_df[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)          # reporting / selection
    search_cv = KFold(n_splits=3, shuffle=True, random_state=SEED)   # inside hyper-parameter search

    # ---------- Stage 1: default models, 5-fold CV ----------
    default_models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
    }
    scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error",
               "R2": "r2"}
    rows = []
    for name, est in default_models.items():
        s = cross_validate(build_pipeline(est), X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        rows.append({"Model": name, "CV_MAE": -s["test_MAE"].mean(),
                     "CV_RMSE": -s["test_RMSE"].mean(), "CV_R2": s["test_R2"].mean(),
                     "CV_RMSE_std": s["test_RMSE"].std()})
        print(f"[default CV] {name:20s} RMSE={rows[-1]['CV_RMSE']:.3f}")
    stage1 = pd.DataFrame(rows)

    # ---------- Stage 2: hyper-parameter tuning ----------
    search_space = {
        "Ridge Regression": (Ridge(), {"model__alpha": [0.01, 0.1, 1, 10, 100]}),
        "Decision Tree": (DecisionTreeRegressor(random_state=SEED),
                          {"model__max_depth": [4, 6, 8, 10, 12],
                           "model__min_samples_leaf": [5, 10, 20, 40]}),
        "Random Forest": (RandomForestRegressor(random_state=SEED, n_jobs=-1),
                          {"model__n_estimators": [100, 150],
                           "model__max_depth": [10, 15, 20, None],
                           "model__min_samples_leaf": [3, 5, 10, 20],
                           "model__max_features": [0.4, 0.6, 0.8, 1.0]}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=SEED),
                              {"model__n_estimators": [150, 250, 350],
                               "model__learning_rate": [0.03, 0.05, 0.1],
                               "model__max_depth": [2, 3, 4],
                               "model__subsample": [0.7, 0.85, 1.0],
                               "model__min_samples_leaf": [5, 10, 20]}),
    }
    tuned, best_params, tune_rows = {}, {}, []
    for name, (est, grid) in search_space.items():
        n_iter = min(8, int(np.prod([len(v) for v in grid.values()])))
        rs = RandomizedSearchCV(build_pipeline(est), grid, n_iter=n_iter, cv=search_cv,
                                scoring="neg_root_mean_squared_error",
                                random_state=SEED, n_jobs=-1, refit=True)
        rs.fit(X_train, y_train)
        tuned[name] = rs.best_estimator_
        best_params[name] = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                             for k, v in rs.best_params_.items()}
        # CV MAE / R2 of the tuned configuration
        s = cross_validate(rs.best_estimator_, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        tune_rows.append({"Model": name, "Tuned_CV_MAE": -s["test_MAE"].mean(),
                          "Tuned_CV_RMSE": -s["test_RMSE"].mean(), "Tuned_CV_R2": s["test_R2"].mean()})
        print(f"[tuned CV]   {name:20s} RMSE={tune_rows[-1]['Tuned_CV_RMSE']:.3f}  {best_params[name]}")
    stage2 = pd.DataFrame(tune_rows)

    # Linear Regression has no hyper-parameters; keep as-is
    final_candidates = {"Mean baseline": build_pipeline(default_models["Mean baseline"]).fit(X_train, y_train),
                        "Linear Regression": build_pipeline(default_models["Linear Regression"]).fit(X_train, y_train)}
    final_candidates.update(tuned)

    # ---------- Stage 3: selection by CV RMSE, then hold-out test ----------
    cv_table = stage1.merge(stage2, on="Model", how="left")
    cv_table["Selection_RMSE"] = cv_table["Tuned_CV_RMSE"].fillna(cv_table["CV_RMSE"])
    best_name = cv_table.loc[cv_table[cv_table.Model != "Mean baseline"].Selection_RMSE.idxmin(), "Model"]

    test_rows = []
    preds = {}
    for name, pipe in final_candidates.items():
        p = pipe.predict(X_test)
        preds[name] = p
        m = metrics(y_test, p)
        test_rows.append({"Model": name, "Test_MAE": m["MAE"], "Test_RMSE": m["RMSE"], "Test_R2": m["R2"]})
    test_table = pd.DataFrame(test_rows)
    full = cv_table.merge(test_table, on="Model")
    full.round(4).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print("\nBest model (selected on CV RMSE):", best_name)

    best_pipe = final_candidates[best_name]
    joblib.dump(best_pipe, MODEL_PATH, compress=3)
    best_pred = preds[best_name]
    best_test = metrics(y_test, best_pred)

    # ---------- Residual buffer for risk alerts (out-of-fold on TRAIN only) ----------
    oof = cross_val_predict(best_pipe, X_train, y_train, cv=cv, n_jobs=1)
    resid = y_train.to_numpy() - oof
    buffer_q90 = float(np.quantile(resid, 0.90))
    interval_cov = float(np.mean(y_test.to_numpy() <= best_pred + buffer_q90))

    # ---------- Robustness: chronological split (train Jan-Sep, test Oct-Dec) ----------
    cut = df.dispatch_time.quantile(0.80)
    tr, te = df[df.dispatch_time <= cut], df[df.dispatch_time > cut]
    chrono_pipe = build_pipeline(best_pipe.named_steps["model"].__class__(**best_pipe.named_steps["model"].get_params()))
    chrono_pipe.fit(tr[RAW_FEATURES], tr[TARGET])
    chrono = metrics(te[TARGET], chrono_pipe.predict(te[RAW_FEATURES]))

    # ---------- Noise floor: error of a PERFECT predictor of the simulator's expected time ----------
    # (Only possible because the data are simulated; unexplainable noise + random incidents
    #  put a ceiling on the accuracy any model can reach.)
    clean = generate_deliveries(N_SAMPLES, SEED).loc[test_df.index]
    floor = metrics(y_test, expected_time(clean))

    # ---------- Error analysis ----------
    err = pd.DataFrame({"abs_err": np.abs(y_test.to_numpy() - best_pred),
                        "weather": test_df.weather.to_numpy(), "vehicle": test_df.vehicle_type.to_numpy(),
                        "peak": test_df.hour_of_day.isin({8, 9, 10, 17, 18, 19}).to_numpy()})
    err_by = {"weather": err.groupby("weather").abs_err.mean().round(3).to_dict(),
              "vehicle": err.groupby("vehicle").abs_err.mean().round(3).to_dict(),
              "peak_vs_offpeak": {"peak": round(float(err[err.peak].abs_err.mean()), 3),
                                  "off_peak": round(float(err[~err.peak].abs_err.mean()), 3)}}

    # ---------- Permutation importance (on raw columns, hold-out) ----------
    pi = permutation_importance(best_pipe, X_test, y_test, n_repeats=10, random_state=SEED,
                                scoring="neg_root_mean_squared_error", n_jobs=1)
    imp = pd.DataFrame({"feature": RAW_FEATURES, "rmse_increase": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("rmse_increase", ascending=False)
    imp.round(4).to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    # ---------- Figures ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    plot = full[full.Model != "Mean baseline"].copy()
    xs = np.arange(len(plot))
    ax.bar(xs - 0.2, plot.Selection_RMSE, 0.4, label="5-fold CV RMSE (train)", color="#93c5fd")
    ax.bar(xs + 0.2, plot.Test_RMSE, 0.4, label="Hold-out test RMSE", color="#2563eb")
    ax.set_xticks(xs); ax.set_xticklabels(plot.Model, rotation=15)
    ax.set(ylabel="RMSE (minutes)", title="Model comparison (lower is better)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "model_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(y_test, best_pred, s=8, alpha=0.5, color="#2563eb")
    lim = [0, max(y_test.max(), best_pred.max()) * 1.02]
    ax[0].plot(lim, lim, "r--", lw=1); ax[0].set(xlabel="Actual (min)", ylabel="Predicted (min)",
                                                  title=f"{best_name}: predicted vs actual")
    ax[1].hist(y_test.to_numpy() - best_pred, bins=50, color="#2563eb", edgecolor="white")
    ax[1].axvline(0, color="r", ls="--", lw=1); ax[1].set(xlabel="Residual = actual - predicted (min)",
                                                          ylabel="count", title="Residual distribution")
    fig.tight_layout(); fig.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    top = imp.head(10).iloc[::-1]
    ax.barh(top.feature, top.rmse_increase, xerr=top["std"], color="#2563eb")
    ax.set(xlabel="Increase in test RMSE when shuffled (min)", title="Permutation feature importance")
    fig.tight_layout(); fig.savefig(FIG_DIR / "feature_importance.png", dpi=150); plt.close(fig)

    summary = {
        "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
        "best_model": best_name, "best_params": best_params.get(best_name, {}),
        "all_best_params": best_params,
        "best_test": {k: round(v, 4) for k, v in best_test.items()},
        "chronological_test": {k: round(v, 4) for k, v in chrono.items()},
        "chronological_test_rows": int(len(te)),
        "noise_floor_perfect_predictor_test": {k: round(v, 4) for k, v in floor.items()},
        "residual_buffer_q90_min": round(buffer_q90, 3),
        "buffer_empirical_coverage_test": round(interval_cov, 4),
        "error_by_segment": err_by,
        "top_features": imp.head(8).round(4).to_dict(orient="records"),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    save_json(summary, RESULTS_DIR / "model_summary.json")
    print(full.round(3).to_string(index=False))
    print(f"Test (best): {summary['best_test']}")


if __name__ == "__main__":
    main()
