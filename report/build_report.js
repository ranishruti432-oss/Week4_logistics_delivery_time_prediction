// Builds report/Logistics_Predictive_Modeling_Report.docx
// All numbers are read from /results so the report always matches the code output.
// Usage: node build_report.js   (requires the npm package "docx")
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, Header, Footer,
  PageNumber, AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, LevelFormat,
  PageBreak,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const R = (f) => path.join(ROOT, "results", f);
const ds = JSON.parse(fs.readFileSync(R("dataset_summary.json")));
const ms = JSON.parse(fs.readFileSync(R("model_summary.json")));
const op = JSON.parse(fs.readFileSync(R("optimization_summary.json")));
const parseCsv = (f) => {
  const [h, ...rows] = fs.readFileSync(R(f), "utf8").trim().split("\n").map((l) => l.split(","));
  return rows.map((r) => Object.fromEntries(h.map((k, i) => [k, r[i]])));
};
const cmp = parseCsv("model_comparison.csv");
const fi = parseCsv("feature_importance.csv");
const verifyStatus = fs.existsSync(R("verification_status.txt"))
  ? fs.readFileSync(R("verification_status.txt"), "utf8").trim() : "";

const n1 = (x) => Number(x).toFixed(1);
const n2 = (x) => Number(x).toFixed(2);
const n3 = (x) => Number(x).toFixed(3);
const n4 = (x) => Number(x).toFixed(4);
const inr = (x) => Number(x).toLocaleString("en-IN", { maximumFractionDigits: 0 });

// ---------- style constants ----------
const NAVY = "1F3A5F", BLUE = "2563EB", LIGHT = "EEF3FA", GREYBG = "F3F4F6", BORDER = "C9D3E0";
const W = 9026; // A4 content width in DXA with 1-inch margins

// ---------- helpers ----------
function runs(text, base = {}) {
  // **bold**, `code`
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean).map((t) => {
    if (t.startsWith("**")) return new TextRun({ ...base, text: t.slice(2, -2), bold: true });
    if (t.startsWith("`")) return new TextRun({ ...base, text: t.slice(1, -1), font: "Consolas", size: (base.size || 22) - 2, color: "9F1239" });
    return new TextRun({ ...base, text: t });
  });
}
const P = (text, o = {}) => new Paragraph({
  children: runs(text, o.run || {}), alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { before: o.before ?? 0, after: o.after ?? 120, line: 288 }, keepNext: o.keepNext,
});
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], keepNext: true });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)], keepNext: true });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)], keepNext: true });
const B = (text, level = 0) => new Paragraph({
  numbering: { reference: "bullets", level }, children: runs(text), spacing: { after: 60, line: 276 },
});
const N = (text) => new Paragraph({
  numbering: { reference: "numbers", level: 0 }, children: runs(text), spacing: { after: 60, line: 276 },
});
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });
const spacer = (after = 120) => new Paragraph({ children: [], spacing: { after } });

function callout(title, text, color = BLUE, fill = LIGHT) {
  const bd = { style: BorderStyle.SINGLE, size: 6, color, space: 6 };
  return [
    new Paragraph({
      children: [new TextRun({ text: title, bold: true, color })], keepNext: true,
      shading: { type: ShadingType.CLEAR, fill }, spacing: { before: 120, after: 0 },
      border: { left: { ...bd, size: 24 }, top: bd, right: bd },
      indent: { left: 120, right: 120 },
    }),
    new Paragraph({
      children: runs(text), alignment: AlignmentType.JUSTIFIED,
      shading: { type: ShadingType.CLEAR, fill }, spacing: { after: 160, line: 276 },
      border: { left: { ...bd, size: 24 }, bottom: bd, right: bd },
      indent: { left: 120, right: 120 },
    }),
  ];
}

function code(title, lines) {
  const bd = { style: BorderStyle.SINGLE, size: 4, color: BORDER, space: 4 };
  const out = [];
  if (title) out.push(new Paragraph({ children: [new TextRun({ text: title, bold: true, size: 20, color: NAVY })], keepNext: true, spacing: { before: 120, after: 40 } }));
  lines.forEach((ln, i) => out.push(new Paragraph({
    children: [new TextRun({ text: ln === "" ? " " : ln, font: "Consolas", size: 17 })],
    shading: { type: ShadingType.CLEAR, fill: GREYBG }, spacing: { after: 0, line: 240 },
    border: { left: { ...bd, size: 18, color: BLUE }, top: i === 0 ? bd : undefined, bottom: i === lines.length - 1 ? bd : undefined, right: bd },
    indent: { left: 100, right: 100 }, keepLines: true,
  })));
  out.push(spacer(100));
  return out;
}

function table(headers, rows, widths, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const sz = o.size || 19;
  const b = { style: BorderStyle.SINGLE, size: 4, color: BORDER };
  const borders = { top: b, bottom: b, left: b, right: b };
  const cell = (txt, i, head, zebra) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders,
    shading: { type: ShadingType.CLEAR, fill: head ? NAVY : zebra ? "F7F9FC" : "FFFFFF" },
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({
      alignment: (o.align && o.align[i]) || AlignmentType.LEFT, spacing: { after: 0 },
      children: runs(String(txt), { size: sz, bold: head, color: head ? "FFFFFF" : undefined }),
    })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((h, i) => cell(h, i, true, false)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, ri % 2 === 1)) })),
    ],
  });
}
const caption = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 },
  children: [new TextRun({ text: t, italics: true, size: 19, color: "4B5563" })],
});
const tcap = (t) => new Paragraph({
  spacing: { before: 160, after: 80 }, keepNext: true,
  children: [new TextRun({ text: t, bold: true, size: 20, color: NAVY })],
});

function figure(file, widthPx, cap) {
  const buf = fs.readFileSync(R(path.join("figures", file)));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const height = Math.round(widthPx * h / w);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: widthPx, height },
        altText: { title: cap, description: cap, name: file } })],
    }),
    caption(cap),
  ];
}

const hl = (t) => new TextRun({ text: t, highlight: "yellow", size: 24 });

// ---------- derived numbers ----------
const best = ms.best_test, floor = ms.noise_floor_perfect_predictor_test, chrono = ms.chronological_test;
const row = (name) => cmp.find((r) => r.Model === name);
const lin = row("Linear Regression"), gb = row("Gradient Boosting"), rf = row("Random Forest"),
  dt = row("Decision Tree"), rd = row("Ridge Regression"), base = row("Mean baseline");
const rmseGainVsLinear = (100 * (lin.Test_RMSE - gb.Test_RMSE) / lin.Test_RMSE);
const maeGainVsLinear = (100 * (lin.Test_MAE - gb.Test_MAE) / lin.Test_MAE);
const A = op.A_dispatch_planning, Bk = op.B_risk_alerts, C = op.C_vehicle_allocation, D = op.D_route_planning;
const hours = A.hourly_table;
const H = (h) => hours.find((x) => x.hour === h);
const peakGap = H(A.worst_hour).model_pred_min - H(A.best_hour).model_pred_min;

// ====================================================================
// CONTENT
// ====================================================================
const cover = [
  new Paragraph({ spacing: { before: 900, after: 120 }, alignment: AlignmentType.LEFT,
    children: [new TextRun({ text: "WEEK 4 TASK REPORT", bold: true, size: 24, color: BLUE, characterSpacing: 60 })] }),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Predictive Modeling and Optimization in Logistics Systems", bold: true, size: 56, color: NAVY })] }),
  new Paragraph({ spacing: { after: 360 }, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: BLUE, space: 8 } },
    children: [new TextRun({ text: "Delivery-time forecasting with machine learning and data-driven dispatch, alerting, vehicle-allocation and route-planning optimization", size: 26, color: "374151" })] }),
  ...callout("DATA DISCLOSURE",
    "All data used in this project is **simulated** (synthetically generated by the Python code in the accompanying repository). It is **not** real company data, and no real organisation, customer, driver or vehicle is represented. Every result, including the cost and time savings in the optimisation section, is a result on this simulated system and should be read as a demonstration of the method, not as a measured saving for any real business.",
    "B45309", "FFF7E6"),
  spacer(240),
  new Paragraph({ spacing: { before: 200, after: 0 }, children: [new TextRun({ text: "Prepared by ", size: 28, color: "4B5563" }), new TextRun({ text: "Shruti Rani", bold: true, size: 28, color: NAVY })] }),
  pageBreak(),
];

const contents = [
  H1("Contents"),
  ...["Abstract", "1. Introduction and Problem Definition", "2. Dataset (Simulated)", "3. Methodology and Model Selection",
    "4. Evaluation Metrics and Validation Strategy", "5. Results and Model Comparison", "6. Optimization Strategies",
    "7. Recommendations and Implementation Roadmap", "8. Limitations and Threats to Validity",
    "9. Reproducibility and Code Verification", "10. Conclusion", "References", "Appendix A: Repository Structure",
    "Appendix B: Tuned Hyper-parameters"].map((t) => new Paragraph({ children: [new TextRun({ text: t, size: 23 })], spacing: { after: 80 }, indent: { left: 200 } })),
  pageBreak(),
];

const abstract = [
  H1("Abstract"),
  P(`Late and unpredictable deliveries inflate cost, hurt customer trust and waste fleet capacity. This project builds a machine-learning model that forecasts **dispatch-to-delivery time (minutes)** for last-mile deliveries and then uses the forecasts to drive four operational decisions: dispatch timing, traffic-risk alerts, vehicle allocation and route sequencing.`),
  P(`Because real operational data was not available, a **simulated dataset of ${ds.n_rows.toLocaleString()} deliveries** was generated with a documented data-generating process (distance, stops, vehicle type, weather, congestion, hour of day and more, plus random noise and rare incidents). Six models were compared with 5-fold cross-validation: a mean baseline, linear regression, ridge regression, a decision tree, a random forest and gradient boosting. Four were tuned by randomized search. Models were compared with **MAE, RMSE and R²**, and the winner was chosen on cross-validation RMSE only; the hold-out test set was used once.`),
  P(`**${ms.best_model}** performed best on the hold-out test set with **MAE = ${n2(best.MAE)} min, RMSE = ${n2(best.RMSE)} min and R² = ${n3(best.R2)}**, compared with an MAE of ${n2(Bk.eta_mae_rule_of_thumb_min)} min for a rule-of-thumb ETA. For reference, a perfect predictor of the simulator's expected time would still show RMSE = ${n2(floor.RMSE)} min because of irreducible noise, so the model is close to the achievable ceiling on this data. Using the predictions, a risk-alert scheme flagged ${n1(Bk.red_or_amber.recall_pct)}% of late deliveries in advance, a model-guided vehicle allocation cut simulated cost by ${n1(C.cost_reduction_pct)}% and late parcels by ${n1(C.late_reduction_pct)}% versus a rule-based dispatcher, and shifting flexible peak-hour deliveries reduced their on-road minutes by ${n1(A.pct_reduction_in_on_road_minutes)}%. A negative finding is also reported honestly: traffic-aware route sequencing added almost nothing beyond a good 2-opt optimiser in this setting.`),
  pageBreak(),
];

// ---------- 1. Introduction ----------
const s1 = [
  H1("1. Introduction and Problem Definition"),
  H2("1.1 Logistics context"),
  P("In last-mile delivery the customer is quoted an estimated delivery time at dispatch. Most operators still quote it with simple rules such as a fixed handling time plus minutes per kilometre. Those rules ignore congestion, weather and vehicle type, so they are right on quiet clear days and wrong precisely when the operation is under stress. Late deliveries then trigger penalties, re-deliveries and support calls, while pessimistic quotes waste vehicle time."),
  H2("1.2 Prediction problem"),
  table(["Item", "Definition"], [
    ["Task type", "Supervised regression"],
    ["Target variable", "`delivery_time_min`: minutes from vehicle dispatch to the parcel being delivered"],
    ["Prediction moment", "At dispatch time. Only information available at that moment is used as input (no post-delivery data, so no target leakage)"],
    ["Unit of analysis", "One delivery (one parcel on one vehicle run)"],
    ["Business use", "Better ETAs, early warning of likely-late deliveries, and inputs to dispatch / allocation / routing decisions"],
  ], [2400, 6626]),
  spacer(80),
  P(`The rule-of-thumb ETA that the business currently quotes (\`promised_eta_min\` = 14 + 3.0 × km + 2.0 × (stops − 1)) is deliberately **excluded from the model's features**. It serves as the business baseline for comparison: ${n1(100 * ds.late_rate_vs_promised_eta)}% of deliveries in the full simulated dataset exceed it.`),
  H2("1.3 Objectives"),
  N("Build and validate a predictive model for delivery time, comparing simple and advanced techniques fairly."),
  N("Explain why MAE, RMSE and R² were chosen and how the models were compared."),
  N("Convert predictions into practical decisions: dispatch planning, traffic-risk alerts, vehicle allocation and route-planning improvements."),
  N("Make the entire study reproducible, with a script that re-runs everything and checks that the results match."),
  H2("1.4 Flow of the study"),
  P("The report follows the workflow of the code: **data simulation → preprocessing → model comparison with cross-validation → tuning → hold-out evaluation → optimisation modules → recommendations**. Code excerpts are shown next to the corresponding analysis; the complete code is in the repository."),
];

// ---------- 2. Dataset ----------
const s2 = [
  H1("2. Dataset (Simulated)"),
  ...callout("Dataset transparency",
    "The dataset is simulated. It was produced by `src/simulator.py` with a fixed random seed (42). The relationships inside it are assumptions based on common logistics knowledge, not measurements from a real fleet. Because the model is trained on data whose structure is known, its accuracy here is likely **higher than it would be on messy real-world data**.",
    "B45309", "FFF7E6"),
  H2("2.1 Why simulate and how the data was generated"),
  P("No operational dataset was available for this assignment, and real delivery data is usually confidential. A transparent simulator also has a research advantage: because the true expected delivery time is known, optimisation decisions can be scored against ground truth rather than against the model's own predictions (Section 6). The generating process combines the following effects:"),
  B("**Driving time** = distance ÷ vehicle speed (bike 16, van 28, truck 22 km/h) × a non-linear congestion factor × a weather factor × small driver-experience and load effects."),
  B("**Congestion** follows a daily profile with a morning peak near 09:00 and an evening peak near 18:00, is higher in the central zone, in bad weather and on weekdays, and is clipped to a 0–10 index."),
  B("**Fixed components**: loading time and handling time that grow with parcel weight, and waiting time that grows with the number of stops on the run."),
  B("**Weather**: bikes are hurt more by rain and storms than vans or trucks; congestion falls less for bikes."),
  B("**Noise**: multiplicative log-normal noise (σ = 0.10) plus a 3% chance of an unpredictable incident adding 8–30 minutes."),
  B("**Dispatch practice**: small, short-distance parcels may go by bike or van; heavy or long ones go by van or truck, mimicking present-day behaviour."),
  H2("2.2 Features and target"),
  table(["Feature", "Type", "Description"], [
    ["distance_km", "numeric", "Route distance to the delivery point (0.5–40 km)"],
    ["package_weight_kg", "numeric", "Parcel weight (0.1–300 kg)"],
    ["stops_on_route", "numeric", "Number of drops on the vehicle's run (1–14)"],
    ["hour_of_day", "numeric", "Dispatch hour (07–20)"],
    ["day_of_week", "numeric", "0 = Monday … 6 = Sunday"],
    ["traffic_index", "numeric", "Congestion level at dispatch, 0 (free flow) to 10 (gridlock); ~1.5% missing"],
    ["driver_experience_years", "numeric", "Driver experience (0.5–20 years); ~1.4% missing"],
    ["vehicle_load_pct", "numeric", "Vehicle load factor (20–100%)"],
    ["vehicle_type", "categorical", "bike, van, truck"],
    ["weather", "categorical", "clear, rain, fog, storm"],
    ["zone", "categorical", "Central, North, South, East, West"],
    ["priority", "categorical", "standard or express"],
    ["**delivery_time_min**", "**target**", "**Minutes from dispatch to delivery**"],
  ], [2600, 1300, 5126], { size: 18 }),
  H2("2.3 Dataset summary"),
  table(["Statistic", "Value"], [
    ["Rows / columns", `${ds.n_rows.toLocaleString()} / ${ds.n_columns}`],
    ["Period covered (simulated)", `${ds.date_range[0].slice(0, 10)} to ${ds.date_range[1].slice(0, 10)}`],
    ["Target mean / median / std", `${n1(ds.target_mean)} / ${n1(ds.target_median)} / ${n1(ds.target_std)} min`],
    ["Target min / max", `${n1(ds.target_min)} / ${n1(ds.target_max)} min`],
    ["Vehicle mix", `van ${n1(100 * ds.vehicle_share.van)}%, bike ${n1(100 * ds.vehicle_share.bike)}%, truck ${n1(100 * ds.vehicle_share.truck)}%`],
    ["Weather mix", `clear ${n1(100 * ds.weather_share.clear)}%, rain ${n1(100 * ds.weather_share.rain)}%, fog ${n1(100 * ds.weather_share.fog)}%, storm ${n1(100 * ds.weather_share.storm)}%`],
    ["Missing values", `traffic_index: ${ds.missing_traffic_index}; driver_experience_years: ${ds.missing_driver_experience}`],
    ["Late rate vs rule-of-thumb ETA", `${n1(100 * ds.late_rate_vs_promised_eta)}%`],
  ], [3600, 5426]),
  spacer(80),
  P("Missing values were injected on purpose (about 1.5% in two columns) to mimic sensor or data-entry gaps and to demonstrate a realistic preprocessing step. They are handled by median imputation inside the modelling pipeline."),
  ...figure("eda_overview.png", 600, "Figure 1. Exploratory view of the simulated dataset: target distribution, daily pattern by vehicle, and time versus distance."),
  P("Three patterns stand out. Delivery time is right-skewed, with a long tail of slow deliveries. Mean time rises in the morning and evening peaks. Time grows with distance, and the scatter around the trend widens with congestion, which signals an interaction between distance and traffic that a linear model cannot fully capture."),
];

// ---------- 3. Methodology ----------
const s3 = [
  H1("3. Methodology and Model Selection"),
  H2("3.1 Pipeline overview"),
  N("**Split**: 80% training (4,800 deliveries) and 20% hold-out test (1,200), random with a fixed seed."),
  N("**Preprocess inside a Pipeline**: median imputation and scaling for numeric inputs; most-frequent imputation and one-hot encoding for categorical inputs. Fitting these steps inside each cross-validation fold prevents information leaking from validation data."),
  N("**Feature engineering**: `is_peak_hour` (08–10 and 17–19), `is_weekend`, and the interaction `dist_x_traffic`."),
  N("**Compare six models** with 5-fold cross-validation on the training set."),
  N("**Tune** four models by randomized search; **select** the best on cross-validation RMSE."),
  N("**Evaluate once** on the hold-out test set; run robustness checks (time-ordered split, error by segment)."),
  ...code("Code 1. Preprocessing and model pipeline (src/train_models.py)", [
    "def build_pipeline(estimator):",
    "    numeric = Pipeline([('impute', SimpleImputer(strategy='median')),",
    "                        ('scale', StandardScaler())])",
    "    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),",
    "                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])",
    "    prep = ColumnTransformer([('num', numeric, NUMERIC_ALL),",
    "                              ('cat', categorical, CATEGORICAL)])",
    "    return Pipeline([('features', FunctionTransformer(add_features)),",
    "                     ('prep', prep), ('model', estimator)])",
  ]),
  H2("3.2 Model candidates and rationale"),
  P("The candidates were chosen to span a range of complexity, so the added value of each step up in flexibility can be measured rather than assumed."),
  table(["Model", "Why it was included", "Expected strength / weakness"], [
    ["Mean baseline", "Floor: predicts the training mean for everyone. Any useful model must beat it.", "R² ≈ 0 by construction"],
    ["Linear regression", "Transparent baseline; coefficients are easy to explain to operations staff.", "Fast, interpretable; cannot model non-linear congestion or interactions without manual features"],
    ["Ridge regression", "Linear model with L2 penalty; tests whether regularisation helps with correlated inputs (hour, traffic, zone).", "Stable with correlated features; still linear"],
    ["Decision tree", "Captures non-linearity and interactions with human-readable rules.", "Easy to read but a single tree has high variance and overfits"],
    ["Random forest", "Averages many de-correlated trees to reduce variance (Breiman, 2001).", "Robust, little tuning; can be less sharp than boosting and larger to store"],
    ["Gradient boosting", "Builds trees sequentially, each correcting the previous errors (Friedman, 2001); a strong default for tabular data.", "Usually most accurate on tabular data; needs tuning and is harder to interpret"],
  ], [1800, 3900, 3326], { size: 18 }),
];

// ---------- 4. Evaluation ----------
const s4 = [
  H1("4. Evaluation Metrics and Validation Strategy"),
  H2("4.1 Why MAE, RMSE and R²"),
  P("No single metric tells the whole story, so three complementary metrics are reported. With y the actual time, ŷ the prediction and n the number of deliveries:"),
  table(["Metric", "Formula", "Why it is used here"], [
    ["MAE", "mean(|y − ŷ|)", "Reported in minutes, so it reads directly as “the ETA is typically off by X minutes”. It treats every minute of error equally and is robust to a few extreme outliers. It is the headline metric for ETA quality."],
    ["RMSE", "√mean((y − ŷ)²)", "Squaring penalises large misses more heavily. A 30-minute miss hurts the customer far more than three 10-minute ones, so RMSE reflects the cost of bad surprises. It was the metric used for model selection and tuning. A large RMSE/MAE ratio signals heavy-tailed errors."],
    ["R²", "1 − Σ(y − ŷ)² / Σ(y − ȳ)²", "Unit-free share of variation in delivery time that the model explains (1 = perfect, 0 = no better than the mean). It lets us compare against the mean baseline and judge the model without needing to know the scale of minutes."],
  ], [1000, 2300, 5726], { size: 18 }),
  spacer(80),
  P("Using all three guards against misleading conclusions: two models can have a similar MAE yet very different RMSE if one makes occasional large errors, and R² shows how much better than a naive guess both are."),
  H2("4.2 How the models were compared"),
  B("**Same data, same folds**: every model was scored on identical 5-fold splits (shuffled, seed 42) of the training set, so differences come from the model and not from luck in the split."),
  B("**Mean ± spread**: cross-validated RMSE is reported with its standard deviation across folds, to see whether a gap between models is larger than fold-to-fold noise."),
  B("**Fair tuning**: the four tunable models received the same budget (randomized search, up to 8 candidates, 3-fold inner CV). Linear regression has no hyper-parameters; the baseline needs none."),
  B("**Selection rule**: the winner was picked on cross-validation RMSE **only**. The hold-out test set was untouched until after selection, so the final test numbers are an unbiased estimate."),
  B("**Sanity checks**: comparison with the mean baseline, a time-ordered split, and error analysis by weather, vehicle and peak/off-peak."),
  ...code("Code 2. Cross-validation and hyper-parameter search (src/train_models.py)", [
    "cv        = KFold(n_splits=5, shuffle=True, random_state=SEED)  # reporting/selection",
    "search_cv = KFold(n_splits=3, shuffle=True, random_state=SEED)  # inside the search",
    "scoring   = {'MAE': 'neg_mean_absolute_error',",
    "             'RMSE': 'neg_root_mean_squared_error', 'R2': 'r2'}",
    "",
    "# Stage 1: default models, 5-fold CV",
    "s = cross_validate(build_pipeline(est), X_train, y_train, cv=cv, scoring=scoring)",
    "",
    "# Stage 2: randomized hyper-parameter search (Bergstra & Bengio, 2012)",
    "rs = RandomizedSearchCV(build_pipeline(est), grid, n_iter=8, cv=search_cv,",
    "                        scoring='neg_root_mean_squared_error', random_state=SEED)",
    "rs.fit(X_train, y_train)",
    "",
    "# Stage 3: choose by CV RMSE, then ONE evaluation on the hold-out test set",
    "pred = best_pipeline.predict(X_test)",
    "mae, rmse, r2 = mean_absolute_error(y_test, pred), \\",
    "                np.sqrt(mean_squared_error(y_test, pred)), r2_score(y_test, pred)",
  ]),
];

// ---------- 5. Results ----------
const cvRows = cmp.map((r) => [r.Model, n2(r.CV_MAE), n2(r.CV_RMSE) + " ± " + n2(r.CV_RMSE_std), n3(r.CV_R2),
  r.Tuned_CV_RMSE ? n2(r.Tuned_CV_RMSE) : "n/a (no tuning)"]);
const testRows = cmp.map((r) => [r.Model === ms.best_model ? `**${r.Model}**` : r.Model,
  n2(r.Test_MAE), n2(r.Test_RMSE), n3(r.Test_R2)]);
const cr = AlignmentType.RIGHT;
const s5 = [
  H1("5. Results and Model Comparison"),
  H2("5.1 Cross-validation on the training set"),
  tcap("Table 1. 5-fold cross-validation results (training data, default settings; last column after tuning)"),
  table(["Model", "CV MAE (min)", "CV RMSE (min) ± std", "CV R²", "Tuned CV RMSE"], cvRows, [2300, 1400, 2226, 1100, 2000],
    { size: 18, align: [AlignmentType.LEFT, cr, cr, cr, cr] }),
  spacer(100),
  H2("5.2 Hold-out test performance"),
  tcap("Table 2. Final test-set performance (1,200 unseen deliveries; models tuned where applicable)"),
  table(["Model", "Test MAE (min)", "Test RMSE (min)", "Test R²"], testRows, [3300, 1900, 1900, 1926],
    { size: 19, align: [AlignmentType.LEFT, cr, cr, cr] }),
  spacer(80),
  ...figure("model_comparison.png", 430, "Figure 2. Cross-validated versus hold-out RMSE for each model. Close agreement between the two bars indicates no overfitting to the training data."),
  H3("What the comparison shows"),
  B(`**Everything beats the baseline.** The mean predictor reaches RMSE ${n2(base.Test_RMSE)} min with R² ≈ 0, so the features carry real signal.`),
  B(`**Linear regression is a strong baseline** (R² ${n3(lin.Test_R2)}), which is expected because delivery time is largely driven by distance. Ridge gave the same result (${n3(rd.Test_R2)}), i.e. regularisation was not needed with only 12 inputs.`),
  B(`**A single decision tree is worse than linear regression** (RMSE ${n2(dt.Test_RMSE)} vs ${n2(lin.Test_RMSE)}), a classic sign of high variance: one deep tree memorises noise.`),
  B(`**Ensembles win.** Random forest improves to RMSE ${n2(rf.Test_RMSE)}; gradient boosting is best at ${n2(gb.Test_RMSE)}. Relative to linear regression, boosting lowers test RMSE by ${n1(rmseGainVsLinear)}% and MAE by ${n1(maeGainVsLinear)}%. The gain comes from modelling the non-linear congestion effect and its interaction with distance and vehicle type.`),
  B(`**No overfitting signal.** Cross-validated and hold-out errors are close for every model (e.g. boosting: CV RMSE ${n2(gb.Tuned_CV_RMSE)} vs test ${n2(gb.Test_RMSE)}), and the fold-to-fold standard deviation is small (about ${n2(gb.CV_RMSE_std)} min).`),
  H2("5.3 Selected model"),
  P(`**${ms.best_model}** was selected on cross-validation RMSE. Tuned hyper-parameters: ${Object.entries(ms.best_params).map(([k, v]) => `\`${k}=${v}\``).join(", ")}. On the hold-out test set it achieves:`),
  table(["Metric", "Value", "Interpretation"], [
    ["MAE", `${n2(best.MAE)} min`, `Typical error of about ${Math.round(best.MAE)} minutes on a mean delivery time of ${n1(ds.target_mean)} min`],
    ["RMSE", `${n2(best.RMSE)} min`, `RMSE/MAE = ${n2(best.RMSE / best.MAE)}: errors are mildly heavy-tailed, driven by unpredictable incidents`],
    ["R²", n3(best.R2), `The model explains about ${Math.round(100 * best.R2)}% of the variation in delivery time`],
    ["Rule-of-thumb ETA MAE", `${n2(Bk.eta_mae_rule_of_thumb_min)} min`, `The model cuts typical ETA error by ${n1(100 * (1 - Bk.eta_mae_model_min / Bk.eta_mae_rule_of_thumb_min))}%`],
  ], [2300, 1600, 5126]),
  spacer(80),
  ...figure("predicted_vs_actual.png", 560, "Figure 3. Predicted versus actual delivery time (left) and residual distribution (right) on the hold-out test set."),
  P("Points hug the diagonal across the whole range, and residuals are centred on zero with a slight right tail. That tail corresponds to the simulated random incidents (3% of deliveries), which no model can predict from dispatch-time information."),
  H2("5.4 Context: how close is the model to the achievable ceiling?"),
  P(`Because the data are simulated, we can compute the error of a **perfect predictor** that knows the exact noise-free expected time. On the same test set it still has MAE ${n2(floor.MAE)} min, RMSE ${n2(floor.RMSE)} min and R² ${n3(floor.R2)}, because of random noise and incidents. The fitted model (RMSE ${n2(best.RMSE)}) is therefore within ${n2(best.RMSE - floor.RMSE)} min RMSE of the best any model could do here. This reference is not available for real data, where the ceiling is unknown and usually lower.`),
  H2("5.5 Robustness checks"),
  B(`**Time-ordered split.** Training on the first 80% of the year and testing on the final 20% gave MAE ${n2(chrono.MAE)}, RMSE ${n2(chrono.RMSE)}, R² ${n3(chrono.R2)}, similar to the random split. (This is expected since the simulator has no drift; on real data, a time-ordered split is the more demanding test and is recommended.)`),
  B(`**Error by segment** (mean absolute error, minutes): clear ${n2(ms.error_by_segment.weather.clear)}, rain ${n2(ms.error_by_segment.weather.rain)}, storm ${n2(ms.error_by_segment.weather.storm)}; bike ${n2(ms.error_by_segment.vehicle.bike)}, van ${n2(ms.error_by_segment.vehicle.van)}, truck ${n2(ms.error_by_segment.vehicle.truck)}; peak ${n2(ms.error_by_segment.peak_vs_offpeak.peak)} vs off-peak ${n2(ms.error_by_segment.peak_vs_offpeak.off_peak)}. The model is least accurate in storms, for trucks (only 1.4% of the data, so few examples) and at peak times, exactly where operational risk is highest, so alerts should carry a safety buffer.`),
  B(`**Safety buffer.** The 90th percentile of out-of-fold residuals on the training data is ${n2(ms.residual_buffer_q90_min)} min. Adding it to a prediction covered ${n1(100 * ms.buffer_empirical_coverage_test)}% of hold-out deliveries, close to the intended 90%, which confirms the buffer is well calibrated. It is used in the alert module.`),
  H2("5.6 Which inputs matter?"),
  tcap("Table 3. Permutation importance on the hold-out set (increase in RMSE when a feature is shuffled)"),
  table(["Rank", "Feature", "RMSE increase (min)"], fi.slice(0, 8).map((r, i) => [String(i + 1), r.feature, n2(r.rmse_increase)]),
    [900, 4200, 3926], { size: 19, align: [AlignmentType.CENTER, AlignmentType.LEFT, cr] }),
  spacer(60),
  ...figure("feature_importance.png", 440, "Figure 4. Permutation feature importance (top 10)."),
  P("Distance dominates, followed by vehicle type, number of stops, weather and the traffic index. Hour of day, day of week and zone show near-zero permutation importance. This does **not** mean they are irrelevant: in the data, their effect operates through the traffic index (the congestion profile depends on hour, zone and weekday), so once traffic is known they add little extra information. Permutation importance understates the value of correlated features; the practical lesson is that a reliable real-time **traffic index is the key input** to collect."),
];

// ---------- 6. Optimization ----------
const hourRows = [7, 9, 12, 15, 18, 20].map((h) => [`${String(h).padStart(2, "0")}:00`, n1(H(h).model_pred_min), n1(H(h).true_expected_min)]);
const lv = Bk.alert_levels;
const s6 = [
  H1("6. Optimization Strategies"),
  P("Prediction alone does not improve operations; decisions do. Four modules turn the model's output into actions, ordered by when they occur in the daily workflow:"),
  table(["Stage", "Module", "Model output used", "Decision"], [
    ["Planning (day before / morning)", "6.1 Dispatch planning", "Predicted time by hour", "When to release flexible deliveries"],
    ["Pre-dispatch", "6.2 Traffic-risk alerts", "Predicted time + safety buffer vs promised ETA", "Warn, re-quote or re-assign at-risk deliveries"],
    ["Dispatch", "6.3 Vehicle allocation", "Predicted time for each parcel × vehicle", "Which parcel goes on which vehicle"],
    ["On the road", "6.4 Route planning", "Learned minutes-per-km versus congestion", "Order of stops on a run"],
  ], [2100, 2000, 2700, 2226], { size: 18 }),
  ...callout("How the optimisation results are evaluated",
    "Decisions are made using the **model's predictions**, but outcomes are scored using the simulator's noise-free ground truth (`expected_time`). This avoids grading the model with its own opinion. The reported gains are **simulated gains** under assumptions stated in each subsection; they show the method works on this synthetic system and are not savings measured at any real company.",
    BLUE, LIGHT),

  H2("6.1 Dispatch planning: shift flexible deliveries away from congestion"),
  P(`**Idea.** Express parcels must leave immediately, but many standard deliveries are booked into time slots. If a standard delivery planned for a congested hour can leave up to two hours later at a lower predicted time, vehicle time is saved. The model-predicted time for a reference delivery (8 km, 5 stops, van, clear weather) is lowest at ${String(A.best_hour).padStart(2, "0")}:00 (${n1(H(A.best_hour).model_pred_min)} min) and highest at ${String(A.worst_hour).padStart(2, "0")}:00 (${n1(H(A.worst_hour).model_pred_min)} min): a spread of ${n1(peakGap)} minutes for identical work.`),
  tcap("Table 4. Predicted delivery time of the reference delivery by dispatch hour"),
  table(["Dispatch hour", "Model prediction (min)", "Simulator truth (min)"], hourRows, [2600, 3213, 3213], { size: 19, align: [AlignmentType.LEFT, cr, cr] }),
  spacer(60),
  ...figure("dispatch_hourly.png", 480, "Figure 5. Predicted time by dispatch hour (shaded bands = morning and evening peaks)."),
  P(`**Method.** For every standard-priority delivery dispatched in a peak hour (08–10, 17–19), the model predicts the time for departures 0, +1 and +2 hours later (traffic index shifted by the hourly profile). A delivery is moved only if the predicted saving is at least 2 minutes (to avoid chasing noise).`),
  ...code("Code 3. Choosing a dispatch shift (src/optimize.py, simplified)", [
    "for s in range(MAX_SHIFT + 1):                      # candidate delays 0, 1, 2 h",
    "    pred_s.append(model.predict(shifted_frame(s)))   # decisions use the MODEL",
    "    true_s.append(expected_time(shifted_clean(s)))   # outcomes use SIMULATOR truth",
    "gain_pred = pred_s[0] - pred_s.min(axis=0)",
    "chosen = np.where(gain_pred >= MIN_GAIN, pred_s.argmin(axis=0), 0)",
    "realised_saving = true_s[0] - true_s[chosen, np.arange(n)]",
  ]),
  table(["Result (hold-out deliveries)", "Value"], [
    ["Standard deliveries dispatched at peak", `${A.peak_standard_deliveries} (${n1(A.share_of_all_test_deliveries_pct)}% of all test deliveries)`],
    ["Shifted by the model", `${A.deliveries_shifted} (${n1(A.shifted_share_pct)}%), by ${n2(A.avg_shift_hours_when_shifted)} h on average`],
    ["True minutes saved per shifted delivery", `${n2(A.avg_true_minutes_saved_per_shifted_delivery)} min`],
    ["Total on-road vehicle minutes saved", `${n1(A.total_vehicle_minutes_saved)} min  (${n1(A.pct_reduction_in_on_road_minutes)}% of the group's on-road time)`],
    ["Share of the perfect-information saving captured", `${n1(A.pct_of_oracle_captured)}%`],
  ], [4700, 4326], { size: 19 }),
  spacer(60),
  P("**Caveats.** The saving is in **vehicle time and cost**, not customer waiting time: shifted deliveries arrive later in the day, so this only suits deliveries booked into flexible slots. Express parcels and tight slots are never moved. In practice the shift window should be set by the customer's slot, not fixed at two hours."),

  H2("6.2 Traffic-risk alerts"),
  P(`**Idea.** Before a vehicle leaves, compare the predicted delivery time with the ETA already promised to the customer and raise a colour-coded alert. This turns the forecast into an actionable early warning that gives dispatchers time to re-quote the customer, change vehicle or re-sequence the run.`),
  table(["Level", "Rule", "Action"], [
    ["Red", "Predicted time > promised ETA", "Likely late: re-quote the customer, upgrade vehicle or re-assign"],
    ["Amber", `Predicted time + ${n1(Bk.buffer_used_min)} min buffer > promised ETA`, "At risk: monitor, avoid adding stops to this run"],
    ["Green", "Otherwise", "Proceed normally"],
  ], [1100, 4100, 3826], { size: 19 }),
  spacer(60),
  ...code("Code 4. Alert logic (src/optimize.py)", [
    "level = np.where(pred > promised, 'Red',",
    "         np.where(pred + buffer_q90 > promised, 'Amber', 'Green'))",
    "late  = actual > promised                      # ground truth, used only for scoring",
  ]),
  tcap("Table 5. Alert performance on the hold-out test set"),
  table(["Alert level", "Deliveries", "Share", "Actual late rate"], [
    ["Red", String(lv.Red.deliveries), `${n1(lv.Red.share_pct)}%`, `${n1(lv.Red.actual_late_rate_pct)}%`],
    ["Amber", String(lv.Amber.deliveries), `${n1(lv.Amber.share_pct)}%`, `${n1(lv.Amber.actual_late_rate_pct)}%`],
    ["Green", String(lv.Green.deliveries), `${n1(lv.Green.share_pct)}%`, `${n1(lv.Green.actual_late_rate_pct)}%`],
    ["All deliveries", "1,200", "100%", `${n1(Bk.overall_late_rate_pct)}%`],
  ], [2600, 2000, 2000, 2426], { size: 19, align: [AlignmentType.LEFT, cr, cr, cr] }),
  spacer(60),
  ...figure("alert_levels.png", 380, "Figure 6. Actual late-delivery rate for each alert level."),
  B(`**The alerts separate risk well:** Red deliveries are late ${n1(lv.Red.actual_late_rate_pct)}% of the time versus ${n1(lv.Green.actual_late_rate_pct)}% for Green, a ${n1(lv.Red.actual_late_rate_pct / lv.Green.actual_late_rate_pct)}-fold difference.`),
  B(`**Coverage:** Red or Amber captured ${n1(Bk.red_or_amber.recall_pct)}% of all late deliveries (precision ${n1(Bk.red_or_amber.precision_pct)}%, flagging ${n1(Bk.red_or_amber.flagged_pct)}% of deliveries). Red alone has precision ${n1(Bk.red_only.precision_pct)}% and recall ${n1(Bk.red_only.recall_pct)}%. Operations can choose the trade-off: act on Red only to limit workload, or on Red + Amber to catch almost every late delivery.`),
  B(`**Versus a simple traffic rule:** a "traffic index ≥ 7" alert flags only ${n1(Bk.baseline_traffic_index_ge_7_rule.flagged_pct)}% of deliveries and catches just ${n1(Bk.baseline_traffic_index_ge_7_rule.recall_pct)}% of late ones. Lateness is mostly caused by combinations (distance, stops, weather, vehicle) that a single-threshold rule misses.`),
  B(`**Why deliveries were flagged:** adverse weather ${n1(Bk.alert_reasons_pct["Adverse weather"] || 0)}%, peak hour ${n1(Bk.alert_reasons_pct["Peak hour"] || 0)}%, long distance or many stops ${n1(Bk.alert_reasons_pct["Long distance / many stops"] || 0)}%, very high traffic ${n1(Bk.alert_reasons_pct["High traffic"] || 0)}%. Showing a reason tag next to each alert helps dispatchers trust and act on it.`),

  H2("6.3 Vehicle allocation"),
  P(`**Idea.** Choosing a vehicle changes both the delivery time and the cost. Bikes are cheap per minute but slow and weather-sensitive; vans are the workhorse; trucks are expensive but needed for heavy parcels. For each dispatch wave, the model predicts the time of every parcel on every feasible vehicle and an assignment algorithm picks the cheapest overall plan.`),
  P(`**Problem.** Minimise total cost = fixed trip cost + cost per minute × predicted time + late penalty × minutes beyond the promised ETA, subject to one parcel per vehicle and feasibility limits (bike: parcel ≤ 10 kg and ≤ 15 km; van: ≤ 150 kg; truck: any). This is a classic assignment problem solved exactly with the Hungarian algorithm (Kuhn, 1955; \`scipy.optimize.linear_sum_assignment\`). Cost figures are **assumptions**: bike ₹4/min + ₹20 per trip, van ₹7/min + ₹60, truck ₹12/min + ₹150, late penalty ₹10/min.`),
  ...code("Code 5. Cost-minimising assignment (src/optimize.py, simplified)", [
    "# C_pred[i, j] = fixed[v] + rate[v]*pred_time + late_penalty*max(0, pred_time - promised)",
    "#                (inf if parcel i is infeasible on vehicle j)",
    "rows, cols = linear_sum_assignment(C_pred)   # optimal one-to-one assignment",
    "",
    "# Baseline = today's rule: bike for small parcels, van for others, truck only if >150 kg,",
    "# first free vehicle in that preference order.",
  ]),
  P(`**Experiment.** ${C.batches} dispatch waves, each with ${C.parcels_per_batch} parcels drawn from the hold-out set and a fleet of ${C.fleet_size} vehicles (${C.fleet_mix.bike} bikes, ${C.fleet_mix.van} vans, ${C.fleet_mix.truck} trucks). Each plan is scored with the simulator's true times.`),
  table(["Per dispatch wave (avg of 200)", "Rule-based", "Model-guided", "Perfect-information bound"], [
    ["Total cost (₹)", inr(C.avg_cost_per_batch_inr.rule_based), inr(C.avg_cost_per_batch_inr.model_guided), inr(C.avg_cost_per_batch_inr.perfect_information_bound)],
    ["Vehicle minutes", n1(C.avg_vehicle_minutes_per_batch.rule_based), n1(C.avg_vehicle_minutes_per_batch.model_guided), n1(C.avg_vehicle_minutes_per_batch.perfect_information_bound)],
    ["Late parcels (of 20)", n2(C.avg_late_parcels_per_batch.rule_based), n2(C.avg_late_parcels_per_batch.model_guided), n2(C.avg_late_parcels_per_batch.perfect_information_bound)],
  ], [3100, 1800, 1900, 2226], { size: 19, align: [AlignmentType.LEFT, cr, cr, cr] }),
  spacer(60),
  ...figure("allocation_comparison.png", 520, "Figure 7. Average cost (left) and late parcels (right) per dispatch wave for three allocation strategies."),
  B(`Model-guided allocation reduces simulated cost by **${n1(C.cost_reduction_pct)}%** and late parcels by **${n1(C.late_reduction_pct)}%**, and is cheaper than the rule-based plan in ${n1(C.batches_where_model_guided_cheaper_pct)}% of waves.`),
  B(`It captures **${n1(C.pct_of_bound_captured)}%** of the saving that would be possible with perfect knowledge of delivery times, showing that the model is accurate enough to support this decision.`),
  B("The main behaviour change: the optimiser sends parcels that are at risk of lateness (long distance, bad weather, many stops) on faster vehicles and keeps bikes for short, low-risk parcels, instead of defaulting to the cheapest feasible vehicle."),
  P("**Caveats.** Counterfactual predictions (the same parcel on a different vehicle) are only trusted inside the feasible region, where the training data has examples of that vehicle type; trucks have few samples. One-parcel-per-vehicle is a simplification of real multi-drop runs, and cost parameters must be replaced with actual company rates."),

  H2("6.4 Route-planning improvements"),
  P(`**Idea.** Within a vehicle's run, the order of stops determines total driving time. Two improvements were tested on ${D.routes_simulated} simulated runs of ${D.stops_per_route} stops in a 20 × 20 km city with a congested core and two random hotspots (for example accidents or events):`),
  B("**A. Nearest neighbour on distance**: the common greedy baseline."),
  B("**B. Nearest neighbour + 2-opt on distance**: an improvement heuristic that repeatedly reverses route segments when that shortens the tour (Croes, 1958)."),
  B("**C. Nearest neighbour + 2-opt on traffic-aware travel times**: the same optimiser, but the cost of each leg comes from the model. The model's **learned minutes-per-km** at the congestion level of the two stops is used: pk(t) = [pred(15 km, t) − pred(5 km, t)] ÷ 10, which cancels fixed loading and waiting time. The curve is smoothed to be non-decreasing (isotonic regression) because congestion cannot speed traffic up."),
  P(`The model's learned curve is close to the truth at moderate congestion (${D.model_learned_min_per_km_at_t0_t5_t10.slice(0, 2).join(" and ")} min/km at index 0 and 5, versus ${D.true_min_per_km_at_t0_t5_t10.slice(0, 2).join(" and ")} true) but underestimates it at the extreme end (${D.model_learned_min_per_km_at_t0_t5_t10[2]} vs ${D.true_min_per_km_at_t0_t5_t10[2]} at index 10), because gridlock rows are rare in the training data.`),
  table(["Route strategy", "Average true route time (min)", "vs A"], [
    ["A. Nearest neighbour (distance)", n1(D.avg_true_route_minutes.A_nearest_neighbour_distance), "-"],
    ["B. 2-opt (distance)", n1(D.avg_true_route_minutes.B_2opt_distance), `-${n1(D.reduction_B_vs_A_pct)}%`],
    ["C. 2-opt (traffic-aware, model)", n1(D.avg_true_route_minutes.C_2opt_traffic_aware_model), `-${n1(D.reduction_C_vs_A_pct)}%`],
    ["Oracle: 2-opt with perfect traffic knowledge", n1(D.avg_true_route_minutes.Oracle_2opt_perfect_traffic_knowledge), `-${n1(D.reduction_oracle_vs_A_pct)}%`],
  ], [4300, 2800, 1926], { size: 19, align: [AlignmentType.LEFT, cr, cr] }),
  spacer(60),
  ...figure("route_example.png", 560, "Figure 8. A median-saving example: distance-only route (left) versus the improved route (right) over the congestion map."),
  H3("An honest finding"),
  P(`Replacing the greedy route by a 2-opt optimised route cut average route time by **${n1(D.reduction_B_vs_A_pct)}%** (${n1(D.minutes_saved_per_route_C_vs_A)} minutes per run), and the optimised route was faster in ${n1(D.routes_where_C_faster_than_A_pct)}% of runs. However, **making the optimiser traffic-aware added only ${n2(D.extra_reduction_C_vs_B_pct)}%** on top of 2-opt, and even perfect traffic knowledge would add only ${n2(D.extra_reduction_oracle_vs_B_pct)}%. The reason is structural: when every stop must be visited and the vehicle returns to the depot, congestion mostly scales all legs similarly and cannot be avoided, so the geometry of the tour dominates. The example in Figure 8 shows this: the gain comes mainly from removing crossing legs, not from dodging the hotspots.`),
  P("**Practical conclusion.** Invest first in a proper route optimiser (2-opt or a solver such as OR-Tools). Traffic-aware travel times become valuable when the problem offers real choices that congestion can influence, such as customer time windows, choosing which stops to put on which run, alternative road paths, or the departure time. That is why the traffic signal is used in modules 6.1 to 6.3, where it gave larger effects."),
];

// ---------- 7. Recommendations ----------
const s7 = [
  H1("7. Recommendations and Implementation Roadmap"),
  P("The recommendations below are ranked by the strength of the evidence in this study and by ease of adoption. Impact figures come from the simulation and must be re-measured on real data before being used for business cases."),
  table(["#", "Recommendation", "Evidence (simulated)", "KPI to track"], [
    ["1", "Replace the rule-of-thumb ETA with the model's prediction for customer quotes", `ETA error ${n2(Bk.eta_mae_rule_of_thumb_min)} → ${n2(Bk.eta_mae_model_min)} min MAE`, "ETA MAE, on-time rate"],
    ["2", "Switch on Red / Amber pre-dispatch alerts with reason tags; start with Red only, add Amber once the team can handle the volume", `Red late rate ${n1(lv.Red.actual_late_rate_pct)}% vs Green ${n1(lv.Green.actual_late_rate_pct)}%; Red+Amber catches ${n1(Bk.red_or_amber.recall_pct)}% of late deliveries`, "Alert precision / recall, late rate, re-quote rate"],
    ["3", "Use model-guided vehicle allocation for each dispatch wave, with a human override", `Cost -${n1(C.cost_reduction_pct)}%, late parcels -${n1(C.late_reduction_pct)}%`, "Cost per delivery, late parcels per wave"],
    ["4", "Offer flexible delivery slots and release flexible deliveries outside the evening and morning peaks", `On-road minutes of eligible deliveries -${n1(A.pct_reduction_in_on_road_minutes)}%`, "Vehicle-hours per delivery, slot uptake"],
    ["5", "Adopt a route optimiser (2-opt or solver) for multi-stop runs before investing in traffic-aware costs", `Route time -${n1(D.reduction_B_vs_A_pct)}%`, "Km and minutes per run"],
    ["6", "Invest in a reliable live traffic index and weather feed", "Traffic and weather are among the top inputs; the traffic index also drives the alerts", "Data completeness, feed latency"],
  ], [450, 3700, 3000, 1876], { size: 18 }),
  H2("7.1 Implementation roadmap"),
  table(["Phase", "Actions"], [
    ["0-30 days: foundation", "Collect real dispatch, GPS and delivery timestamps; define the target precisely; build the data pipeline; replicate this study on real data using the same scripts."],
    ["30-90 days: pilot", "Run the model in shadow mode (predict but do not act) on one depot; compare predictions with actual times; calibrate the alert buffer; test Red alerts with dispatchers."],
    ["90+ days: scale", "Enable allocation support and flexible-slot release; add route optimisation; extend to more depots."],
    ["Ongoing: monitoring", "Track weekly MAE and bias by weather, vehicle and hour; retrain monthly or when MAE degrades by more than a set threshold; log every override to learn from dispatcher judgement."],
  ], [2400, 6626], { size: 18 }),
];

// ---------- 8. Limitations ----------
const s8 = [
  H1("8. Limitations and Threats to Validity"),
  B("**Simulated data.** The most important limitation. Performance on real data will almost certainly be lower and will need re-validation. The model has learned the simulator's rules, not the behaviour of a real city."),
  B("**Optimistic evaluation.** Optimisation gains are scored against the same simulator that generated the training data. They demonstrate the method, not a measured business saving."),
  B("**Assumed costs.** All cost rates, fixed trip costs and late penalties in Section 6.3 are illustrative assumptions."),
  B("**No drift.** The simulated process is stationary; real logistics data changes with seasons, road works and demand, so monitoring and retraining are essential."),
  B("**Simplified decisions.** One parcel per vehicle in the allocation experiment, a fixed 20 × 20 km city and simple hotspot congestion in the routing experiment, and a fixed flexibility window in dispatch planning."),
  B("**Sparse segments.** Trucks and storms have few observations and larger errors."),
  B("**Correlated features.** Permutation importance understates the role of hour, zone and weekday, whose effects pass through the traffic index."),
  B("**Fairness and workload.** In a real deployment, alerts and allocation changes affect drivers' workloads; they should be introduced with human oversight and clear override rules."),
];

// ---------- 9. Reproducibility ----------
const s9 = [
  H1("9. Reproducibility and Code Verification"),
  P("Every random process uses a fixed seed (42), and all preprocessing happens inside a scikit-learn Pipeline, so each run produces the same dataset, models, metrics and figures. Two scripts are provided:"),
  ...code("Commands", [
    "pip install -r requirements.txt",
    "python src/run_pipeline.py            # data -> models -> optimisation (about 2-5 minutes)",
    "python verify_reproducibility.py      # re-runs everything in a temp folder and compares",
  ]),
  P(`\`verify_reproducibility.py\` re-runs the **entire** pipeline in a temporary folder and compares every headline number in the three result files with the committed results (tolerance 0.001), and requires the regenerated dataset CSV to be byte-identical.`),
  ...(verifyStatus ? callout("Verification result at the time of writing", verifyStatus, "166534", "ECFDF3") : []),
  P("The report itself is generated from the result files by `report/build_report.js`, so the numbers in this document cannot drift from the code output."),
];

// ---------- 10. Conclusion ----------
const s10 = [
  H1("10. Conclusion"),
  P(`This project defined a clear logistics forecasting problem, built a simulated but realistic dataset, and compared six regression techniques with consistent cross-validation, tuning and hold-out testing. Gradient boosting was the best model (test MAE ${n2(best.MAE)} min, RMSE ${n2(best.RMSE)} min, R² ${n3(best.R2)}), outperforming linear regression by about ${n1(rmseGainVsLinear)}% in RMSE and clearly beating the rule-of-thumb ETA.`),
  P("The model's value was shown through decisions rather than accuracy alone. Pre-dispatch alerts separate high-risk from low-risk deliveries, model-guided vehicle allocation reduces simulated cost and lateness, dispatch timing can trim on-road time for flexible deliveries, and route optimisation shortens runs, although traffic-aware routing itself added little. Reporting that negative result, together with the explicit limits of simulated data, is part of keeping the conclusions honest."),
  P("The natural next step is to replicate the study on real operational data, where the same scripts, evaluation protocol and monitoring plan can be reused."),
];

const refs = [
  H1("References"),
  ...[
    "Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13, 281-305.",
    "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
    "Croes, G. A. (1958). A method for solving traveling-salesman problems. Operations Research, 6(6), 791-812.",
    "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189-1232.",
    "Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.",
    "Kuhn, H. W. (1955). The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2(1-2), 83-97.",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
  ].map((t) => new Paragraph({ children: [new TextRun({ text: t, size: 21 })], spacing: { after: 100 }, indent: { left: 360, hanging: 360 } })),
];

const appA = [
  H1("Appendix A: Repository Structure"),
  ...code(null, [
    "logistics-delivery-time-prediction/",
    "|-- README.md                      project overview and how to run",
    "|-- requirements.txt               Python dependencies",
    "|-- verify_reproducibility.py      re-runs pipeline and checks results match",
    "|-- data/",
    "|   `-- delivery_data.csv          SIMULATED dataset (6,000 deliveries)",
    "|-- src/",
    "|   |-- common.py                  paths, seed, features, split",
    "|   |-- simulator.py               data-generating process (digital twin)",
    "|   |-- generate_data.py           step 1: create dataset + EDA figure",
    "|   |-- train_models.py            step 2: CV, tuning, evaluation",
    "|   |-- optimize.py                step 3: dispatch, alerts, allocation, routes",
    "|   `-- run_pipeline.py            runs steps 1-3",
    "|-- results/                       metrics (JSON/CSV), model, figures/",
    "`-- report/",
    "    |-- build_report.js            generates this report from results/",
    "    `-- Logistics_Predictive_Modeling_Report.docx",
  ]),
];

const bp = ms.all_best_params;
const appB = [
  H1("Appendix B: Tuned Hyper-parameters"),
  P("Selected by randomized search (up to 8 candidates, 3-fold inner cross-validation, RMSE criterion) on the training set."),
  table(["Model", "Selected hyper-parameters"], Object.entries(bp).map(([k, v]) => [k, Object.entries(v).map(([a, b]) => `${a} = ${b}`).join(";  ")]),
    [2400, 6626], { size: 18 }),
];

// ---------- assemble ----------
const doc = new Document({
  creator: "Shruti Rani", title: "Predictive Modeling and Optimization in Logistics Systems",
  description: "Week 4 task report (simulated dataset)",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 34, bold: true, color: NAVY, font: "Calibri" },
        paragraph: { spacing: { before: 120, after: 200 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: BLUE, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 27, bold: true, color: "1D4ED8", font: "Calibri" },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, color: NAVY, font: "Calibri" },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1000, hanging: 270 } } } }] },
      { reference: "numbers", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 320 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1300, left: 1440 } }, titlePage: true },
    headers: {
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BORDER, space: 4 } },
        children: [new TextRun({ text: "Predictive Modeling and Optimization in Logistics Systems  |  Simulated-data study", size: 17, color: "6B7280" })] })] }),
      first: new Header({ children: [new Paragraph({ children: [] })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
        new TextRun({ text: "All data is simulated. Not real company data.   |   Page ", size: 17, color: "6B7280" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 17, color: "6B7280" }),
        new TextRun({ text: " of ", size: 17, color: "6B7280" }),
        new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 17, color: "6B7280" })] })] }),
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
    },
    children: [...cover, ...contents, ...abstract, ...s1, ...s2, ...s3, ...s4, ...s5, ...s6, ...s7, ...s8, ...s9, ...s10, ...refs, ...appA, ...appB],
  }],
});

const out = path.join(__dirname, "Logistics_Predictive_Modeling_Report.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("Report written:", out, (buf.length / 1024).toFixed(0) + " KB"); });
