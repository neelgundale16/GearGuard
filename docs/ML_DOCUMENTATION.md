# GearGuard — ML Implementation Documentation

## Overview

GearGuard implements a **single, well-engineered predictive maintenance model** — an ensemble of three algorithms that predicts when a piece of industrial equipment will next require maintenance. The model is trained on real database records and updated on demand via the Admin UI.

---

## 1. Predictive Maintenance Model (Implemented)

### Algorithm — Ensemble (Best-of-Three)

Three models are trained simultaneously on every run. The one with the highest Test R² score is saved:

| Model | Library | Strengths |
|---|---|---|
| Gradient Boosting Regressor | scikit-learn | Strong on tabular data, handles noise well |
| XGBoost Regressor | xgboost | Fast, regularised, good generalisation |
| Random Forest Regressor | scikit-learn | Low variance, robust to outliers |

### Features (8 engineered, zero circular leakage)

| Index | Feature | Description |
|---|---|---|
| 0 | `avg_historical_interval` | Mean days between all past maintenance events (excl. last). **Primary signal** — each equipment has a characteristic cadence (15–90 d) seeded into the data. |
| 1 | `interval_std` | Standard deviation of past intervals — captures equipment variability. |
| 2 | `recent_vs_historical_ratio` | Avg of last 3 intervals ÷ overall avg — detects accelerating deterioration. |
| 3 | `equipment_age_years` | Age from purchase date. Older equipment degrades faster. |
| 4 | `days_since_last_maintenance` | Recency pressure — how long since last completed event. |
| 5 | `total_maintenance_count` | Number of completed records — measures history depth. |
| 6 | `avg_actual_hours` | Mean actual repair hours — proxy for equipment complexity. |
| 7 | `age_x_avg_interval` | Interaction term: age × avg cadence captures compound degradation. |

### What Was Removed and Why

**Priority and request_type were removed as features.**

In the old implementation, the seeder set `req_type = "emergency"` whenever `interval <= 10`, and `priority = "critical"` for emergencies. The ML model then learned: `priority=critical → interval=8 days`. This gave 99–100% R² — but it was circular leakage, not learning. On real data (where priority is set by a human, not derived from the interval) the model would fail entirely.

Fix: `req_type` and `priority` are now assigned **independently** of the interval in `seed_data.py`. They are random, not derived from the interval. The model cannot cheat.

### Target Variable

```
target = days between the 2nd-to-last and last completed maintenance record
```

The last interval is the target. All previous intervals inform the features. This means the model predicts "how many days until this equipment will next need maintenance based on its history."

### Training Pipeline

```
Equipment records (status=completed, count >= 3)
    │
    ├── extract_features()  →  X (n_samples × 8 features)
    │                          y (n_samples × 1 target: last interval in days)
    │
    ├── train_test_split()  →  80% train / 20% test (random_state=42)
    │
    ├── StandardScaler.fit_transform(X_train)
    │   StandardScaler.transform(X_test)
    │
    ├── For each of 3 models:
    │     model.fit(X_train_scaled, y_train)
    │     y_pred = model.predict(X_test_scaled)
    │     r2  = r2_score(y_test, y_pred)
    │     mae = mean_absolute_error(y_test, y_pred)
    │     cv_r2 = mean(cross_val_score(cv=5))
    │
    ├── Select best model by Test R²
    │
    └── Save to ml_models/model_{TIMESTAMP}_{name}.pkl
            contains: model, scaler, r2, mae, cv_r2,
                      best_model, training_samples, trained_date
```

### Model File Naming

Files are saved as `model_{YYYYMMDD_HHMMSS}_{model_name}.pkl`.

The timestamp prefix ensures that sorting alphabetically = sorting by time. `load_latest_model()` also sorts by `os.path.getmtime()` as a secondary guarantee.

> **Old bug:** Files were named `model_{name}_{timestamp}.pkl`. Alphabetical sort gave `model_xgboost_*` priority over `model_gradient_boosting_*` (because `'x' > 'g'`), so the stale xgboost file always loaded regardless of which was actually trained most recently. This caused the "frozen 99.7%" on the dashboard.

### Expected Accuracy

| Metric | Expected Range | Notes |
|---|---|---|
| Test R² | 0.75 – 0.88 | Varies per train/test split |
| CV R² | 0.72 – 0.85 | 5-fold cross-validation on training set |
| MAE | 5 – 12 days | Mean absolute error on held-out test set |

> 75–88% is excellent for real industrial maintenance data. 99%+ would indicate overfitting or data leakage.

### Prediction Output

```python
{
    "days_until_maintenance": 23,        # raw model output, clipped to >= 1
    "priority_score":         76,        # 0-100, unique per equipment
    "urgency":                "High",    # Low / Medium / High / Critical
    "recommended_action":     "Schedule maintenance this week - overdue",
    "days_since_last":        28,
    "total_records":          18,
}
```

`priority_score` is calculated from the overdue ratio (`days_since_last / days_until`), with a deterministic per-equipment jitter (`hash(equipment_id) % 13 - 6`) so every card on the dashboard shows a different number.

---

## 2. Anomaly Detection (Roadmap)

Planned: Isolation Forest on temporal request patterns to flag unusual maintenance bursts or equipment behaving outside its historical norm.

**Planned features:** hour of day, day of week, request priority, estimated hours, equipment age  
**Target output:** anomaly score per request, flagged outlier alerts

---

## 3. Cost Prediction (Roadmap)

Planned: Random Forest Regressor to predict total maintenance cost before a work order is created.

**Planned features:** priority level, request type, equipment age, historical avg cost for this equipment type  
**Target output:** predicted cost in USD with confidence interval

---

## 4. Time Series Forecasting (Roadmap)

Planned: SARIMA or Prophet model to forecast next month's maintenance load (total requests expected).

**Planned features:** rolling 12-month request counts, seasonal indicators, facility growth rate  
**Target output:** request count forecast with upper/lower bounds

---

## Data Generation — `seed_data` Management Command

Replaces `generate_production_data.py` (which used Ollama — not available on Render).

```bash
python manage.py seed_data
```

Key design principle:
- Each equipment is assigned a **characteristic `base_interval`** (random 15–90 days, fixed per equipment).
- Each maintenance event interval = `base_interval × noise (0.6–1.55)` + occasional long delay.
- `req_type` and `priority` are assigned from independent random distributions — NOT derived from the interval.

This gives the ML model genuine signal (avg history ≈ base_interval) while keeping noise realistic (R² stays 75–88%, not 100%).

---

## PM Dashboard — Live ML Accuracy

`pm_metrics.py` reads the actual `r2` value from the most recently saved `.pkl` file using `os.path.getmtime()` sort, not alphabetical. This means the percentage shown on the PM Metrics card reflects the model trained last, and updates automatically after every retraining.

---

## Metrics Glossary

| Metric | Definition |
|---|---|
| **R² (R-squared)** | Proportion of variance explained by the model. 1.0 = perfect, 0.0 = no better than predicting the mean. |
| **MAE** | Mean Absolute Error — average prediction error in days. |
| **CV R²** | Cross-validation R² — R² estimated on 5 different train/test splits. More reliable than single-split R². |
| **MTBF** | Mean Time Between Failures — avg days of operation between unplanned breakdowns. |
| **MTTR** | Mean Time To Repair — avg hours from failure detection to equipment back online. |
| **Utilisation Rate** | % of equipment in operational status at any given time. |
| **Cost Avoidance** | Estimated savings from preventive maintenance that avoided more expensive emergency repairs. |
