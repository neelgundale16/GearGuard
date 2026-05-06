# GearGuard — AI-Powered Industrial Maintenance Management System

## Overview

GearGuard is a comprehensive predictive maintenance management system built with Django and an ensemble ML pipeline. It predicts equipment failure windows, optimises maintenance scheduling, and surfaces real-time operational analytics for industrial facilities — with zero paid API dependency.

### Key Features

✅ **Equipment Tracking** — Complete asset management with full maintenance histories  
✅ **ML Predictions** — 75–88% accurate failure-window forecasting (Gradient Boosting / XGBoost / Random Forest ensemble)  
✅ **Real-time Dashboard** — Live stats pulled from the database, auto-refreshes every 30 s  
✅ **Maintenance Scheduling** — Calendar and Kanban workflow views  
✅ **PM Analytics Dashboard** — MTBF, MTTR, utilisation, cost avoidance, ROI — all live from DB  
✅ **Role-based Access** — Admin and Technician permission levels  
✅ **Mobile Responsive** — Bootstrap 5 — works on all devices  
✅ **Cloud-ready** — Deployed on Render (PostgreSQL + Gunicorn + WhiteNoise)  
✅ **Optional LLM** — Google Gemini 1.5 Flash (free tier) for insights; full rule-based fallback if no key set

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django 5.0 (Python 3.11) |
| Database | SQLite (dev) / PostgreSQL (Render production) |
| ML | Scikit-learn, XGBoost, NumPy, Joblib |
| Frontend | Bootstrap 5.3, Chart.js, Font Awesome, ES6 JS |
| Deployment | Render (free tier), Gunicorn, WhiteNoise |
| LLM (optional) | Google Gemini 1.5 Flash via `GEMINI_API_KEY` env var |

> **Note:** Ollama / Llama 3.2 has been removed. It requires a localhost process and does not run on Render. The system works fully without any LLM key set.

---

## Machine Learning

### Predictive Maintenance Ensemble

**Algorithms:** Gradient Boosting + XGBoost + Random Forest — best model saved per training run  
**Accuracy:** 75–88% R² (realistic for noisy industrial data — 100% would be overfitting)  
**MAE:** 5–12 days

### Features (zero circular leakage)

| # | Feature | Why it is useful |
|---|---|---|
| 0 | `avg_historical_interval` | Each equipment has a characteristic base maintenance cadence (15–90 d) — primary signal |
| 1 | `interval_std` | Variability in that equipment's cadence |
| 2 | `recent_vs_historical_ratio` | Is the equipment deteriorating faster recently? |
| 3 | `equipment_age_years` | Older equipment needs more frequent maintenance |
| 4 | `days_since_last_maintenance` | Recency pressure |
| 5 | `total_maintenance_count` | Equipment maturity / history depth |
| 6 | `avg_actual_hours` | Job complexity proxy |
| 7 | `age_x_avg_interval` | Interaction term |

> **Priority and request type are NOT features.** In the previous broken version they were derived from the interval itself (interval ≤ 10 → emergency → critical), so the model learned priority → interval and scored 100% R². Now they are assigned independently of interval, eliminating the circular leakage.

### Output per Equipment

- Days until next maintenance
- Priority score (0–100, unique per equipment)
- Urgency: Low / Medium / High / Critical
- Recommended action text

---

## Data Generation

No Ollama. No external scripts. One Django management command:

```bash
python manage.py seed_data
```

Generates in ~25 seconds:
- **300 equipment** across 10 realistic work centres
- **5 000+ completed maintenance records** with realistic per-equipment intervals + noise
- **20 technician users**
- ~70 live pending / in-progress requests

---

## Quick Start (Local Development)

```bash
# 1. Clone
git clone https://github.com/neelgundale16/GearGuard.git
cd GearGuard

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file in project root
SECRET_KEY=any-random-string-here
DEBUG=True
# Optional — only needed for AI insight features:
# GEMINI_API_KEY=your-free-gemini-key

# 5. Run migrations
python manage.py migrate

# 6. Seed data (~25 s)
python manage.py seed_data

# 7. Start server
python manage.py runserver

# 8. Open http://127.0.0.1:8000
#    Admin login:      admin / admin123
#    Technician login: raj_sharma / tech123
```

### Train the ML Model (after seeding)

**Via UI:** Log in as admin → Analytics → Train Model → Start Training

**Via shell:**
```bash
python manage.py shell -c "
from maintenance.ml_engine import RealMLEngine
from maintenance.models import Equipment
r = RealMLEngine().train_models(Equipment.objects.all())
print(r['best_model'], round(r['results'][r['best_model']]['r2']*100, 1), '%')
"
```

---

## Commands Reference — What to Run After Each Change

| What you changed | Commands needed |
|---|---|
| Python / view / template / HTML only | Just restart `python manage.py runserver` — nothing else |
| `models.py` (new field, new model) | `python manage.py makemigrations && python manage.py migrate` then restart |
| Want fresh data (wipe + regenerate) | `python manage.py seed_data` |
| Retrain ML model | Admin UI → Train Model, or shell command above |
| Static files (CSS / JS) | `python manage.py collectstatic` (production only; dev auto-serves) |
| First-time setup | `migrate` → `seed_data` → `runserver` |
| Pushed new code to Render | Render auto-runs `build.sh` (migrate + seed_data included) |

---

## Deployment on Render

### `render.yaml`

```yaml
services:
  - type: web
    name: gearguard
    env: python
    plan: free
    buildCommand: ./build.sh
    startCommand: gunicorn gearguard_project.wsgi:application
    envVars:
      - key: SECRET_KEY
        generateValue: true
      - key: DEBUG
        value: False
      - key: DATABASE_URL
        fromDatabase:
          name: gearguard-db
          property: connectionString
      # - key: GEMINI_API_KEY
      #   value: your-key-here   (optional)

databases:
  - name: gearguard-db
    databaseName: gearguard
    plan: free
```

### `build.sh`

```bash
#!/usr/bin/env bash
set -o errexit
pip install -r requirements.txt
python manage.py collectstatic --no-input
python manage.py migrate
python manage.py seed_data
```

### Post-Deploy — Train Model Once

```bash
# In Render Shell (Dashboard → Shell tab)
python manage.py shell -c "
from maintenance.ml_engine import RealMLEngine
from maintenance.models import Equipment
r = RealMLEngine().train_models(Equipment.objects.all())
print('Done:', r['best_model'], round(r['results'][r['best_model']]['r2']*100, 1), '%')
"
```

---

## User Roles

| Role | Access |
|---|---|
| **Admin** | Full access — train ML, manage users, delete records, view PM metrics dashboard |
| **Technician** | Create / update maintenance requests, view equipment, read-only analytics |

---

## Project Structure

```
GearGuard/
├── maintenance/
│   ├── management/
│   │   ├── __init__.py
│   │   └── commands/
│   │       ├── __init__.py
│   │       └── seed_data.py       ← replaces generate_production_data.py
│   ├── ml_engine.py               ← ensemble training + prediction (fixed leakage + mtime sort)
│   ├── llm_engine.py              ← Gemini free tier / rule-based fallback (Ollama removed)
│   ├── pm_metrics.py              ← PM dashboard (live DB + live model accuracy via mtime sort)
│   ├── models.py
│   ├── views.py
│   └── ...
├── templates/
├── static/
├── ml_models/                     ← auto-created, stores timestamped .pkl files
├── build.sh
├── render.yaml
└── requirements.txt
```

---

## Performance

| Metric | Value |
|---|---|
| ML Accuracy | 75–88% R² (varies per run — intentional, not a bug) |
| Training time | ~15–30 s (300 equipment, 5 000+ records) |
| Seed time | ~25 s (bulk inserts) |
| Page load | < 2 s |
| API response | < 100 ms |

---

## Author

**Neel Gundale**
- LinkedIn: [linkedin.com/in/neelgundale16](https://www.linkedin.com/in/neelgundale16/)
- Email: neelgundale@gmail.com

Built for industrial facilities worldwide.
