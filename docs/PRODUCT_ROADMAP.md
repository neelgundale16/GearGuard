# GearGuard — Product Roadmap

## Product Vision

Enterprise-grade SaaS platform for AI-powered predictive maintenance management — reducing unplanned downtime by up to 40% through ML-driven failure forecasting and smart scheduling.

---

## Current State (v1.0 — Deployed)

| Feature | Status |
|---|---|
| Equipment asset management | ✅ Live |
| Maintenance request lifecycle (Kanban + Calendar) | ✅ Live |
| Predictive maintenance ML model (ensemble) | ✅ Live |
| Analytics dashboard (predictions, urgency scores) | ✅ Live |
| PM Metrics dashboard (DAU, ROI, cost avoidance, live ML accuracy) | ✅ Live |
| Role-based access (Admin / Technician) | ✅ Live |
| Cloud deployment on Render (PostgreSQL) | ✅ Live |
| Data seeding management command (300 equip, 5000+ records) | ✅ Live |
| Optional LLM insights (Gemini free tier / rule-based fallback) | ✅ Live |

---

## Target Market

- Manufacturing plants (primary)
- Facility management companies
- Industrial equipment operators
- Oil & gas, pharmaceuticals, food processing

---

## Key Metrics (KPIs)

### User Engagement
- Daily Active Users (DAU) / Monthly Active Users (MAU)
- DAU/MAU Stickiness ratio
- Session duration and page depth

### Business Metrics
- Monthly Recurring Revenue (MRR)
- Customer Acquisition Cost (CAC)
- Lifetime Value (LTV)
- Churn Rate

### Product Metrics
- Equipment tracked per tenant
- Maintenance requests processed per month
- ML prediction accuracy (R² score, live on PM dashboard)
- Cost avoidance generated ($)
- Emergency-to-Preventive ratio (lower = better)

---

## Feature Prioritisation (RICE Score)

| Feature | Reach | Impact | Confidence | Effort | RICE | Priority |
|---|---|---|---|---|---|---|
| ML Predictive Maintenance (done) | 1000 | 3 | 85% | 5 | 510 | ✅ Done |
| PM Metrics Dashboard (done) | 800 | 2 | 90% | 3 | 480 | ✅ Done |
| Anomaly Detection | 700 | 3 | 75% | 4 | 394 | Q3 2025 |
| Email / SMS Notifications | 900 | 2 | 80% | 3 | 480 | Q3 2025 |
| Cost Prediction Model | 600 | 3 | 70% | 4 | 315 | Q4 2025 |
| Multi-tenancy (Organisation isolation) | 500 | 3 | 70% | 8 | 131 | Q4 2025 |
| REST API (third-party integrations) | 400 | 3 | 75% | 6 | 150 | Q1 2026 |
| IoT Sensor Integration | 300 | 3 | 60% | 10 | 54 | Q2 2026 |
| Mobile App (iOS / Android) | 500 | 2 | 65% | 12 | 54 | Q2 2026 |
| Time Series Forecasting | 400 | 2 | 65% | 5 | 104 | Q1 2026 |

---

## Roadmap by Quarter

### Q3 2025 — Alerts & Anomaly Detection
- [ ] Email / SMS notification system (maintenance due, overdue, critical alerts)
- [ ] Isolation Forest anomaly detection model
- [ ] Anomaly alert dashboard card
- [ ] Bulk equipment import via CSV

### Q4 2025 — Cost Intelligence & Multi-Tenancy
- [ ] Cost prediction model (Random Forest on historical cost data)
- [ ] Cost avoidance breakdown per equipment / work centre
- [ ] Organisation-level data isolation (multi-tenant architecture)
- [ ] Subscription plan enforcement (equipment limits per plan)

### Q1 2026 — API & Integrations
- [ ] REST API (Django REST Framework, JWT auth)
- [ ] API documentation (Swagger / ReDoc)
- [ ] Third-party integrations (SAP PM, Salesforce Field Service)
- [ ] Time Series Forecasting (SARIMA / Prophet for monthly load prediction)

### Q2 2026 — IoT & Mobile
- [ ] IoT sensor data ingestion (MQTT / WebSocket)
- [ ] Real-time equipment telemetry dashboard
- [ ] Mobile app (React Native — iOS + Android)
- [ ] Offline-capable mobile maintenance logging

### Q3 2026 — Enterprise
- [ ] White-label / custom branding
- [ ] Advanced RBAC (custom permission sets)
- [ ] Automated PDF / Excel report generation
- [ ] NLP on maintenance notes (extract failure categories automatically)
- [ ] Multi-language support (Hindi, Arabic, German)

---

## Go-to-Market Strategy

### Phase 1 — Validation (Now)
- Free tier for SMBs (up to 10 equipment)
- Focus: Get 20 pilot customers, gather feedback on ML accuracy

### Phase 2 — Monetisation (Q4 2025)
- Launch paid tiers
- Target: $10K MRR by end of Q4 2025

### Phase 3 — Scale (2026)
- Enterprise sales motion
- System integrator partnerships
- Target: $100K ARR

---

## Pricing Strategy

| Plan | Price | Equipment Limit | Users | ML Model | Support |
|---|---|---|---|---|---|
| **Free** | $0/mo | 10 | 3 | Basic predictions | Community |
| **Starter** | $49/mo | 50 | 10 | Full ensemble | Email |
| **Professional** | $199/mo | 200 | 50 | Full ensemble + anomaly | Priority |
| **Enterprise** | Custom | Unlimited | Unlimited | Custom models + API | Dedicated |

---

## Technical Debt & Known Issues

| Issue | Priority | Target |
|---|---|---|
| `ml_models/` directory not persistent on Render free tier (ephemeral disk) — model is lost on restart | High | Add S3/R2 storage for model files |
| DAU/WAU/MAU low in dev (only admin logs in) — stickiness shows 100% | Medium | Simulate sessions in seed_data |
| No background task runner — seed_data and training block the web process | Medium | Add Celery + Redis |
| Organisation FK on Equipment / WorkCenter is nullable — multi-tenancy not enforced | High | Complete multi-tenant isolation |
