# InsureAI — AI-Powered Car Insurance Pricing System

An end-to-end machine learning system for actuarially-sound car insurance pricing with a React frontend, FastAPI backend, and continuous retraining pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Customer Browser (React + Tailwind + Recharts)                  │
│  Multi-step Quote Form → Premium Display → Policy Purchase      │
└──────────────────────────────┬──────────────────────────────────┘
                               │ REST API
┌──────────────────────────────▼──────────────────────────────────┐
│  FastAPI Backend  (src/api/)                                     │
│  POST /quote  POST /policies  POST /claims  GET /admin/*        │
└────────────┬────────────────────────────────────────────────────┘
             │
     ┌───────▼────────┐   ┌──────────────────────────────────────┐
     │ ML Models       │   │ Pricing Engine (rules-based, no LLM) │
     │ Model 1:        │──▶│                                      │
     │  Claim Freq     │   │  expected_loss = P(claim) × severity │
     │  (XGBClassifier)│   │  base_premium = expected_loss × 1.35 │
     │ Model 2:        │   │  + business rule multipliers          │
     │  Claim Severity │   │  → final_premium (floor ₹3k / cap    │
     │  (XGBRegressor) │   │    ₹1.5L) + SHAP explanation         │
     └───────┬─────────┘   └──────────────────────────────────────┘
             │
     ┌───────▼──────────────────┐  ┌──────────────────────────────┐
     │ PostgreSQL (SQLAlchemy)   │  │ Feedback Loop (APScheduler)  │
     │ 8 tables with FK/indexes  │  │ Weekly retrain on actual     │
     │ + read-only role for NLP  │  │ claim outcomes               │
     └───────────────────────────┘  └──────────────────────────────┘
             │
     ┌───────▼────────────────────┐
     │ NLP Assistant (LangChain)  │
     │ GPT-4 → SQL → read-only DB │
     │ Parameterized · whitelist  │
     └────────────────────────────┘
```

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Python 3.11+ (for local dev)
- Node.js 20+ (for local frontend dev)

### 1. Start with Docker

```bash
cd insurance-pricing-system

# Copy env file and configure
cp backend/.env.example backend/.env
# Edit backend/.env — add OPENAI_API_KEY if you want the NLP assistant

# Start all services
docker compose up -d

# Run database migrations
docker compose exec api alembic upgrade head

# Generate synthetic training data
docker compose exec api python scripts/generate_synthetic_data.py

# Train ML models (takes ~3-5 minutes)
docker compose exec api python scripts/train_models.py
```

Services:
| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| pgAdmin | http://localhost:5050 (profile: dev) |

### 2. Local Development (without Docker)

```bash
# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Set up PostgreSQL and update .env
cp .env.example .env

# Run migrations
alembic upgrade head

# Generate data and train models
python scripts/generate_synthetic_data.py
python scripts/train_models.py

# Start API
uvicorn src.api.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

## API Reference

### POST /api/v1/quote
Get an AI-calculated insurance premium.

**Request body:**
```json
{
  "age": 32,
  "gender": "male",
  "city": "mumbai",
  "car_brand": "hyundai",
  "car_model": "Creta",
  "engine_cc": 1500,
  "vehicle_age_years": 2,
  "vehicle_value_inr": 1200000,
  "driving_score": 78.5,
  "annual_mileage_km": 15000,
  "previous_claims_count": 0,
  "years_licensed": 8
}
```

**Response:**
```json
{
  "prediction_id": "uuid",
  "premium_amount_inr": 18450.0,
  "risk_level": "medium",
  "claim_probability": 0.17823,
  "expected_claim_amount_inr": 72000.0,
  "explanation": {
    "base_premium": 17379.5,
    "expected_loss": 12873.0,
    "adjustments": [
      {"reason": "no_claims_bonus", "factor": 0.9, "impact_inr": -1737.9}
    ],
    "final_premium": 15641.5,
    "risk_level": "medium",
    "summary": "Your premium is ₹15,642 (Medium Risk). Your clean claims history saved you ₹1,738."
  },
  "model_version": "v1.0.0",
  "created_at": "2026-01-15T10:30:00Z"
}
```

### POST /api/v1/policies
Convert a quote into an active policy.
```json
{ "prediction_id": "uuid", "customer_data": {...} }
```

### POST /api/v1/claims
File a claim against a policy.
```json
{ "policy_id": "uuid", "claimed_amount_inr": 45000 }
```

### GET /api/v1/admin/dashboard
Requires `X-Admin-Key` header. Returns KPI stats.

### POST /api/v1/assistant/query
Requires `X-Admin-Key`. Natural language to SQL.
```json
{ "question": "What is the average premium for high-risk customers in Mumbai?" }
```

### POST /api/v1/feedback
Submit actual claim outcome for model retraining.
```json
{
  "prediction_id": "uuid",
  "actual_claim_occurred": true,
  "actual_claim_amount_inr": 38000
}
```

### POST /api/v1/admin/retrain
Manually trigger the retraining pipeline. Requires `X-Admin-Key`.

## ML Pipeline

### Models
| Model | Type | Algorithm | Primary Metric |
|-------|------|-----------|----------------|
| Frequency | Binary classification | XGBoostClassifier | AUC-ROC |
| Severity | Regression (log-scale) | XGBoostRegressor | RMSE |

**Expected Loss Formula:**
```
expected_loss = P(claim) × E[claim_amount | claim occurred]
base_premium  = expected_loss × 1.35  (35% loading factor)
```

### Pricing Rules
| Rule | Trigger | Factor |
|------|---------|--------|
| Young driver | age < 25 | ×1.30 |
| Senior driver | age > 70 | ×1.15 |
| Safe driver discount | score ≥ 85 | ×0.85 |
| Poor driver surcharge | score < 50 | ×1.25 |
| No claims bonus | 0 claims | ×0.90 |
| High claims surcharge | ≥ 3 claims | ×1.40 |
| High mileage | > 30,000 km | ×1.20 |
| Low mileage discount | < 5,000 km | ×0.92 |
| Luxury vehicle | value > ₹20L | ×1.15 |

**Floor:** ₹3,000 · **Cap:** ₹1,50,000

### Retraining Pipeline
- Runs every Sunday at 02:00 UTC (APScheduler)
- Requires ≥ 500 new feedback records
- Triggers if AUC-ROC drops below 0.72 or RMSE degrades >15%
- Shadow testing: new model runs in parallel for 24h before promotion
- Admin email notification on promotion

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Async PostgreSQL URL | — |
| `OPENAI_API_KEY` | Required for NLP assistant | — |
| `ADMIN_API_KEY` | Admin endpoint auth | `admin-secret-key` |
| `MIN_PREMIUM_INR` | Premium floor | `3000` |
| `MAX_PREMIUM_INR` | Premium cap | `150000` |
| `MODEL_VERSION` | Active model version tag | `v1.0.0` |
| `RETRAIN_MIN_SAMPLES` | Min feedback rows for retraining | `500` |

## Running Tests

```bash
cd backend
pytest tests/ -v
```

## Project Structure

```
insurance-pricing-system/
├── backend/
│   ├── src/
│   │   ├── api/           FastAPI routes, Pydantic schemas
│   │   ├── preprocessing/ InsurancePreprocessor pipeline
│   │   ├── models/        Frequency, Severity, CombinedPredictor
│   │   ├── engine/        Deterministic pricing engine
│   │   ├── nlp_assistant/ LangChain SQL agent
│   │   ├── retraining/    Feedback pipeline + APScheduler
│   │   └── database/      SQLAlchemy ORM models + session
│   ├── tests/             pytest test suite
│   ├── alembic/           DB migrations
│   ├── scripts/           Data generation + training
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/    QuoteForm, QuoteResult, AdminDashboard
│       ├── pages/         Home, Quote, Policy, Admin
│       ├── api/           Typed API client
│       └── types/         TypeScript interfaces
├── docker-compose.yml
└── docker-compose.prod.yml
```
