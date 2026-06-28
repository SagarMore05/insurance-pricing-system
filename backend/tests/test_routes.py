"""
Route-level integration tests.

Uses httpx.AsyncClient + ASGITransport — no real HTTP port is opened.
All DB, ML predictor, and NLP assistant dependencies are isolated via
app.dependency_overrides or unittest.mock.patch.
No real PostgreSQL, Groq API, or model files are required.
"""
import uuid
from datetime import datetime, date
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.api.main import app
from src.database.session import get_db
from src.config import settings
from src.auth.security import create_access_token

_assistant_route_registered = any(
    "assistant" in getattr(r, "path", "")
    for r in app.routes
)

PREFIX = "/api/v1"
ADMIN_HDR = {"Authorization": f"Bearer {create_access_token(username=settings.ADMIN_USERNAME)}"}
WRONG_HDR = {"Authorization": "Bearer invalid.jwt.token"}


# ── DB mock helpers ────────────────────────────────────────────────────────────

def scalar_result(value):
    """Mock execute() result whose .scalar() and .scalar_one_or_none() return value."""
    r = MagicMock()
    r.scalar.return_value = value
    r.scalar_one_or_none.return_value = value
    return r


def all_result(rows):
    """Mock execute() result whose .all() returns rows."""
    r = MagicMock()
    r.all.return_value = rows
    return r


def make_db(*execute_returns):
    """
    Return a get_db dependency override yielding a mock AsyncSession.
    execute_returns are consumed in sequence by each db.execute() call the
    route handler makes. Routes that only call db.add/flush need no arguments.
    """
    async def _override():
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=list(execute_returns))
        session.add = MagicMock()
        session.flush = AsyncMock()
        yield session
    return _override


# ── Shared valid request body ──────────────────────────────────────────────────

_VALID_QUOTE = {
    "age": 30,
    "gender": "male",
    "city": "mumbai",
    "car_brand": "maruti",
    "car_model": "swift",
    "engine_cc": 1200,
    "vehicle_age_years": 3,
    "vehicle_value_inr": 600000,
    "driving_score": 75.0,
    "annual_mileage_km": 12000,
    "previous_claims_count": 0,
    "years_licensed": 5,
}


# ══════════════════════════════════════════════════════════════════════════════
# GET /health
# ══════════════════════════════════════════════════════════════════════════════

async def test_health_models_ready(client):
    with patch("src.api.main.get_predictor") as mock_gp:
        mock_gp.return_value.is_ready.return_value = True
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True


async def test_health_models_not_ready(client):
    with patch("src.api.main.get_predictor") as mock_gp:
        mock_gp.return_value.is_ready.return_value = False
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["models_loaded"] is False


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/v1/quote
# ══════════════════════════════════════════════════════════════════════════════

async def test_quote_returns_201_when_models_ready(client):
    app.dependency_overrides[get_db] = make_db()
    try:
        with patch("src.api.routes.quote.get_predictor") as mock_gp:
            mock_gp.return_value.is_ready.return_value = True
            mock_gp.return_value.predict.return_value = {
                "claim_probability": 0.12,
                "expected_claim_amount_inr": 50000.0,
                "expected_loss_inr": 6000.0,
            }
            resp = await client.post(f"{PREFIX}/quote", json=_VALID_QUOTE)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 201
    body = resp.json()
    assert "prediction_id" in body
    assert "premium_amount_inr" in body
    assert body["risk_level"] in ("low", "medium", "high")
    assert body["claim_probability"] == pytest.approx(0.12)


async def test_quote_returns_503_when_models_not_ready(client):
    app.dependency_overrides[get_db] = make_db()
    try:
        with patch("src.api.routes.quote.get_predictor") as mock_gp:
            mock_gp.return_value.is_ready.return_value = False
            resp = await client.post(f"{PREFIX}/quote", json=_VALID_QUOTE)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 503


async def test_quote_returns_422_on_invalid_payload(client):
    resp = await client.post(f"{PREFIX}/quote", json={"age": 10})
    assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/v1/policies
# ══════════════════════════════════════════════════════════════════════════════

async def test_create_policy_returns_201(client):
    pred_id = uuid.uuid4()
    pol_id  = uuid.uuid4()
    cust_id = uuid.uuid4()
    veh_id  = uuid.uuid4()

    mock_pred = MagicMock()
    mock_pred.prediction_id = pred_id
    mock_pred.policy_id = pol_id

    mock_pol = MagicMock()
    mock_pol.policy_id = pol_id
    mock_pol.customer_id = cust_id
    mock_pol.vehicle_id = veh_id
    mock_pol.premium_amount_inr = 45000.0
    mock_pol.risk_level = "low"
    mock_pol.model_version = "v1.0.0"
    mock_pol.created_at = datetime(2026, 1, 1)
    mock_pol.is_active = True

    app.dependency_overrides[get_db] = make_db(
        scalar_result(mock_pred),
        scalar_result(mock_pol),
    )
    try:
        resp = await client.post(
            f"{PREFIX}/policies",
            json={"prediction_id": str(pred_id), "customer_data": _VALID_QUOTE},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 201
    body = resp.json()
    assert body["policy_id"] == str(pol_id)
    assert body["is_active"] is True


async def test_create_policy_returns_404_when_prediction_missing(client):
    app.dependency_overrides[get_db] = make_db(scalar_result(None))
    try:
        resp = await client.post(
            f"{PREFIX}/policies",
            json={"prediction_id": str(uuid.uuid4()), "customer_data": _VALID_QUOTE},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/policies/{policy_id}
# ══════════════════════════════════════════════════════════════════════════════

async def test_get_policy_returns_200(client):
    pol_id  = uuid.uuid4()
    cust_id = uuid.uuid4()
    veh_id  = uuid.uuid4()

    mock_pol = MagicMock()
    mock_pol.policy_id = pol_id
    mock_pol.customer_id = cust_id
    mock_pol.vehicle_id = veh_id
    mock_pol.premium_amount_inr = 45000.0
    mock_pol.risk_level = "medium"
    mock_pol.model_version = "v1.0.0"
    mock_pol.created_at = datetime(2026, 1, 1)
    mock_pol.is_active = True

    app.dependency_overrides[get_db] = make_db(
        scalar_result(mock_pol),
        scalar_result(None),
    )
    try:
        resp = await client.get(f"{PREFIX}/policies/{pol_id}")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    body = resp.json()
    assert body["policy_id"] == str(pol_id)
    assert body["prediction"] is None


async def test_get_policy_returns_404_when_missing(client):
    app.dependency_overrides[get_db] = make_db(scalar_result(None))
    try:
        resp = await client.get(f"{PREFIX}/policies/{uuid.uuid4()}")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/v1/claims
# ══════════════════════════════════════════════════════════════════════════════

async def test_file_claim_returns_201(client):
    pol_id = uuid.uuid4()
    mock_pol = MagicMock()
    mock_pol.policy_id = pol_id
    mock_pol.is_active = True

    app.dependency_overrides[get_db] = make_db(scalar_result(mock_pol))
    try:
        resp = await client.post(
            f"{PREFIX}/claims",
            json={"policy_id": str(pol_id), "claimed_amount_inr": 25000.0},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 201
    assert resp.json()["claim_status"] == "pending"


async def test_file_claim_returns_404_when_policy_missing(client):
    app.dependency_overrides[get_db] = make_db(scalar_result(None))
    try:
        resp = await client.post(
            f"{PREFIX}/claims",
            json={"policy_id": str(uuid.uuid4()), "claimed_amount_inr": 10000.0},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 404


async def test_file_claim_returns_400_on_inactive_policy(client):
    pol_id = uuid.uuid4()
    mock_pol = MagicMock()
    mock_pol.policy_id = pol_id
    mock_pol.is_active = False

    app.dependency_overrides[get_db] = make_db(scalar_result(mock_pol))
    try:
        resp = await client.post(
            f"{PREFIX}/claims",
            json={"policy_id": str(pol_id), "claimed_amount_inr": 10000.0},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
# PATCH /api/v1/claims/{claim_id}
# ══════════════════════════════════════════════════════════════════════════════

async def test_update_claim_no_key_rejected(client):
    resp = await client.patch(
        f"{PREFIX}/claims/{uuid.uuid4()}",
        json={"claim_status": "approved"},
    )
    assert resp.status_code in (401, 422)


async def test_update_claim_rejects_wrong_key(client):
    resp = await client.patch(
        f"{PREFIX}/claims/{uuid.uuid4()}",
        headers=WRONG_HDR,
        json={"claim_status": "approved"},
    )
    assert resp.status_code == 401


async def test_update_claim_returns_200(client):
    claim_id = uuid.uuid4()
    pol_id   = uuid.uuid4()

    mock_claim = MagicMock()
    mock_claim.claim_id = claim_id
    mock_claim.policy_id = pol_id
    mock_claim.claimed_amount_inr = 30000.0
    mock_claim.approved_amount_inr = 30000.0
    mock_claim.claim_date = date(2026, 1, 15)
    mock_claim.claim_status = "approved"

    app.dependency_overrides[get_db] = make_db(scalar_result(mock_claim))
    try:
        resp = await client.patch(
            f"{PREFIX}/claims/{claim_id}",
            headers=ADMIN_HDR,
            json={"claim_status": "approved", "approved_amount_inr": 30000.0},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    assert resp.json()["claim_status"] == "approved"


async def test_update_claim_returns_404_when_missing(client):
    app.dependency_overrides[get_db] = make_db(scalar_result(None))
    try:
        resp = await client.patch(
            f"{PREFIX}/claims/{uuid.uuid4()}",
            headers=ADMIN_HDR,
            json={"claim_status": "rejected"},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/v1/feedback
# ══════════════════════════════════════════════════════════════════════════════

async def test_submit_feedback_returns_201(client):
    pred_id = uuid.uuid4()

    app.dependency_overrides[get_db] = make_db(scalar_result(MagicMock()))
    try:
        resp = await client.post(
            f"{PREFIX}/feedback",
            json={
                "prediction_id": str(pred_id),
                "actual_claim_occurred": True,
                "actual_claim_amount_inr": 20000.0,
            },
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 201


async def test_submit_feedback_returns_404_when_prediction_missing(client):
    app.dependency_overrides[get_db] = make_db(scalar_result(None))
    try:
        resp = await client.post(
            f"{PREFIX}/feedback",
            json={"prediction_id": str(uuid.uuid4()), "actual_claim_occurred": False},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/admin/dashboard
# ══════════════════════════════════════════════════════════════════════════════

async def test_admin_dashboard_no_key_rejected(client):
    resp = await client.get(f"{PREFIX}/admin/dashboard")
    assert resp.status_code in (401, 422)


async def test_admin_dashboard_returns_200(client):
    mock_risk_row = MagicMock()
    mock_risk_row.risk_level = "low"
    mock_risk_row.cnt = 5
    mock_risk_row.avg_premium = 30000.0

    app.dependency_overrides[get_db] = make_db(
        scalar_result(10),
        scalar_result(8),
        scalar_result(45000.0),
        scalar_result(3),
        scalar_result(360000.0),
        all_result([mock_risk_row]),
    )
    try:
        resp = await client.get(f"{PREFIX}/admin/dashboard", headers=ADMIN_HDR)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_policies"] == 10
    assert body["active_policies"] == 8
    assert body["total_claims"] == 3


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/admin/customers
# ══════════════════════════════════════════════════════════════════════════════

async def test_admin_customers_no_key_rejected(client):
    resp = await client.get(f"{PREFIX}/admin/customers")
    assert resp.status_code in (401, 422)


async def test_admin_customers_returns_200(client):
    app.dependency_overrides[get_db] = make_db(
        scalar_result(0),
        all_result([]),
    )
    try:
        resp = await client.get(f"{PREFIX}/admin/customers", headers=ADMIN_HDR)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["items"] == []
    assert body["page"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/admin/analytics/premium-trends
# ══════════════════════════════════════════════════════════════════════════════

async def test_premium_trends_no_key_rejected(client):
    resp = await client.get(f"{PREFIX}/admin/analytics/premium-trends")
    assert resp.status_code in (401, 422)


async def test_premium_trends_returns_200(client):
    app.dependency_overrides[get_db] = make_db(all_result([]))
    try:
        resp = await client.get(
            f"{PREFIX}/admin/analytics/premium-trends", headers=ADMIN_HDR
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    assert resp.json() == []


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/v1/assistant/query
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _assistant_route_registered, reason="NLP assistant router not registered (langchain_groq dependency missing)")
async def test_assistant_requires_admin_key(client):
    resp = await client.post(
        f"{PREFIX}/assistant/query",
        json={"question": "How many policies are there?"},
    )
    assert resp.status_code in (401, 422)


@pytest.mark.skipif(not _assistant_route_registered, reason="NLP assistant router not registered (langchain_groq dependency missing)")
async def test_assistant_returns_503_when_groq_key_missing(client, monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")
    resp = await client.post(
        f"{PREFIX}/assistant/query",
        headers=ADMIN_HDR,
        json={"question": "How many policies are there?"},
    )
    assert resp.status_code == 503


@pytest.mark.skipif(not _assistant_route_registered, reason="NLP assistant router not registered (langchain_groq dependency missing)")
async def test_assistant_returns_200_with_mock(client, monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
    with patch("src.nlp_assistant.routes.get_assistant") as mock_ga:
        mock_ga.return_value.query = AsyncMock(
            return_value={
                "answer": "There are 10 policies.",
                "sql_used": "SELECT COUNT(*) FROM policies LIMIT 100",
                "data": [{"col_0": 10}],
                "chart_suggestion": "none",
            }
        )
        resp = await client.post(
            f"{PREFIX}/assistant/query",
            headers=ADMIN_HDR,
            json={"question": "How many policies are there?"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "There are 10 policies."
    assert body["chart_suggestion"] == "none"
    assert body["sql_used"] is not None


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/v1/admin/retrain
# ══════════════════════════════════════════════════════════════════════════════

async def test_retrain_no_key_rejected(client):
    resp = await client.post(f"{PREFIX}/admin/retrain")
    assert resp.status_code in (401, 422)


async def test_retrain_returns_200_with_mock(client):
    app.dependency_overrides[get_db] = make_db()
    try:
        with patch("sqlalchemy.create_engine"), \
             patch("sqlalchemy.orm.sessionmaker") as mock_sm, \
             patch("src.retraining.pipeline.RetrainingPipeline") as MockPipeline:
            mock_sm.return_value.return_value = MagicMock()
            MockPipeline.return_value.run.return_value = {
                "frequency": {"auc_roc": 0.75},
            }
            resp = await client.post(f"{PREFIX}/admin/retrain", headers=ADMIN_HDR)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    body = resp.json()
    assert body["triggered"] is True
    assert "Retraining pipeline completed" in body["message"]
