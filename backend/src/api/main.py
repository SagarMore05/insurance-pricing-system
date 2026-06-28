import sys
import asyncio
import time
import logging
from contextlib import asynccontextmanager

# Windows ProactorEventLoop raises RuntimeError on __del__ when the loop is
# already closed. SelectorEventLoop does not have this cleanup issue.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.api.routes import quote, policies, claims, admin, feedback, monitoring, drift, model_registry_api, model_selection_api, quote_v2
from src.auth.routes import router as auth_router
from src.models.combined_predictor import get_predictor
from src.config import settings

logger = logging.getLogger("insurance_api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background scheduler for weekly model retraining — guarded by config flag.
    # Disabled by default to prevent V1 pipeline from displacing V4 champions.
    if settings.ENABLE_SCHEDULED_RETRAINING:
        try:
            from src.retraining.scheduler import start_scheduler
            start_scheduler()
            logger.info("Scheduled retraining ENABLED — weekly job registered (Sunday 02:00 UTC)")
        except Exception as _sched_exc:
            logger.warning("APScheduler not started: %s", _sched_exc)
    else:
        logger.warning(
            "Scheduled retraining DISABLED by configuration "
            "(ENABLE_SCHEDULED_RETRAINING=false). "
            "Set to true in .env only after V4 online pipeline is ready."
        )

    # Load ML models at startup
    predictor = get_predictor()
    if predictor.is_ready():
        logger.info("ML models loaded successfully (version %s)", settings.MODEL_VERSION)
    else:
        logger.warning("ML models not found — run training pipeline before serving predictions")

    # Initialise LangGraph HITL checkpointer and compile underwriting graph
    try:
        from src.agents.underwriting.checkpointer import setup_checkpointer
        from src.agents.underwriting.graph import setup_underwriting_graph
        await setup_checkpointer()
        await setup_underwriting_graph()
        logger.info("Underwriting HITL checkpointer and graph ready")
    except Exception as _hitl_exc:
        logger.warning(
            "HITL setup failed — underwriting agent will run without persistent checkpointer: %s",
            _hitl_exc,
        )

    # Warn on insecure default credentials
    if settings.JWT_SECRET_KEY == "change-me-jwt-secret-key-min-32-chars":
        logger.warning("SECURITY: JWT_SECRET_KEY is the default value — set a strong secret in .env before production")
    if settings.ADMIN_PASSWORD == "admin123":
        logger.warning("SECURITY: ADMIN_PASSWORD is the default value — change in .env before production")
    if settings.SECRET_KEY == "change-me-in-production":
        logger.warning("SECURITY: SECRET_KEY is the default value — rotate before production")

    yield

    # Gracefully stop the scheduler on shutdown
    try:
        from src.retraining.scheduler import stop_scheduler
        stop_scheduler()
    except Exception:
        pass


app = FastAPI(
    title="Insurance Pricing API",
    version="1.0.0",
    description="AI-Powered Car Insurance Pricing System",
    lifespan=lifespan,
)

# ─── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 1)
    logger.info("%s %s → %d (%sms)", request.method, request.url.path, response.status_code, duration)
    return response


# ─── Exception Handlers ───────────────────────────────────────────────────────

def _serializable_errors(exc: RequestValidationError) -> list:
    """
    Pydantic v2 field_validators that raise ValueError leave a live exception
    object at ctx["error"], which json.dumps cannot serialize.  Convert every
    Exception in any ctx dict to its string representation so JSONResponse
    never raises TypeError.
    """
    out = []
    for e in exc.errors():
        entry = dict(e)
        if "ctx" in entry:
            entry["ctx"] = {
                k: str(v) if isinstance(v, Exception) else v
                for k, v in entry["ctx"].items()
            }
        out.append(entry)
    return out


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        media_type="application/problem+json",
        content={
            "type": "https://tools.ietf.org/html/rfc7807",
            "title": "Validation Error",
            "status": 422,
            "detail": _serializable_errors(exc),
            "instance": str(request.url),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        media_type="application/problem+json",
        content={
            "type": "https://tools.ietf.org/html/rfc7807",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An unexpected error occurred. Please try again later.",
            "instance": str(request.url),
        },
    )


# ─── Routers ──────────────────────────────────────────────────────────────────

PREFIX = "/api/v1"

# Auth router — must be registered before protected routers, no auth dependency
app.include_router(auth_router, prefix=PREFIX)

app.include_router(quote.router, prefix=PREFIX, tags=["Quotes"])
app.include_router(policies.router, prefix=PREFIX, tags=["Policies"])
app.include_router(claims.router, prefix=PREFIX, tags=["Claims"])
app.include_router(admin.router, prefix=PREFIX, tags=["Admin"])
app.include_router(monitoring.router, prefix=PREFIX, tags=["Monitoring"])
app.include_router(drift.router, prefix=PREFIX, tags=["Drift Detection"])
app.include_router(feedback.router, prefix=PREFIX, tags=["Feedback"])
app.include_router(model_registry_api.router,   prefix=PREFIX, tags=["Model Registry"])
app.include_router(model_selection_api.router, prefix=PREFIX, tags=["Model Selection"])
app.include_router(quote_v2.router, prefix="/api/v2", tags=["Quotes V2"])

# NLP assistant router (optional — requires GROQ_API_KEY)
try:
    from src.nlp_assistant.routes import router as assistant_router
    app.include_router(assistant_router, prefix=PREFIX, tags=["Assistant"])
    logger.info("NLP assistant loaded")
except Exception as e:
    logger.warning("NLP assistant not loaded: %s", e)

# Retraining router
try:
    from src.retraining.routes import router as retrain_router
    app.include_router(retrain_router, prefix=PREFIX, tags=["Retraining"])
except Exception as e:
    logger.warning("Retraining routes not loaded: %s", e)

# Analytics Agent router (optional — requires GROQ_API_KEY + Redis)
try:
    from src.agents.analytics.routes import router as analytics_router
    app.include_router(analytics_router, prefix=PREFIX, tags=["Analytics Agent"])
    logger.info("Analytics agent loaded")
except Exception as e:
    logger.warning("Analytics agent not loaded: %s", e)

# Underwriting Agent router (optional — requires GROQ_API_KEY + ML models)
try:
    from src.agents.underwriting.routes import router as underwriting_router
    from src.agents.underwriting.review_routes import router as review_router
    app.include_router(underwriting_router, prefix=PREFIX, tags=["Underwriting Agent"])
    app.include_router(review_router, prefix=PREFIX, tags=["Underwriting Review"])
    logger.info("Underwriting agent loaded")
except Exception as e:
    logger.warning("Underwriting agent not loaded: %s", e)


@app.get("/health")
async def health_check():
    predictor = get_predictor()
    return {
        "status": "ok",
        "models_loaded": predictor.is_ready(),
        "model_version": settings.MODEL_VERSION,
    }
