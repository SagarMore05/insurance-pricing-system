from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://insurance_user:insurance_pass@localhost:5432/insurance_pricing_db"
    DATABASE_URL_SYNC: str = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_pricing_db"
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    SECRET_KEY: str = "change-me-in-production"
    ADMIN_API_KEY: str = "admin-secret-key"
    DEBUG: bool = False

    # Admin JWT authentication
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "admin123"
    JWT_SECRET_KEY: str = "change-me-jwt-secret-key-min-32-chars"
    JWT_EXPIRE_MINUTES: int = 60

    MIN_PREMIUM_INR: float = 3000.0
    MAX_PREMIUM_INR: float = 150000.0
    MODEL_VERSION: str = "v1.0.0"
    RETRAIN_MIN_SAMPLES: int = 500

    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASS: str = ""
    ADMIN_EMAIL: str = ""

    REDIS_URL: str = "redis://localhost:6379/0"
    MODELS_DIR: str = "models/saved"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # Safety guard: must be explicitly set to true to enable the weekly scheduler.
    # Default false prevents accidental V1 retraining from displacing V4 champions.
    ENABLE_SCHEDULED_RETRAINING: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()