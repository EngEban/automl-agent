"""
Application configuration.
Values are read from environment variables (or a .env file).
"""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- Paths ----------
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    REPORT_DIR: Path = STORAGE_DIR / "reports"
    FIGURE_DIR: Path = STORAGE_DIR / "figures"
    MODEL_DIR: Path = STORAGE_DIR / "models"

    # ---------- API ----------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8100
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # ---------- Celery ----------
    CELERY_BROKER_URL: str = "redis://localhost:6479/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6479/0"

    # ---------- Database ----------
    DATABASE_URL: str = (
        "postgresql+psycopg2://postgres:postgres@localhost:5433/automl"
    )

    # ---------- Ml Pipeline / Agent ----------
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "qwen3:14b"
    LLM_API_KEY: str = "ollama"
    LLM_BASE_URL: str | None = None
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 4096
    AGENT_MAX_ITERATIONS: int = Field(default=8, gt=0, le=32)
    AGENT_MAX_TOOL_CALLS: int = Field(default=12, gt=0, le=64)
    AGENT_LLM_TIMEOUT_SECONDS: int = Field(default=120, gt=0, le=600)
    AGENT_TOOL_TIMEOUT_SECONDS: int = Field(default=600, gt=0, le=3600)
    # ---------- Report Agent (Groq) ----------
    REPORT_LLM_PROVIDER: str = "groq"
    REPORT_LLM_MODEL: str = "openai/gpt-oss-20b"
    REPORT_LLM_API_KEY: str = ""
    REPORT_AGENT_ENABLED: bool = False
    REPORT_LLM_MAX_TOKENS: int = Field(default=4096, gt=0, le=16_000)
    REPORT_LLM_TIMEOUT_SECONDS: int = Field(default=90, gt=0, le=600)

    # ---------- Data limits ----------
    MAX_FILE_SIZE_MB: int = 200
    MAX_ROWS_SAMPLE: int = Field(
        default=100_000,
        gt=0,
        description="Maximum loaded data rows per dataset after deterministic sampling.",
    )
    CSV_CHUNK_SIZE: int = Field(
        default=10_000,
        gt=0,
        le=1_000_000,
        description="Rows read per CSV parser chunk while loading and sampling.",
    )
    DATASET_MEMORY_WARNING_MB: int = Field(
        default=512,
        gt=0,
        description="Warn when the deep pandas estimate for a dataset exceeds this many MiB.",
    )

    # ---------- ML ----------
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.1
    MAX_CANDIDATE_MODELS: int = 5
    MAX_DENSE_CONVERSION_MB: int = Field(
        default=512,
        gt=0,
        description="Maximum estimated size of a single sparse-to-dense conversion in MiB.",
    )
    PREDICTION_BATCH_SIZE: int = Field(
        default=256,
        gt=0,
        description="Maximum number of records transformed together during prediction.",
    )

    # ---------- App Settings ----------
    APP_NAME: str = "AutoML Agent"
    ENVIRONMENT: str = "development"
    APP_BASE_URL: str = "http://localhost:8100"
    FRONTEND_URL: str = "http://localhost:5173"

    # ---------- Auth (JWT) ----------
    SECRET_KEY: str = "CHANGE_ME"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080

    # ---------- Google OAuth ----------
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8100/api/auth/google/callback"

    # ---------- Email (Resend) ----------
    RESEND_API_KEY: str = ""
    EMAIL_FROM: str = "onboarding@resend.dev"
    EMAIL_FROM_NAME: str = "AutoML Support"
    EMAIL_VERIFY_SUBJECT: str = "Verify your Account"

    def ensure_dirs(self) -> None:
        """Create storage directories if they do not exist yet."""
        for directory in (
            self.UPLOAD_DIR,
            self.REPORT_DIR,
            self.FIGURE_DIR,
            self.MODEL_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()


def validate_security_settings(config: Settings) -> None:
    """Fail fast when production is configured with an unsafe JWT secret."""
    if config.ENVIRONMENT.strip().lower() != "production":
        return
    secret_key = config.SECRET_KEY.strip()
    if secret_key == "CHANGE_ME" or len(secret_key) < 32:
        raise RuntimeError(
            "SECRET_KEY must be at least 32 characters and must not be "
            "'CHANGE_ME' when ENVIRONMENT=production."
        )


validate_security_settings(settings)
settings.ensure_dirs()
