"""
Application configuration.
Values are read from environment variables (or a .env file).
"""
from pathlib import Path
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
        "postgresql+psycopg2://postgres:postgres@localhost:5433/AutoML"
    )

    # ---------- LLM / Agent ----------
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "qwen3:14b"
    LLM_API_KEY: str = "ollama"
    LLM_BASE_URL: str | None = None
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 4096
    AGENT_MAX_ITERATIONS: int = 8

    # ---------- Data limits ----------
    MAX_FILE_SIZE_MB: int = 200
    MAX_ROWS_SAMPLE: int = 100_000

    # ---------- ML ----------
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.1
    MAX_CANDIDATE_MODELS: int = 5

    # ---------- App Settings ----------
    APP_NAME: str = "AutoML Agent"
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
settings.ensure_dirs()