# AutoML Agent

AutoML Agent is a session-based data analysis backend. It profiles datasets, prepares and evaluates models, records analysis runs, and produces reports and reusable model artifacts.

## Backend capabilities

- FastAPI endpoints for authentication, datasets, chat, runs, SSE events, artifacts, and predictions.
- Separate Simple and Advanced Pipeline Agent modes, with bounded tools and run lifecycle controls.
- Dataset profiling, target/task inference, leakage-aware preprocessing and evaluation, sampling, and memory safeguards.
- Independent LangGraph Report Agent with separate Groq configuration and a retained legacy report fallback.
- PostgreSQL persistence, Alembic migrations, and Celery/Redis background execution.

## Technology

- Backend: FastAPI, Pydantic, SQLAlchemy, PostgreSQL, Celery, and Redis.
- Pipeline Agent: LiteLLM and the configured provider.
- Report Agent: LangGraph, LiteLLM, and its separate provider settings.
- Data and ML: pandas, NumPy, SciPy, scikit-learn, and Optuna.
- PDF and notebook artifacts: Jinja2 and Playwright for PDF rendering; the core code generator creates notebook source.
- Authentication and notifications: JWT, Google OAuth code exchange, Passlib, Python-Jose, and Resend.

The backend uses the canonical `app.*` Python namespace. Configuration is in `app.config.settings`, the Pipeline Agent is in `app.agents.pipeline_agent`, and the Celery entry point is `app.worker.worker:celery_app`.

The frontend foundation and design-system preview are in [frontend](frontend/README.md). Full product pages are planned for later tasks. See [Architecture](docs/ARCHITECTURE.md), [API reference](docs/API.md), [Agent workflows](docs/AGENT.md), and [Setup](docs/SETUP.md).

## Docker Compose

The Compose file defines PostgreSQL and Redis alongside application, worker, and migration services. To start only local infrastructure, create .env only if it is absent, then run from the repository root:

    if (-not (Test-Path .env)) { Copy-Item .env.example .env }
    docker compose --env-file .env -f docker/docker-compose.yml up -d postgres redis

Stop only the infrastructure services while preserving their data with:

    docker compose --env-file .env -f docker/docker-compose.yml stop postgres redis

Do not use `down -v`; it removes the named database and Redis volumes. These commands do not start the API, worker, migrations, or frontend.

## Local backend setup

Use Python 3.11 or later. From the repository root, create .env only if one does not already exist:

    # PowerShell
    if (-not (Test-Path .env)) { Copy-Item .env.example .env }

Configure PostgreSQL and Redis URLs, CORS, and the separate Pipeline and Report Agent settings. Keep real credentials in .env; never copy example placeholders over an existing configuration.

    cd backend
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements-dev.txt
    cd ..
    alembic -c backend/alembic.ini upgrade head
    uvicorn app.main:app --app-dir backend --reload --port 8100

The API is available at http://localhost:8100/docs. Local runtime requires PostgreSQL and Redis. The automated test suite does not require live LLM endpoints.

Production/runtime deployments install `requirements.txt`; local development and testing use `requirements-dev.txt`, which includes the runtime pins. Backend and Compose values belong in the repository-root `.env` (create it from `.env.example` only when absent). The frontend reads only `NEXT_PUBLIC_API_BASE_URL` from `frontend/.env.local`, documented by `frontend/.env.example`.

For tests, checks, and migration adoption safety, follow [Setup](docs/SETUP.md).
