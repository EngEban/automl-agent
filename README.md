# 🤖 AutoML Agent

An intelligent, autonomous AI Agent that automates the entire Data Science lifecycle. Upload a dataset, and the agent will perform EDA, clean the data, train multiple models, select the best one, and generate a professional PDF report along with a reproducible Jupyter Notebook.

## ✨ Features
- **Autonomous Agent:** Powered by LLMs (via Ollama, OpenAI, or Groq) using function calling to decide the best analysis path.
- **Smart Preprocessing:** Automatically handles missing values, categorical encoding, and feature scaling.
- **AutoML Pipeline:** Trains and compares multiple models (Random Forest, Gradient Boosting, etc.).
- **Rich Reporting:** Generates a polished PDF report with insights, charts, and model comparisons.
- **Reproducible Code:** Generates a ready-to-run `.ipynb` Jupyter Notebook.
- **Secure Authentication:** JWT-based authentication with Google OAuth and Email Verification.
- **Notifications:** Real-time Web Push notifications and Email alerts upon completion.

## 🛠️ Tech Stack
- **Backend:** FastAPI, Python 3.11, SQLAlchemy, PostgreSQL.
- **AI/Agent:** LiteLLM, Ollama (Local LLM), LangChain concepts.
- **Machine Learning:** Pandas, Scikit-learn, Matplotlib, Seaborn.
- **Authentication:** JWT, Passlib, Python-Jose, Google OAuth2.
- **Reporting:** WeasyPrint (PDF), nbformat (Jupyter).
- **Frontend:** React, Vite, Tailwind CSS *(Coming soon)*.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker Desktop with Compose v2
- Ollama (for a local LLM) or an API key for Groq/OpenAI

### Docker Compose

The default Compose stack contains PostgreSQL, Redis, the FastAPI backend, a
Celery worker, and a one-shot migration service. Frontend services are not part
of the default stack yet.

```bash
copy .env.example .env  # Windows
# cp .env.example .env  # macOS/Linux; then edit the values
docker compose --env-file .env -f docker/docker-compose.yml up --build
```

The migration service waits for PostgreSQL to become healthy. The backend and
worker start only after `alembic upgrade head` succeeds. PostgreSQL data is
stored in the `postgres_data` volume, Redis data in `redis_data`, and backend
artifacts in `backend/storage`.

The API is available at http://localhost:8100/docs.

### Local backend development

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

Copy `.env.example` to `.env`, start PostgreSQL and Redis, and use the host
URLs in `.env`. From the repository root, run the migrations and application:

```bash
alembic -c backend/alembic.ini upgrade head
uvicorn app.main:app --app-dir backend --reload --port 8100
```

### Alembic commands

```bash
# Apply all migrations to a fresh database
alembic -c backend/alembic.ini upgrade head

# Inspect the current database revision
alembic -c backend/alembic.ini current

# List the migration history
alembic -c backend/alembic.ini history
```

Migration `0001_initial_schema` is the original five-table baseline. Migration
`0002_analysis_runs` adds `analysis_runs` and `run_events`.

#### Existing databases created by the old application

Do not stamp or reset an existing database automatically. First take a backup,
then verify that its `users`, `sessions`, `messages`, `operations`, and
`artifacts` tables match `0001_initial_schema` (columns, types, nullability,
foreign keys, indexes, and unique constraints). Only after that verification:

```bash
alembic -c backend/alembic.ini stamp 0001_initial_schema
alembic -c backend/alembic.ini upgrade head
```

The stamp records the verified baseline without changing application tables;
the upgrade then applies only the new run lifecycle tables.
