# Local Setup Guide

## Prerequisites

- Python 3.11+
- Docker Desktop with Compose v2
- Ollama (for a local LLM) or an API key for Groq/OpenAI

## Recommended setup: Docker Compose

From the repository root:

```powershell
Copy-Item .env.example .env
# Edit .env before starting if you need non-default credentials or services.
docker compose --env-file .env -f docker/docker-compose.yml up --build
```

The stack starts PostgreSQL 16, Redis 7, the FastAPI backend, the Celery
worker, and a one-shot Alembic migration service. PostgreSQL and Redis are
reachable from containers as `postgres:5432` and `redis:6379`; host tools use
`localhost:5433` and `localhost:6479`.

The migration service runs `alembic upgrade head` after PostgreSQL is healthy.
The backend and worker are held until that service exits successfully. No
frontend service is included in the default Compose file.

To stop the stack without deleting database data:

```powershell
docker compose --env-file .env -f docker/docker-compose.yml down
```

Do not use `docker compose down -v` unless you intentionally want to delete
the named PostgreSQL and Redis volumes.

## Environment variables

Copy `.env.example` to `.env` and configure at least:

- `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`
- `DATABASE_URL` for host-side commands
- `DATABASE_URL_DOCKER` for services inside Compose
- LLM settings appropriate for your provider
- OAuth and email settings if those features are enabled

Database passwords containing URL-reserved characters must be percent-encoded
inside both database URLs. For example, `@` becomes `%40`.

## Local Python development

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
```

With PostgreSQL and Redis running and the host URLs configured in `.env`:

```powershell
alembic -c backend/alembic.ini upgrade head
uvicorn app.main:app --app-dir backend --reload --port 8100
```

The API is available at http://localhost:8100/docs.

## Database migrations

Run Alembic from the repository root:

```powershell
alembic -c backend/alembic.ini upgrade head
alembic -c backend/alembic.ini current
alembic -c backend/alembic.ini history
```

For a fresh database, `upgrade head` creates all seven application tables.
The revisions are:

- `0001_initial_schema`: `users`, `sessions`, `messages`, `operations`, and `artifacts`
- `0002_analysis_runs`: `analysis_runs` and `run_events`

### Safely adopting an existing database

The old application could have created the first five tables with
`Base.metadata.create_all()`. Never run `stamp` against an unverified database.

1. Back up the database and confirm the backup can be restored.
2. Inspect the existing five-table schema and compare it to
   `backend/alembic/versions/0001_initial_schema.py`: column types,
   nullability, indexes, foreign keys, and unique constraints must match.
3. On the verified database only, record the baseline and apply the new
   migration:

   ```powershell
   alembic -c backend/alembic.ini stamp 0001_initial_schema
   alembic -c backend/alembic.ini upgrade head
   ```

The stamp does not alter application tables. `upgrade head` then creates only
the two analysis-run tables. If the schema does not match, stop and reconcile
it manually; do not drop tables or records.
