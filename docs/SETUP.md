# Local setup and verification

## Requirements

- Python 3.11 or later.
- PostgreSQL and Redis for the full API and worker runtime.
- An Ollama endpoint or another supported LiteLLM provider for live Pipeline Agent requests.
- Groq credentials only if live Report Agent generation is enabled.

The backend can be imported and its automated tests can run without PostgreSQL, Redis, Kaggle credentials, or live LLM requests. Runtime dependencies are pinned in `backend/requirements.txt`; test and development dependencies are in `backend/requirements-dev.txt`, which includes the runtime file.

## Environment

Create .env from .env.example only if .env does not already exist. Preserve existing .env files and never put real credentials in .env.example.

PowerShell:

    if (-not (Test-Path .env)) { Copy-Item .env.example .env }

Configure the host-side DATABASE_URL, optional Compose DATABASE_URL_DOCKER, POSTGRES_* values, Redis URLs, CORS origins, and the two separate agent configurations. REPORT_AGENT_ENABLED defaults to false; provide a valid Report Agent key before enabling live Groq generation. The Next.js frontend has its own non-secret NEXT_PUBLIC_API_BASE_URL setting in frontend/.env.example.

Percent-encode URL-reserved characters in database passwords inside both database URLs; for example, encode @ as %40.

## User-managed provider credentials

Provider profiles store credentials only as Fernet authenticated ciphertext. Set PROVIDER_CREDENTIAL_ENCRYPTION_KEY in the backend-only root .env; never put it in a frontend environment file, commit it, reuse SECRET_KEY, or store it in the database. The application does not require this key at startup. Credential create/replace operations fail closed until a valid key is configured.

Generate a Fernet key in a trusted local shell and copy it directly into the backend .env:

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

The example value in .env.example is deliberately not a valid key. The API exposes only whether a credential is configured; it never returns saved secrets or ciphertext. Provider connection checks are explicit user-triggered requests and use short synthetic prompts without datasets or chat history.

Credential ciphertext is versioned (enc:v1:). Do not replace the active key while v1 credentials exist: they would become unreadable. A future rotation must add a key ring that can decrypt the old version and encrypt a new version, re-encrypt each row in a controlled transaction or resumable batch, verify all rows, and only then retire the old key. Back up the encrypted database and old key securely before rotation; never log or expose either key.

## Docker Compose development

The existing Compose file also defines application, worker, and migration services. For local infrastructure checks, select only PostgreSQL and Redis so no application containers or migrations start. From the repository root, create .env only when it is absent, then run:

    docker compose --env-file .env -f docker/docker-compose.yml up -d postgres redis

Stop only those services without removing their named data volumes with:

    docker compose --env-file .env -f docker/docker-compose.yml stop postgres redis

Do not use `down -v`; it deletes the persistent PostgreSQL and Redis volumes. The service ports are bound to localhost only.

## Python environment

From the repository root:

    cd backend
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements-dev.txt

Production/runtime installations use `python -m pip install -r requirements.txt` instead. Docker runtime images also install only `requirements.txt`.

Start the local API after configuring PostgreSQL and Redis:

    alembic -c backend/alembic.ini upgrade head
    uvicorn app.main:app --app-dir backend --reload --port 8100

Open /docs for interactive OpenAPI documentation. The Celery application object is `app.worker.worker:celery_app`. Start a local worker from the backend directory with `celery -A app.worker.worker:celery_app worker --pool=solo --loglevel=info` on Windows.

## Migration history

Fresh databases are migrated through alembic upgrade head. The current model set contains 11 tables. The revision chain is:

1. 0001_initial_schema: original users, sessions, messages, operations, and artifacts.
2. 0002_analysis_runs: analysis runs and run events.
3. b3ee8c7dbb01_add_model_registry_tables: experiments and model versions.
4. 18cd42a0a3f0_add_dataset_sources_table: dataset sources and analysis plans.
5. f7a4c12d8e90_run_event_order_and_idempotency: stable event pagination and run idempotency changes.

For a database created by the pre-Alembic application:

1. Back it up and confirm the backup can be restored.
2. Compare the original five tables against 0001_initial_schema, including column types, nullability, indexes, foreign keys, and unique constraints.
3. Only after that verification, run the stamp and upgrade commands from the repository root:

       alembic -c backend/alembic.ini stamp 0001_initial_schema
       alembic -c backend/alembic.ini upgrade head

The upgrade applies the remaining six tables and later indexes/columns. Never stamp, drop, or reset an unverified database.

## Automated tests and checks

Run these from the backend directory. The tests use mocks and local test databases; they do not make live Ollama or Groq requests.

    $env:LITELLM_LOCAL_MODEL_COST_MAP = 'True'
    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe -m compileall -q app tests scripts
    .\.venv\Scripts\python.exe -m pip check

The live diagnostics in scripts/smoke_test_ollama_pipeline.py and scripts/smoke_test_report_agent.py are separate from pytest. Run them only when a live provider check is intended; the Report Agent script can make Groq requests.

## Runtime data

Do not remove or reset backend/storage, PostgreSQL volumes, user datasets, reports, figures, session caches, or model artifacts as part of routine setup or verification.
