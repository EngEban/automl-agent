# AutoML Agent - Architecture

## Overview
AutoML Agent is designed using a modular, scalable architecture. The system separates the core Machine Learning pipeline from the AI Agent and the API layer, ensuring loose coupling and high maintainability.

## Core Components

1. **API Layer (FastAPI)**:
   - Handles HTTP requests, CORS, and routing.
   - Manages authentication (JWT, Google OAuth) and authorization.
   - Interacts with the database using SQLAlchemy.

2. **AI Agent (`app/agent/`)**:
   - The "brain" of the application. It uses **LiteLLM** to communicate with the LLM (e.g., Ollama/Qwen).
   - Maintains conversation memory and context window.
   - Decides which tools to call based on the user's prompt.
   - Parses tool calls and passes them to the Executor.

3. **Core ML Pipeline (`app/core/`)**:
   - Pure Python/Scikit-learn logic. 
   - Contains modules for Data Loading, EDA, Preprocessing, Training, and Report Generation.
   - It knows nothing about the LLM or HTTP; it simply takes a DataFrame and returns results.

4. **Database (`app/db/`)**:
   - PostgreSQL managed via SQLAlchemy and Alembic.
   - Stores `Users`, `Sessions`, `Messages`, `Operations`, `Artifacts`,
     `AnalysisRuns`, and `RunEvents`.
   - The application does not call `Base.metadata.create_all()` at startup;
     schema creation and updates are explicit Alembic migrations.

5. **Notifications & Storage**:
   - Generates PDFs using WeasyPrint and Jupyter Notebooks using `nbformat`.
   - Stores artifacts locally and notifies users via Email (Resend) and Web Push.

## Database infrastructure

The default Docker Compose stack contains four long-running application
services plus a one-shot migration service:

```text
PostgreSQL 16 (postgres:5432) ──> migrate (alembic upgrade head)
          │                                  │
          └───────────────┬──────────────────┘
                          ▼
                 FastAPI backend + Celery worker

Redis 7 (redis:6379) ────────> backend and worker
```

PostgreSQL and Redis use health checks. The migration service waits for a
healthy PostgreSQL server, and the backend and worker wait for the migration
service to complete successfully. PostgreSQL and Redis use named persistent
volumes; backend-generated files remain in `backend/storage`.

## Migration history and safety

- `0001_initial_schema` reconstructs the original five-table schema that was
  previously created by SQLAlchemy `create_all()`.
- `0002_analysis_runs` adds `analysis_runs` and `run_events`, including their
  foreign keys and indexes.

Fresh installations use:

```bash
alembic -c backend/alembic.ini upgrade head
```

For a database created by the old application, back it up and verify the
five-table schema against revision `0001_initial_schema` before recording the
baseline with `alembic -c backend/alembic.ini stamp 0001_initial_schema`.
Only then run `alembic -c backend/alembic.ini upgrade head`. This adoption
procedure is manual by design and never runs automatically.

Inspect the applied revision with:

```bash
alembic -c backend/alembic.ini current
```

## Data Flow
1. The user uploads a dataset via the API.
2. The agent receives a prompt and analyzes the dataset schema.
3. The agent decides on a sequence of tools (e.g., EDA -> Preprocessing -> Training).
4. The `executor.py` triggers the Core ML Pipeline.
5. The pipeline saves artifacts to the local storage and updates the database.
6. The agent returns the final response to the user.
