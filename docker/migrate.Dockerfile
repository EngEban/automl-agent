# Minimal image for explicit Alembic migrations; runtime services use backend.Dockerfile.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install alembic==1.13.2 sqlalchemy==2.0.35 \
        psycopg2-binary==2.9.9 pydantic-settings==2.14.1

COPY backend/alembic.ini ./alembic.ini
COPY backend/alembic/ ./alembic/
COPY backend/app/__init__.py ./app/__init__.py
COPY backend/app/config.py ./app/config.py
COPY backend/app/db/ ./app/db/
