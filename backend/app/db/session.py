"""
SQLAlchemy engine and session factory.
"""
from __future__ import annotations

import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    future=True,
    echo=False,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db():
    """FastAPI dependency: yields a DB session, closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Create all tables. Idempotent (CREATE TABLE IF NOT EXISTS).
    Raises if the database is unreachable — caller decides how to handle it.
    """
    from app.db.base import Base
    from app.db import models  # noqa: F401 - register models in metadata

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured.")