"""
SQLAlchemy ORM models.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON, Boolean, DateTime, ForeignKey, Index, String, Text, func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

def _uuid() -> str:
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    google_id: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # حقل إشعارات المتصفح (Web Push)
    push_subscription: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    sessions: Mapped[list["Session"]] = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    analysis_runs: Mapped[list["AnalysisRun"]] = relationship("AnalysisRun", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User email={self.email!r}>"

class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    target_column: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    task_type: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    dataset_meta: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="sessions")
    messages: Mapped[list["Message"]] = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    operations: Mapped[list["Operation"]] = relationship("Operation", back_populates="session", cascade="all, delete-orphan")
    artifacts: Mapped[list["Artifact"]] = relationship("Artifact", back_populates="session", cascade="all, delete-orphan")
    analysis_runs: Mapped[list["AnalysisRun"]] = relationship("AnalysisRun", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Session id={self.id} name={self.name!r} status={self.status}>"

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tool_calls: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    tool_results: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session: Mapped["Session"] = relationship("Session", back_populates="messages")

class Operation(Base):
    __tablename__ = "operations"
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    op_type: Mapped[str] = mapped_column(String(50), nullable=False)
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="applied", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session: Mapped["Session"] = relationship("Session", back_populates="operations")

class Artifact(Base):
    __tablename__ = "artifacts"
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    artifact_type: Mapped[str] = mapped_column(String(50), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session: Mapped["Session"] = relationship("Session", back_populates="artifacts")

# ==========================================
# NEW MODELS FOR ASYNCHRONOUS JOB LIFECYCLE
# ==========================================

class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (
        Index(
            "uq_analysis_runs_owner_idempotency",
            "requested_by",
            "idempotency_key",
            unique=True,
        ),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    requested_by: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Status: queued, running, awaiting_approval, succeeded, failed, cancelled
    status: Mapped[str] = mapped_column(String(50), default="queued", nullable=False, index=True)
    current_step: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    progress_percent: Mapped[int] = mapped_column(default=0, nullable=False)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    error_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    session: Mapped["Session"] = relationship("Session", back_populates="analysis_runs")
    user: Mapped["User"] = relationship("User", back_populates="analysis_runs")
    events: Mapped[list["RunEvent"]] = relationship("RunEvent", back_populates="run", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<AnalysisRun id={self.id} status={self.status}>"


class RunEvent(Base):
    __tablename__ = "run_events"
    __table_args__ = (
        Index("uq_run_events_run_sequence", "run_id", "sequence", unique=True),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    sequence: Mapped[int] = mapped_column(nullable=False)
    
    # event_type: run.started, step.started, step.progress, run.succeeded, run.failed, run.cancelled
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    step: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    progress_percent: Mapped[int] = mapped_column(default=0, nullable=False)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    run: Mapped["AnalysisRun"] = relationship("AnalysisRun", back_populates="events")

    def __repr__(self) -> str:
        return f"<RunEvent id={self.id} type={self.event_type}>"

class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    run: Mapped["AnalysisRun"] = relationship("AnalysisRun")
    session: Mapped["Session"] = relationship("Session")
    model_versions: Mapped[list["ModelVersion"]] = relationship("ModelVersion", back_populates="experiment", cascade="all, delete-orphan")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    
    version: Mapped[int] = mapped_column(default=1, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False) # Path to .pkl
    schema_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False) # Expected input columns
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False) # R2, F1, etc.
    parameters_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False) # Optuna params
    lifecycle_status: Mapped[str] = mapped_column(String(20), default="candidate", nullable=False) # candidate, production, archived
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="model_versions")

class AnalysisPlan(Base):
    __tablename__ = "analysis_plans"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    version: Mapped[int] = mapped_column(default=1, nullable=False)
    plan_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    confidence: Mapped[float] = mapped_column(default=1.0, nullable=False)
    approval_status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False) # pending, approved, rejected
    approved_by: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=False), nullable=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    run: Mapped["AnalysisRun"] = relationship("AnalysisRun")

class DatasetSource(Base):
    __tablename__ = "dataset_sources"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False) # e.g., kaggle, uci, huggingface, local
    source_id: Mapped[str] = mapped_column(String(255), nullable=False) # The identifier or URL
    revision: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    license: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    checksum: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    download_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session: Mapped["Session"] = relationship("Session")
