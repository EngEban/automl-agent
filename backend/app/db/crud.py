"""
CRUD helpers. Keep thin; business logic belongs in services/agent.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.db.models import Artifact, Message, Operation, Session, User

# ---------- User ----------
def get_user_by_email(db: DBSession, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()

def create_user(db: DBSession, **kwargs) -> User:
    obj = User(**kwargs)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def update_user(db: DBSession, user_id: str, **updates) -> Optional[User]:
    obj = db.get(User, user_id)
    if obj is None:
        return None
    for key, value in updates.items():
        setattr(obj, key, value)
    db.commit()
    db.refresh(obj)
    return obj

# ---------- Session ----------
def create_session(db: DBSession, **kwargs) -> Session:
    obj = Session(**kwargs)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def get_session(db: DBSession, session_id: str) -> Optional[Session]:
    try:
        return db.get(Session, session_id)
    except Exception:
        # Catch invalid UUID format errors from PostgreSQL
        return None

def list_sessions(db: DBSession, user_id: str, limit: int = 50, offset: int = 0) -> list[Session]:
    stmt = (
        select(Session)
        .where(Session.user_id == user_id)
        .order_by(Session.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(stmt))

def update_session(db: DBSession, session_id: str, **updates) -> Optional[Session]:
    try:
        obj = db.get(Session, session_id)
        if obj is None:
            return None
        for key, value in updates.items():
            setattr(obj, key, value)
        db.commit()
        db.refresh(obj)
        return obj
    except Exception:
        db.rollback()
        return None

def delete_session(db: DBSession, session_id: str) -> bool:
    try:
        obj = db.get(Session, session_id)
        if obj is None:
            return False
        db.delete(obj)
        db.commit()
        return True
    except Exception:
        db.rollback()
        return False

# ---------- Message ----------
def create_message(db: DBSession, **kwargs) -> Message:
    obj = Message(**kwargs)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def list_messages(db: DBSession, session_id: str, limit: int = 200) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .limit(limit)
    )
    return list(db.scalars(stmt))

# ---------- Operation ----------
def create_operation(db: DBSession, **kwargs) -> Operation:
    obj = Operation(**kwargs)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def list_operations(db: DBSession, session_id: str) -> list[Operation]:
    stmt = (
        select(Operation)
        .where(Operation.session_id == session_id)
        .order_by(Operation.created_at.asc())
    )
    return list(db.scalars(stmt))

# ---------- Artifact ----------
def create_artifact(db: DBSession, **kwargs) -> Artifact:
    obj = Artifact(**kwargs)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def list_artifacts(db: DBSession, session_id: str) -> list[Artifact]:
    stmt = (
        select(Artifact)
        .where(Artifact.session_id == session_id)
        .order_by(Artifact.created_at.asc())
    )
    return list(db.scalars(stmt))

# ---------- AnalysisRun ----------
from app.db.models import AnalysisRun, RunEvent

def create_run(db: DBSession, **kwargs) -> AnalysisRun:
    obj = AnalysisRun(**kwargs)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def get_run(db: DBSession, run_id: str) -> Optional[AnalysisRun]:
    try:
        return db.get(AnalysisRun, run_id)
    except Exception:
        return None

def update_run(db: DBSession, run_id: str, **updates) -> Optional[AnalysisRun]:
    obj = db.get(AnalysisRun, run_id)
    if obj is None:
        return None
    for key, value in updates.items():
        setattr(obj, key, value)
    db.commit()
    db.refresh(obj)
    return obj

def list_runs_for_session(db: DBSession, session_id: str) -> list[AnalysisRun]:
    stmt = (
        select(AnalysisRun)
        .where(AnalysisRun.session_id == session_id)
        .order_by(AnalysisRun.created_at.desc())
    )
    return list(db.scalars(stmt))

# ---------- RunEvent ----------
def create_event(db: DBSession, **kwargs) -> RunEvent:
    obj = RunEvent(**kwargs)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def list_events_for_run(db: DBSession, run_id: str, limit: int = 100) -> list[RunEvent]:
    stmt = (
        select(RunEvent)
        .where(RunEvent.run_id == run_id)
        .order_by(RunEvent.created_at.asc())
        .limit(limit)
    )
    return list(db.scalars(stmt))