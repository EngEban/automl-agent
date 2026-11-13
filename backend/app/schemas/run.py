from __future__ import annotations
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel

class AnalysisRunCreate(BaseModel):
    session_id: str

class AnalysisRunRead(BaseModel):
    id: str
    session_id: str
    status: str
    current_step: Optional[str]
    progress_percent: int
    message: Optional[str]
    error_message: Optional[str]
    error_code: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    model_config = {"from_attributes": True}

class RunEventRead(BaseModel):
    id: str
    run_id: str
    sequence: int
    event_type: str
    step: Optional[str]
    progress_percent: int
    message: Optional[str]
    payload: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}
