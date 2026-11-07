"""FastAPI routes."""
from __future__ import annotations

import uuid
from pathlib import Path

from celery.result import AsyncResult
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.schemas import AnalyzeURLRequest, TaskResponse, TaskStatus
from app.tasks.analyze import analyze_dataset
from app.worker import celery_app

router = APIRouter()


@router.post("/analyze/upload", response_model=TaskResponse)
async def analyze_upload(
    file: UploadFile = File(...),
    target_column: str | None = Form(None),
):
    """Upload a file and enqueue analysis."""
    task_id = str(uuid.uuid4())
    ext = Path(file.filename or "data.csv").suffix.lower()
    if ext not in {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls"}:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    dest = settings.UPLOAD_DIR / f"{task_id}{ext}"
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    analyze_dataset.apply_async(
        args=["upload", str(dest), target_column],
        task_id=task_id,
    )
    return TaskResponse(task_id=task_id)


@router.post("/analyze/url", response_model=TaskResponse)
async def analyze_url(req: AnalyzeURLRequest):
    """Provide a URL and enqueue analysis."""
    task_id = str(uuid.uuid4())
    analyze_dataset.apply_async(
        args=["url", req.url, req.target_column],
        task_id=task_id,
    )
    return TaskResponse(task_id=task_id)


@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    state = result.state
    meta = result.info if isinstance(result.info, dict) else {}

    if state == "PENDING":
        return TaskStatus(task_id=task_id, status="queued")
    if state in ("PROGRESS", "STARTED"):
        return TaskStatus(
            task_id=task_id, status="running",
            step=meta.get("step"), progress=meta.get("progress"),
        )
    if state == "SUCCESS":
        return TaskStatus(task_id=task_id, status="done", result=meta)
    return TaskStatus(task_id=task_id, status="failed", error=str(result.info))


@router.get("/download/report/{task_id}")
async def download_report(task_id: str):
    path = settings.REPORT_DIR / f"{task_id}.pdf"
    if not path.exists():
        raise HTTPException(404, "Report not found")
    return FileResponse(str(path), media_type="application/pdf",
                        filename=f"report_{task_id}.pdf")


@router.get("/download/model/{task_id}")
async def download_model(task_id: str):
    path = settings.MODEL_DIR / f"{task_id}.pkl"
    if not path.exists():
        raise HTTPException(404, "Model not found")
    return FileResponse(str(path), media_type="application/octet-stream",
                        filename=f"model_{task_id}.pkl")