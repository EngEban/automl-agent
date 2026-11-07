"""Authenticated endpoints for listing and downloading session artifacts."""
from __future__ import annotations

import logging
import mimetypes
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.deps import CurrentUser, DB
from app.config import settings
from app.db import crud

logger = logging.getLogger(__name__)
router = APIRouter()


def _owned_session(session_id: str, db: DB, current_user: CurrentUser):
    """Return a session only when it belongs to the authenticated user."""
    session = crud.get_session(db, session_id)
    if session is None or session.user_id != current_user.id:
        # Do not disclose whether another user's session exists.
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _stored_file_path(raw_path: str, artifact_label: str) -> Path:
    """Validate that a stored artifact is a regular file under storage."""
    try:
        storage_root = settings.STORAGE_DIR.resolve()
        path = Path(raw_path).resolve()
        path.relative_to(storage_root)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Rejected invalid %s path: %s", artifact_label, raw_path)
        raise HTTPException(
            status_code=404,
            detail=f"{artifact_label} file is not available",
        ) from exc

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"{artifact_label} file is missing on disk",
        )
    return path


def _latest_artifact_path(
    session_id: str,
    db: DB,
    artifact_type: str,
    artifact_label: str,
) -> Path:
    """Find and validate the newest artifact of a given type."""
    rows = [
        artifact
        for artifact in crud.list_artifacts(db, session_id)
        if artifact.artifact_type == artifact_type
    ]
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"{artifact_label} has not been generated for this session",
        )
    return _stored_file_path(rows[-1].file_path, artifact_label)


def _download_filename(raw_name: object, fallback: str) -> str:
    """Return a safe basename for the Content-Disposition filename."""
    name = str(raw_name or "").replace("\\", "/")
    name = re.sub(r"[\x00-\x1f\x7f]", "", Path(name).name)
    return name if name not in {"", ".", ".."} else fallback


class ArtifactOut(BaseModel):
    id: str
    session_id: str
    artifact_type: str
    file_path: str

    model_config = {"from_attributes": True}


@router.get("/{session_id}", response_model=list[ArtifactOut])
def list_artifacts(
    session_id: str,
    db: DB,
    current_user: CurrentUser,
) -> list[ArtifactOut]:
    _owned_session(session_id, db, current_user)
    rows = crud.list_artifacts(db, session_id)
    return [ArtifactOut.model_validate(a) for a in rows]


@router.get("/{session_id}/report")
def download_report(
    session_id: str,
    db: DB,
    current_user: CurrentUser,
) -> FileResponse:
    """Download the PDF report of a session."""
    _owned_session(session_id, db, current_user)
    path = _latest_artifact_path(
        session_id,
        db,
        artifact_type="report",
        artifact_label="Report",
    )
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"report_{session_id}.pdf",
    )


@router.get("/{session_id}/notebook")
def download_notebook(
    session_id: str,
    db: DB,
    current_user: CurrentUser,
) -> FileResponse:
    """Download the generated Jupyter notebook of a session."""
    _owned_session(session_id, db, current_user)
    path = _latest_artifact_path(
        session_id,
        db,
        artifact_type="notebook",
        artifact_label="Notebook",
    )
    return FileResponse(
        path,
        media_type="application/x-ipynb+json",
        filename="analysis_pipeline.ipynb",
    )


@router.get("/{session_id}/dataset")
def download_dataset(
    session_id: str,
    db: DB,
    current_user: CurrentUser,
) -> FileResponse:
    """Download the original uploaded dataset for a session."""
    session = _owned_session(session_id, db, current_user)
    if session.source_type != "upload":
        raise HTTPException(
            status_code=404,
            detail="Original uploaded dataset is not available for this session",
        )

    path = _stored_file_path(session.source_path, "Dataset")
    metadata = session.dataset_meta if isinstance(session.dataset_meta, dict) else {}
    original_filename = metadata.get("original_filename") or metadata.get("filename")
    filename = _download_filename(
        original_filename,
        fallback=path.name or f"dataset_{session_id}.csv",
    )
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)


@router.get("/{session_id}/cleaned-dataset")
def download_cleaned_dataset(
    session_id: str,
    db: DB,
    current_user: CurrentUser,
) -> FileResponse:
    """Download the imputed, unscaled dataset for an owned session."""
    _owned_session(session_id, db, current_user)
    path = _latest_artifact_path(
        session_id,
        db,
        artifact_type="cleaned_data",
        artifact_label="Cleaned dataset",
    )
    return FileResponse(
        path,
        media_type="text/csv",
        filename="cleaned_data.csv",
    )
