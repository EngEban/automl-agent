"""Containment checks shared by authenticated artifact download routes."""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException


def resolve_artifact_path(
    raw_path: str,
    *,
    artifact_label: str,
    permitted_root: Path,
) -> Path:
    """Resolve a stored file and require it to remain below its allowed root."""
    try:
        root = Path(permitted_root).resolve()
        resolved = Path(raw_path).resolve()
        resolved.relative_to(root)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"{artifact_label} file is not available",
        ) from exc

    if not resolved.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"{artifact_label} file is missing",
        )
    return resolved
