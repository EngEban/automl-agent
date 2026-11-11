from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.api.deps import CurrentUser, DB
from app.db.crud import (
    get_experiment,
    get_model_version,
    get_session,
    list_model_versions,
)
from app.schemas.predict import PredictRequest, PredictResponse
from app.services import inference_service

logger = logging.getLogger(__name__)
router = APIRouter()


def _owned_session_or_404(db: DB, session_id: str, user_id: str):
    session = get_session(db, session_id)
    if session is None or str(session.user_id) != str(user_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@router.post("/{model_version_id}", response_model=PredictResponse)
def predict(
    model_version_id: str,
    payload: PredictRequest,
    db: DB,
    current_user: CurrentUser,
) -> PredictResponse:
    """Run batch inference only for a model owned by the authenticated user."""
    model_version = get_model_version(db, model_version_id)
    if model_version is None:
        raise HTTPException(status_code=404, detail="Model version not found.")

    experiment = get_experiment(db, model_version.experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Model version not found.")

    try:
        _owned_session_or_404(db, experiment.session_id, current_user.id)
    except HTTPException as exc:
        # Keep this response identical to an absent model version so the API
        # does not reveal whether another user's model identifier is valid.
        raise HTTPException(status_code=404, detail="Model version not found.") from exc

    try:
        return inference_service.run_inference(
            db,
            model_version_id,
            payload.records,
        )
    except inference_service.InvalidPredictionInput as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Model version not found.") from exc
    except Exception as exc:
        logger.exception("Prediction failed for model version %s", model_version_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error.",
        ) from exc


@router.get("/models/{session_id}", response_model=list[dict])
def list_models_for_session(
    session_id: str,
    db: DB,
    current_user: CurrentUser,
) -> list[dict]:
    """List registered models only for a session owned by the user."""
    _owned_session_or_404(db, session_id, current_user.id)
    models = list_model_versions(db, session_id)
    return [
        {
            "id": model.id,
            "model_name": model.model_name,
            "version": model.version,
            "metrics": model.metrics_json,
            "status": model.lifecycle_status,
        }
        for model in models
    ]
