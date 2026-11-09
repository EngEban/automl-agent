from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException

from app.api.deps import DB, CurrentUser
from app.schemas.predict import PredictRequest, PredictResponse
from app.services import inference_service
from app.db.crud import get_model_version ,list_model_versions

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/{model_version_id}", response_model=PredictResponse)
def predict(
    model_version_id: str, 
    payload: PredictRequest, 
    db: DB, 
    current_user: CurrentUser
):
    """Run batch inference on a registered model."""
    try:
        return inference_service.run_inference(db, model_version_id, payload.records)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(500, f"Prediction failed: {exc}")

@router.get("/models/{session_id}", response_model=list[dict])
def list_models_for_session(session_id: str, db: DB, current_user: CurrentUser):
    """List all trained models for a specific session."""
    models = list_model_versions(db, session_id)
    return [
        {
            "id": m.id,
            "model_name": m.model_name,
            "version": m.version,
            "metrics": m.metrics_json,
            "status": m.lifecycle_status
        }
        for m in models
    ]    