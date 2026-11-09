from __future__ import annotations
import logging
import joblib
import pandas as pd
from sqlalchemy.orm import Session as DBSession

from app.db.crud import get_model_version
from app.schemas.predict import PredictResponse

logger = logging.getLogger(__name__)

def run_inference(db: DBSession, model_version_id: str, records: list[dict]) -> PredictResponse:
    """Load the fitted pipeline and predict on new records."""
    model_version = get_model_version(db, model_version_id)
    if not model_version:
        raise ValueError("Model version not found.")

    # 1. Load the pipeline (preprocessor + model)
    try:
        pipeline = joblib.load(model_version.artifact_uri)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifact: {exc}")

    # 2. Convert input records to DataFrame
    df = pd.DataFrame(records)

    # 3. Validate columns (basic check)
    expected_cols = list(model_version.schema_json.keys())
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for inference: {missing_cols}")

    # 4. Predict
    try:
        preds = pipeline.predict(df[expected_cols])
        probs = None
        
        # Try to get probabilities if it's a classifier
        if hasattr(pipeline.named_steps['model'], "predict_proba"):
            probs = pipeline.predict_proba(df[expected_cols]).tolist()
            
        # Convert numpy types to native Python types
        predictions = [p.item() if hasattr(p, "item") else p for p in preds]
        
        return PredictResponse(
            predictions=predictions,
            probabilities=probs,
            model_version_id=model_version.id,
            model_name=model_version.model_name
        )
    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}")