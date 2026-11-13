from __future__ import annotations

import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session as DBSession

from app.config import settings
from app.db.crud import get_model_version
from app.schemas.predict import PredictResponse

logger = logging.getLogger(__name__)


class InvalidPredictionInput(ValueError):
    """Raised when prediction records do not match the model input schema."""


def _validated_artifact_path(artifact_uri: str) -> Path:
    """Resolve a model artifact and require it to be a regular file in storage."""
    try:
        storage_root = Path(settings.STORAGE_DIR).resolve(strict=True)
        artifact_path = Path(artifact_uri).resolve(strict=True)
        artifact_path.relative_to(storage_root)
        if not artifact_path.is_file():
            raise ValueError("Model artifact is not a regular file.")
        return artifact_path
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Rejected unavailable or unsafe model artifact path", exc_info=True)
        raise RuntimeError("Model artifact is unavailable.") from exc


def run_inference(db: DBSession, model_version_id: str, records: list[dict]) -> PredictResponse:
    """Load the fitted pipeline and predict on new records."""
    model_version = get_model_version(db, model_version_id)
    if not model_version:
        raise LookupError("Model version not found.")

    # Resolve and validate the path (including symlinks) before loading.
    try:
        artifact_path = _validated_artifact_path(model_version.artifact_uri)
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        logger.exception("Failed to load artifact for model version %s", model_version_id)
        raise RuntimeError("Model artifact could not be loaded.") from exc

    target_encoder = None
    if isinstance(artifact, dict) and "artifact_format_version" in artifact:
        if artifact.get("artifact_format_version") != 1:
            raise RuntimeError("Model artifact format is not supported.")
        pipeline = artifact.get("pipeline")
        target_encoder = artifact.get("target_encoder")
        input_columns = artifact.get("input_columns")
        if not isinstance(input_columns, (list, tuple)) or not all(
            isinstance(column, str) for column in input_columns
        ):
            raise RuntimeError("Model artifact input schema is invalid.")
        expected_cols = list(input_columns)
    else:
        # Legacy registry artifacts contain the fitted pipeline directly.
        pipeline = artifact
        expected_cols = list(model_version.schema_json.keys())

    if pipeline is None:
        raise RuntimeError("Model artifact does not contain a fitted pipeline.")

    try:
        available_cols = set()
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("Each prediction record must be an object.")
            available_cols.update(record.keys())
    except (TypeError, ValueError) as exc:
        raise InvalidPredictionInput("Prediction records are invalid.") from exc

    missing_cols = [c for c in expected_cols if c not in available_cols]
    if missing_cols:
        raise InvalidPredictionInput(
            f"Missing required columns for prediction: {missing_cols}"
        )

    batch_size = int(settings.PREDICTION_BATCH_SIZE)
    batch_count = math.ceil(len(records) / batch_size) if records else 0
    logger.info(
        "Processing %d prediction rows in %d batches (batch size %d)",
        len(records),
        batch_count,
        batch_size,
    )
    predictions: list = []
    probabilities: list[list[float]] | None = None
    model = getattr(pipeline, "named_steps", {}).get("model")
    has_probabilities = callable(getattr(model, "predict_proba", None))
    if has_probabilities:
        probabilities = []

    try:
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            batch = pd.DataFrame(batch_records).reindex(columns=expected_cols)
            preds, probs = _predict_batch(pipeline, batch, model)
            if target_encoder is not None:
                try:
                    preds = target_encoder.inverse_transform(np.asarray(preds))
                except Exception as exc:
                    logger.exception(
                        "Model returned invalid encoded targets for model version %s",
                        model_version_id,
                    )
                    raise RuntimeError("Model produced invalid encoded predictions.") from exc
            predictions.extend(
                value.item() if isinstance(value, np.generic) else value
                for value in preds
            )
            if probabilities is not None and probs is not None:
                probabilities.extend(np.asarray(probs).tolist())
    except RuntimeError as exc:
        if str(exc) == "Model produced invalid encoded predictions.":
            raise
        logger.exception("Inference failed for model version %s", model_version_id)
        raise RuntimeError("Inference failed.") from exc
    except ValueError as exc:
        logger.info("Prediction input rejected for model version %s: %s", model_version_id, exc)
        raise InvalidPredictionInput(
            "Prediction records are incompatible with the model input schema."
        ) from exc
    except Exception as exc:
        logger.exception("Inference failed for model version %s", model_version_id)
        raise RuntimeError("Inference failed.") from exc

    try:
        return PredictResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_version_id=model_version.id,
            model_name=model_version.model_name,
        )
    except ValueError as exc:
        logger.info("Prediction input rejected for model version %s: %s", model_version_id, exc)
        raise InvalidPredictionInput(
            "Prediction records are incompatible with the model input schema."
        ) from exc
    except Exception as exc:
        logger.exception("Inference failed for model version %s", model_version_id)
        raise RuntimeError("Inference failed.") from exc


def _predict_batch(pipeline, batch: pd.DataFrame, model):
    """Transform once and reuse that representation for predictions and scores."""
    steps = getattr(pipeline, "steps", None)
    if steps and len(steps) > 1 and steps[-1][0] == "model":
        transformed = pipeline[:-1].transform(batch)
        predictions = model.predict(transformed)
        probabilities = (
            model.predict_proba(transformed)
            if callable(getattr(model, "predict_proba", None))
            else None
        )
        return predictions, probabilities

    predictions = pipeline.predict(batch)
    if not callable(getattr(model, "predict_proba", None)):
        return predictions, None
    pipeline_predict_proba = getattr(pipeline, "predict_proba", None)
    if callable(pipeline_predict_proba):
        probabilities = pipeline_predict_proba(batch)
    else:
        probabilities = model.predict_proba(batch)
    return predictions, probabilities
