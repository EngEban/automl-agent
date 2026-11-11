from __future__ import annotations
import logging
import zipfile
from pathlib import Path
from typing import Any

from app.core.connectors.base import DataSourceConnector

logger = logging.getLogger(__name__)


class KaggleConfigurationError(RuntimeError):
    """Raised when the Kaggle API cannot be configured for an operation."""


class KaggleConnector(DataSourceConnector):
    def __init__(self):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            self.api = KaggleApi()
            self.api.authenticate()
        except OSError as exc:
            raise KaggleConfigurationError(
                "Kaggle credentials are not configured. Configure a Kaggle API "
                "token by placing kaggle.json in ~/.kaggle or setting the "
                "KAGGLE_USERNAME and KAGGLE_KEY environment variables."
            ) from exc

    def validate_reference(self, reference: str) -> bool:
        # Kaggle references look like: "owner/dataset-name"
        return "/" in reference and not reference.startswith("http")

    def download_to_staging(self, reference: str, destination: Path) -> dict[str, Any]:
        logger.info(f"Downloading Kaggle dataset: {reference}")
        destination.mkdir(parents=True, exist_ok=True)
        
        # Download to a zip file
        self.api.dataset_download_files(reference, path=str(destination), unzip=False)
        
        # Find the downloaded zip file
        zip_path = next(destination.glob("*.zip"))
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination)
        
        # Remove the zip file to save space
        zip_path.unlink()
        
        # Find the CSV file (assuming the first CSV in the folder is the target)
        csv_files = list(destination.glob("*.csv"))
        if not csv_files:
            raise ValueError("No CSV file found in the Kaggle dataset.")
            
        dataset_file = csv_files[0]
        
        return {
            "provider": "kaggle",
            "source_id": reference,
            "license": "Kaggle API (See dataset page)",
            "file_path": str(dataset_file)
        }
