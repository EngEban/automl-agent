from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class DataSourceConnector(ABC):
    """Base interface for all dataset connectors."""
    
    @abstractmethod
    def validate_reference(self, reference: str) -> bool:
        """Check if the provided reference/URL is valid for this connector."""
        pass

    @abstractmethod
    def download_to_staging(self, reference: str, destination: Path) -> dict[str, Any]:
        """
        Download the dataset to the destination path.
        Returns a dictionary with provenance metadata (provider, license, checksum).
        """
        pass