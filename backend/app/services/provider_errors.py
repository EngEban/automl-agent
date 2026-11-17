"""Safe errors shared by AI provider management services and routes."""
from __future__ import annotations


class ProviderManagementError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
