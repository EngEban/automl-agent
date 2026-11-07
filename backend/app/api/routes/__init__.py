"""
Routes package. Re-exports for backward compatibility.
"""
from app.api.routes.analysis import router

__all__ = ["router"]