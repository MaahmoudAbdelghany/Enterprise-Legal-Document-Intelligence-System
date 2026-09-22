"""LexisGraph API route handlers."""

from app.api.routes.documents import router as documents_router
from app.api.routes.query import router as query_router

__all__ = ["documents_router", "query_router"]
