"""SQL-backed retriever implementation for Retrieval-Augmented Generation."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from database import db

logger = logging.getLogger(__name__)


class SQLRetriever:
    """Retrieve knowledge snippets from a SQL knowledge base."""

    def __init__(
        self,
        db_instance=None,
        *,
        table_name: str = "knowledge_documents",
        search_fields: Optional[Sequence[str]] = None,
    ) -> None:
        self.db = db_instance if db_instance is not None else db
        self.table_name = table_name
        self.search_fields = list(search_fields) if search_fields else None

    def configure(
        self,
        *,
        table_name: Optional[str] = None,
        search_fields: Optional[Sequence[str]] = None,
    ) -> None:
        """Update retriever configuration at runtime for flexibility."""

        if table_name:
            self.table_name = table_name
        if search_fields is not None:
            self.search_fields = list(search_fields)

    def retrieve(self, query: str, limit: int = 3) -> List[Dict[str, object]]:
        """Return relevant documents for the provided query."""

        if not query or not query.strip():
            return []
        if not self.db:
            logger.debug("SQLRetriever missing database; returning empty result.")
            return []

        try:
            return self.db.search_knowledge_documents(
                query=query,
                limit=limit,
                table_name=self.table_name,
                search_fields=self.search_fields,
            )
        except AttributeError:
            logger.warning(
                "Database instance lacks knowledge search; returning empty result."
            )
            return []
        except Exception as exc:
            logger.exception("Unexpected error during SQL retrieval: %s", exc)
            return []
