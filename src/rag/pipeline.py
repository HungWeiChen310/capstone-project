"""Composable utilities for SQL-based Retrieval-Augmented Generation."""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, Iterable, List, Optional

from .sql_retriever import SQLRetriever

logger = logging.getLogger(__name__)

ContextHeaderFactory = Callable[[Optional[str], int], str]
DocumentFormatter = Callable[[Dict[str, object], int], str]


def _default_header_factory(language: Optional[str], document_count: int) -> str:
    if document_count <= 0:
        return ""
    headers = {
        "zh-Hant": "以下是知識庫中與您的提問相關的資料：",
        "zh-Hans": "以下是知识库中与您的提问相关的资料：",
        "en": "Here are knowledge base excerpts related to the question:",
        "ja": "以下は質問に関連するナレッジベースからの情報です：",
        "ko": "아래는 질문과 관련된 지식 베이스 정보입니다:",
    }
    return headers.get(language, headers["zh-Hant"])


class RAGPipeline:
    """Simple yet flexible RAG pipeline for SQL-backed knowledge bases."""

    def __init__(
        self,
        retriever: SQLRetriever,
        *,
        max_documents: int = 3,
        max_characters: int = 480,
        header_factory: Optional[ContextHeaderFactory] = None,
        document_formatter: Optional[DocumentFormatter] = None,
    ) -> None:
        self.retriever = retriever
        self.max_documents = max(1, max_documents)
        self.max_characters = max(120, max_characters)
        self.header_factory = header_factory or _default_header_factory
        self.document_formatter = document_formatter or self._default_document_formatter

    def retrieve_documents(self, query: str) -> List[Dict[str, object]]:
        """Fetch documents using the underlying retriever."""

        return self.retriever.retrieve(query, limit=self.max_documents)

    def build_context(self, query: str, *, language: Optional[str] = None) -> str:
        """Return a formatted context string for the query."""

        documents = self.retrieve_documents(query)
        if not documents:
            return ""
        formatted_chunks = [
            self.document_formatter(doc, index)
            for index, doc in enumerate(documents, start=1)
        ]
        body = "\n\n".join(chunk for chunk in formatted_chunks if chunk)
        if not body.strip():
            return ""
        header = self.header_factory(language, len(documents))
        if header:
            return f"{header}\n{body}"
        return body

    def build_context_message(self, query: str, *, language: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Create a system message containing contextual knowledge."""

        content = self.build_context(query, language=language)
        if not content:
            return None
        return {"role": "system", "content": content}

    def inject_context(
        self,
        messages: Iterable[Dict[str, object]],
        query: str,
        *,
        language: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Inject the context message into a conversation copy."""

        base_messages = [dict(message) for message in messages] if messages else []
        context_message = self.build_context_message(query, language=language)
        if context_message:
            insert_index = 1 if base_messages and base_messages[0].get("role") == "system" else 0
            base_messages.insert(insert_index, context_message)
        return base_messages

    # ------------------------------------------------------------------
    # Default formatting helpers
    # ------------------------------------------------------------------

    def _default_document_formatter(self, document: Dict[str, object], index: int) -> str:
        title = str(document.get("title") or f"Document {index}")
        content = self._prepare_snippet(document.get("content"))
        metadata_parts: List[str] = []
        tags = document.get("tags")
        source = document.get("source")
        updated = document.get("last_updated")
        if tags:
            metadata_parts.append(f"Tags: {tags}")
        if source:
            metadata_parts.append(f"Source: {source}")
        if updated:
            metadata_parts.append(f"Updated: {updated}")
        metadata = f" ({'; '.join(metadata_parts)})" if metadata_parts else ""
        return f"[{index}] {title}{metadata}\n{content}".strip()

    def _prepare_snippet(self, value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return "(內容空白)"
        if len(text) <= self.max_characters:
            return text
        truncated = text[: self.max_characters].rstrip()
        return f"{truncated}..."


def _parse_int(value: Optional[str], default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def build_default_pipeline(db_instance=None) -> RAGPipeline:
    """Factory that creates a configurable default pipeline."""

    table_name = os.getenv("SQL_RAG_TABLE", "knowledge_documents")
    search_fields_env = os.getenv("SQL_RAG_SEARCH_FIELDS", "title,content,tags")
    search_fields = [field.strip() for field in search_fields_env.split(",") if field.strip()]
    max_docs = _parse_int(os.getenv("SQL_RAG_MAX_DOCS"), 3, minimum=1, maximum=10)
    max_chars = _parse_int(os.getenv("SQL_RAG_MAX_CHARS_PER_DOC"), 480, minimum=120, maximum=2000)
    custom_header = os.getenv("SQL_RAG_CONTEXT_HEADER")

    retriever = SQLRetriever(
        db_instance=db_instance,
        table_name=table_name,
        search_fields=search_fields or None,
    )

    header_factory: ContextHeaderFactory
    if custom_header is not None:
        header_factory = lambda language, count: custom_header if count > 0 else ""
    else:
        header_factory = _default_header_factory

    return RAGPipeline(
        retriever,
        max_documents=max_docs,
        max_characters=max_chars,
        header_factory=header_factory,
    )
