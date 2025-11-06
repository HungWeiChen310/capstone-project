"""Simple retrieval-augmented generation utilities for the LINE bot project."""
from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import pyodbc
from . import database


logger = logging.getLogger(__name__)

# Regular expression that keeps latin words, numbers and CJK characters.
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class KnowledgeDocument:
    """A single chunk of text that can be retrieved by the RAG engine."""

    doc_id: str
    content: str
    metadata: Dict[str, str]


@dataclass(frozen=True)
class RetrievalResult:
    """Represents the outcome of a similarity search."""

    document: KnowledgeDocument
    score: float


class RAGKnowledgeBase:
    """Loads project files with TF-IDF and optionally ingests MSSQL data for retrieval."""

    allowed_extensions = {
        ".md",
        ".txt",
        ".py",
        ".html",
        ".htm",
        ".json",
        ".yaml",
        ".yml",
    }

    def __init__(
        self,
        source_paths: Optional[Sequence[os.PathLike[str] | str]] = None,
        *,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        max_file_size: int = 512_000,
        enable_db_ingestion: bool = True,
        db_instance: Optional[database.Database] = None,
    ) -> None:
        """Configure the knowledge base and load initial documents."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            logger.warning(
                "chunk_overlap (%s) is not smaller than chunk_size (%s); adjusting automatically",
                chunk_overlap,
                chunk_size,
            )
            chunk_overlap = max(0, chunk_size // 4)

        self.project_root = Path(__file__).resolve().parent.parent
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_size = max_file_size
        self._enable_db_ingestion = enable_db_ingestion and os.getenv(
            "ENABLE_RAG_DB", "true"
        ).lower() not in {"false", "0", "no"}
        self._db_instance: Optional[database.Database] = db_instance
        self._default_db_top_k = _safe_int(os.getenv("RAG_DB_TOP_K"), default=3)
        if self._default_db_top_k < 0:
            self._default_db_top_k = 0
        self._documents: List[KnowledgeDocument] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_term_frequencies: List[Counter[str]] = []
        self._doc_vectors: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}
        self._lock = threading.RLock()

        env_sources = os.getenv("RAG_SOURCE_PATHS")
        book_dir = self.project_root / "src_book"
        if source_paths is None:
            if book_dir.exists():
                source_paths = [
                    path
                    for path in book_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in {".txt", ".md"}
                ]
            elif env_sources:
                source_paths = [path for path in env_sources.split(os.pathsep) if path]

        self._source_paths: Sequence[Path] = self._resolve_sources(source_paths)
        with self._lock:
            self._load_documents()
            self._load_database_documents()


    #----------------------------------------------------------------
    # MS SQL ingestion
    #----------------------------------------------------------------

    def ingest_from_mssql(
        self,
        db: database.Database,
    ) -> None:
        """Load MSSQL tables and build KnowledgeDocument entries for the knowledge base."""

        if db is None:
            raise ValueError("Database instance is required for MSSQL ingestion.")

        def _format_value(value: object) -> str:
            if value is None:
                return ""
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, (datetime, date)):
                return value.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(value, Decimal):
                normalized = format(value, "f")
                return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized
            return str(value).strip()

        def _rows_to_dicts(cursor, rows) -> List[Dict[str, object]]:
            columns = [column[0] for column in cursor.description]
            dict_rows: List[Dict[str, object]] = []
            for row in rows:
                dict_rows.append({column: row[idx] for idx, column in enumerate(columns)})
            return dict_rows

        def _ingest_rows(
            rows: List[Dict[str, object]],
            *,
            id_columns: Sequence[str],
            text_columns: Sequence[str],
            meta_columns: Sequence[str],
            source_tag: str,
            table_name: str,
        ) -> Tuple[List[KnowledgeDocument], List[List[str]]]:
            new_docs: List[KnowledgeDocument] = []
            new_tokens: List[List[str]] = []
            for row in rows:
                doc_id_parts = [
                    str(row.get(column))
                    for column in id_columns
                    if row.get(column) not in (None, "")
                ]
                if not doc_id_parts:
                    continue
                content_lines: List[str] = []
                for column in text_columns:
                    raw_value = _format_value(row.get(column))
                    if not raw_value:
                        continue
                    label = column.replace("_", " ").title()
                    content_lines.append(f"{label}: {raw_value}")
                content = "\n".join(content_lines).strip()
                if not content:
                    continue
                metadata: Dict[str, str] = {
                    "source": source_tag,
                    "row_id": "::".join(doc_id_parts),
                    "origin": "database",
                    "source_type": "mssql",
                    "source_table": table_name,
                }
                for column in meta_columns:
                    value = _format_value(row.get(column))
                    if value:
                        metadata[column] = value
                doc_id = f"{source_tag}::" + "::".join(doc_id_parts)
                document = KnowledgeDocument(doc_id=doc_id, content=content, metadata=metadata)
                new_docs.append(document)
                new_tokens.append(self._tokenize(content))
            return new_docs, new_tokens

        data_sources = [
            {
                "table": "equipment",
                "source_tag": "mssql:equipment",
                "query": (
                    "SELECT equipment_id, name, equipment_type, status, last_updated "
                    "FROM equipment"
                ),
                "id_columns": ("equipment_id",),
                "text_columns": ("name", "equipment_type", "status"),
                "meta_columns": ("last_updated",),
            },
            {
                "table": "alert_history",
                "source_tag": "mssql:alert_history",
                "query": (
                    "SELECT ah.error_id, ah.equipment_id, e.name AS equipment_name, "
                    "ah.detected_anomaly_type, ah.severity_level, ah.is_resolved, ah.created_time, "
                    "ah.resolved_time, ah.resolved_by, ah.resolution_notes "
                    "FROM alert_history AS ah "
                    "LEFT JOIN equipment AS e ON ah.equipment_id = e.equipment_id"
                ),
                "id_columns": ("error_id", "equipment_id"),
                "text_columns": ("equipment_name", "detected_anomaly_type", "severity_level", "resolution_notes"),
                "meta_columns": ("is_resolved", "created_time", "resolved_time", "resolved_by"),
            },
            {
                "table": "equipment_metrics",
                "source_tag": "mssql:equipment_metrics",
                "query": (
                    "SELECT em.id, em.equipment_id, e.name AS equipment_name, "
                    "em.metric_type, em.status, em.value, em.threshold_min, "
                    "em.threshold_max, em.unit, em.last_updated "
                    "FROM equipment_metrics AS em "
                    "LEFT JOIN equipment AS e ON em.equipment_id = e.equipment_id"
                ),
                "id_columns": ("id",),
                "text_columns": ("equipment_name", "metric_type", "status", "value"),
                "meta_columns": ("threshold_min", "threshold_max", "unit", "last_updated"),
            },
            {
                "table": "error_logs",
                "source_tag": "mssql:error_logs",
                "query": (
                    "SELECT el.error_id, el.equipment_id, e.name AS equipment_name, "
                    "el.detected_anomaly_type, el.severity_level, el.log_date, el.event_time, "
                    "el.resolved_time, el.downtime_sec "
                    "FROM error_logs AS el "
                    "LEFT JOIN equipment AS e ON el.equipment_id = e.equipment_id"
                ),
                "id_columns": ("error_id", "equipment_id"),
                "text_columns": ("equipment_name", "detected_anomaly_type", "severity_level"),
                "meta_columns": ("log_date", "event_time", "resolved_time", "downtime_sec"),
            },
        ]

        ingested_batches: List[Tuple[str, List[KnowledgeDocument], List[List[str]]]] = []

        try:
            with db._get_connection() as conn:
                for source in data_sources:
                    cursor = conn.cursor()
                    try:
                        cursor.execute(source["query"])
                        rows_raw = cursor.fetchall()
                        if not rows_raw:
                            continue
                        row_dicts = _rows_to_dicts(cursor, rows_raw)
                        docs, tokens = _ingest_rows(
                            row_dicts,
                            id_columns=source["id_columns"],
                            text_columns=source["text_columns"],
                            meta_columns=source["meta_columns"],
                            source_tag=source["source_tag"],
                            table_name=source["table"],
                        )
                        if docs:
                            ingested_batches.append((source["source_tag"], docs, tokens))
                            logger.info(
                                "Loaded %s documents from MSSQL table '%s'.",
                                len(docs),
                                source["table"],
                            )
                    except pyodbc.Error as exc:
                        logger.error(
                            "Failed to ingest MSSQL table '%s': %s",
                            source["table"],
                            exc,
                        )
                    except Exception as exc:
                        logger.exception(
                            "Unexpected error while ingesting table '%s': %s",
                            source["table"],
                            exc,
                        )
                    finally:
                        cursor.close()
        except pyodbc.Error as exc:
            logger.error("Failed to connect to MSSQL for RAG ingestion: %s", exc)
            return

        if not ingested_batches:
            logger.info("MSSQL ingestion completed with no new documents.")
            return

        prefixes = {f"{source_tag}::" for source_tag, _, _ in ingested_batches}

        with self._lock:
            retained_docs: List[KnowledgeDocument] = []
            retained_tokens: List[List[str]] = []
            for doc, tokens in zip(self._documents, self._doc_tokens):
                if any(doc.doc_id.startswith(prefix) for prefix in prefixes):
                    continue
                retained_docs.append(doc)
                retained_tokens.append(tokens)

            for _, docs, tokens in ingested_batches:
                retained_docs.extend(docs)
                retained_tokens.extend(tokens)

            self._documents = retained_docs
            self._doc_tokens = retained_tokens
            self._rebuild_index()

        total_docs = sum(len(docs) for _, docs, _ in ingested_batches)
        logger.info("MSSQL ingestion added %s documents to the knowledge base.", total_docs)

    def _load_database_documents(self) -> None:
        """Load database-backed knowledge chunks when enabled."""
        if not self._enable_db_ingestion:
            return
        db_instance = self._db_instance or getattr(database, "db", None)
        if db_instance is None:
            logger.debug("Skipping MSSQL ingestion for RAG knowledge base: no database instance available.")
            return
        self._db_instance = db_instance
        try:
            self.ingest_from_mssql(db_instance)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to ingest MSSQL data for RAG knowledge base: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def documents(self) -> List[KnowledgeDocument]:
        return list(self._documents)

    @property
    def is_ready(self) -> bool:
        return bool(self._documents)

    def refresh(self) -> None:
        """Reload documents and rebuild the internal index."""
        with self._lock:
            self._load_documents()
            self._load_database_documents()

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float = 0.05,
        include_db_results: Optional[bool] = None,
        db_top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Search the knowledge base prioritising local TF-IDF chunks and optionally appending MSSQL rows."""
        query = (query or "").strip()
        if not query or not self._documents:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        query_vector = self._vector_from_tokens(tokens)
        if not query_vector:
            return []

        include_db = self._enable_db_ingestion if include_db_results is None else include_db_results
        effective_db_top_k = db_top_k if db_top_k is not None else self._default_db_top_k
        if effective_db_top_k is None:
            effective_db_top_k = 0
        if effective_db_top_k < 0:
            effective_db_top_k = 0

        text_hits: List[RetrievalResult] = []
        db_hits: List[RetrievalResult] = []

        for document, doc_vector in zip(self._documents, self._doc_vectors):
            if not doc_vector:
                continue
            score = sum(query_vector.get(term, 0.0) * weight for term, weight in doc_vector.items())
            if score < min_score:
                continue
            result = RetrievalResult(document=document, score=score)
            origin = document.metadata.get("origin")
            if origin == "database":
                db_hits.append(result)
            else:
                text_hits.append(result)

        if not text_hits and not db_hits:
            return []

        text_hits.sort(key=lambda item: item.score, reverse=True)
        primary_results = text_hits[:top_k]

        if not include_db or effective_db_top_k == 0:
            return primary_results

        db_hits.sort(key=lambda item: item.score, reverse=True)
        supplemental = db_hits[:effective_db_top_k]
        if not supplemental:
            return primary_results

        return primary_results + supplemental

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_sources(
        self,
        source_paths: Optional[Sequence[os.PathLike[str] | str]],
    ) -> Sequence[Path]:
        if not source_paths:
            defaults: List[Path] = [
                self.project_root / "Documentary.md",
                self.project_root / "README.md",
                self.project_root / "src",
                self.project_root / "templates",
            ]
            return tuple(path for path in defaults if path.exists())

        resolved: List[Path] = []
        for raw in source_paths:
            path = Path(raw)
            if not path.is_absolute():
                path = self.project_root / path
            if path.exists():
                resolved.append(path)
            else:
                logger.debug("Skipping unavailable RAG source: %s", path)
        return tuple(resolved)

    def _load_documents(self) -> None:
        documents: List[KnowledgeDocument] = []
        doc_tokens: List[List[str]] = []

        for file_path in self._iter_source_files(self._source_paths):
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                logger.warning("Failed to read file %s: %s", file_path, exc)
                continue

            normalized = text.replace("\r\n", "\n")
            chunks = [chunk.strip() for chunk in self._split_into_chunks(normalized) if chunk.strip()]
            if not chunks:
                continue

            relative_path: str
            try:
                relative_path = str(file_path.relative_to(self.project_root))
            except ValueError:
                relative_path = str(file_path)

            total_chunks = len(chunks)
            for idx, chunk in enumerate(chunks, start=1):
                metadata = {
                    "source": relative_path,
                    "chunk_index": str(idx),
                    "chunk_count": str(total_chunks),
                    "origin": "text",
                    "source_type": "filesystem",
                }
                doc_id = f"{relative_path}::chunk-{idx}"
                documents.append(KnowledgeDocument(doc_id=doc_id, content=chunk, metadata=metadata))
                doc_tokens.append(self._tokenize(chunk))

        self._documents = documents
        self._doc_tokens = doc_tokens
        self._rebuild_index()
        if documents:
            logger.info("RAG knowledge base loaded %s filesystem chunks.", len(documents))
        else:
            logger.warning("RAG knowledge base did not load any filesystem documents; verify source paths.")

    def _iter_source_files(self, paths: Sequence[Path]) -> Iterable[Path]:
        for path in paths:
            if path.is_file() and self._is_allowed_file(path):
                yield path
            elif path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file() and self._is_allowed_file(child):
                        yield child

    def _is_allowed_file(self, file_path: Path) -> bool:
        if file_path.name.startswith("."):
            return False
        if file_path.suffix.lower() not in self.allowed_extensions:
            return False
        try:
            if file_path.stat().st_size > self.max_file_size:
                logger.debug("Skipping oversized file for RAG: %s", file_path)
                return False
        except OSError:
            return False
        return True

    def _split_into_chunks(self, text: str) -> Iterable[str]:
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        if not paragraphs:
            yield text
            return

        current: List[str] = []
        current_length = 0

        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            if current and current_length + paragraph_length + 2 > self.chunk_size:
                yield "\n\n".join(current)
                current = []
                current_length = 0

            if paragraph_length <= self.chunk_size:
                current.append(paragraph)
                current_length += paragraph_length + 2
                continue

            # Paragraph is larger than chunk size: split with overlap
            start = 0
            while start < paragraph_length:
                end = min(start + self.chunk_size, paragraph_length)
                chunk = paragraph[start:end]
                if chunk:
                    yield chunk
                if end == paragraph_length:
                    break
                start += max(1, self.chunk_size - self.chunk_overlap)

        if current:
            yield "\n\n".join(current)

    def _tokenize(self, text: str) -> List[str]:
        return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]

    def _rebuild_index(self) -> None:
        doc_count = len(self._documents)
        term_document_counts: Counter[str] = Counter()
        doc_term_frequencies: List[Counter[str]] = []

        for tokens in self._doc_tokens:
            frequency = Counter(tokens)
            doc_term_frequencies.append(frequency)
            term_document_counts.update(frequency.keys())

        self._doc_term_frequencies = doc_term_frequencies
        if not doc_count:
            self._idf = {}
            self._doc_vectors = []
            return

        self._idf = {
            term: math.log((doc_count + 1) / (count + 1)) + 1.0
            for term, count in term_document_counts.items()
        }

        vectors: List[Dict[str, float]] = []
        for frequency in self._doc_term_frequencies:
            total_terms = sum(frequency.values())
            if not total_terms:
                vectors.append({})
                continue

            vector: Dict[str, float] = {}
            for term, count in frequency.items():
                weight = (count / total_terms) * self._idf.get(term, 0.0)
                if weight:
                    vector[term] = weight

            norm = math.sqrt(sum(weight ** 2 for weight in vector.values()))
            if norm:
                vector = {term: weight / norm for term, weight in vector.items()}
            vectors.append(vector)

        self._doc_vectors = vectors

    def _vector_from_tokens(self, tokens: List[str]) -> Dict[str, float]:
        frequency = Counter(tokens)
        total_terms = sum(frequency.values())
        if not total_terms:
            return {}

        vector: Dict[str, float] = {}
        for term, count in frequency.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            weight = (count / total_terms) * idf
            if weight:
                vector[term] = weight

        norm = math.sqrt(sum(weight ** 2 for weight in vector.values()))
        if not norm:
            return {}
        return {term: weight / norm for term, weight in vector.items()}


_default_kb: Optional[RAGKnowledgeBase] = None
_default_kb_lock = threading.Lock()


def get_default_knowledge_base() -> RAGKnowledgeBase:
    """Return a lazily instantiated knowledge base shared across the app."""
    global _default_kb
    if _default_kb is None:
        with _default_kb_lock:
            if _default_kb is None:
                chunk_size = _safe_int(os.getenv("RAG_CHUNK_SIZE"), default=600)
                chunk_overlap = _safe_int(os.getenv("RAG_CHUNK_OVERLAP"), default=120)
                _default_kb = RAGKnowledgeBase(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
    return _default_kb


def _safe_int(raw_value: Optional[str], *, default: int) -> int:
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning("Unable to parse integer value %s; using default %s", raw_value, default)
        return default
    return value
