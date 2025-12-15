"""
Vector-based retrieval-augmented generation utilities for the LINE bot project.
Replaces the original TF-IDF implementation with a sentence-transformer and ChromaDB backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
import threading
import time
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import chromadb
import pyodbc
import torch
from sentence_transformers import SentenceTransformer

try:
    from .database import db
    from . import database
    from .utils import _format_value
except ImportError:
    from database import db
    import database
    from utils import _format_value

logger = logging.getLogger(__name__)


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
    score: float  # Similarity score, where higher is better


class RAGKnowledgeBase:
    """
    Loads project files and MSSQL data into a ChromaDB vector store for retrieval,
    using a sentence-transformer model for semantic embeddings.
    """

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
    # A powerful multilingual model that works well with Traditional Chinese
    DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
    DEFAULT_CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "rag_db")
    DEFAULT_COLLECTION_NAME = "knowledge_base"
    DEFAULT_COLLECTION_METADATA = {"hnsw:space": "cosine"}
    STATE_VERSION = 2

    def __init__(
        self,
        source_paths: Optional[Sequence[os.PathLike[str] | str]] = None,
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        max_file_size: int = 512_000,
        enable_db_ingestion: bool = True,
        db_instance: Optional[database.Database] = None,
        embedding_model_name: Optional[str] = None,
        chroma_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        auto_refresh_interval: int = 3600,  # Default refresh every 1 hour
    ) -> None:
        """Configure the knowledge base, initialize the model, and load documents."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 4)

        self.project_root = Path(__file__).resolve().parent.parent
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_size = max_file_size
        # Default to the global db instance so MSSQL rows are ingested into the vector store.
        self._db_instance: Optional[database.Database] = db_instance or db
        self._lock = threading.RLock()
        self.auto_refresh_interval = auto_refresh_interval
        self._known_anomaly_types: Optional[Set[str]] = None

        # Setup Sentence Transformer model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = embedding_model_name or self.DEFAULT_EMBEDDING_MODEL
        logger.info("Initializing SentenceTransformer model '%s' on device '%s'.", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self._embedding_model_name = model_name

        # Setup ChromaDB
        _chroma_path = chroma_path or self.DEFAULT_CHROMA_PATH
        self._collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        self._collection_metadata = dict(self.DEFAULT_COLLECTION_METADATA)
        self.chroma_client = chromadb.PersistentClient(path=_chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            self._collection_name,
            metadata=self._collection_metadata,
        )

        self._state_file = Path(_chroma_path) / "rag_state.json"

        # Resolve source paths
        env_sources = os.getenv("RAG_SOURCE_PATHS")
        if source_paths is None and env_sources:
            source_paths = [p for p in env_sources.split(os.pathsep) if p]
        self._source_paths: Sequence[Path] = self._resolve_sources(source_paths)

        self._enable_db_ingestion = enable_db_ingestion and os.getenv(
            "ENABLE_RAG_DB", "true"
        ).lower() not in {"false", "0", "no"}
        self._default_db_top_k = int(os.getenv("RAG_DB_TOP_K", "3"))
        self._ensure_store_matches_config()

        # Initial data load
        with self._lock:
            # Always try to sync on startup, but efficiently
            logger.info("Initializing Knowledge Base. Syncing documents...")
            self._sync_documents()
            self._sync_database_documents()

        # Start background refresh thread
        self._start_auto_refresh_thread()

    def _start_auto_refresh_thread(self):
        def refresh_task():
            while True:
                time.sleep(self.auto_refresh_interval)
                try:
                    logger.info("Auto-refreshing RAG knowledge base...")
                    self.refresh()
                except Exception as e:
                    logger.error(f"Error during auto-refresh: {e}")

        thread = threading.Thread(target=refresh_task, daemon=True, name="RAGRefreshThread")
        thread.start()

    def _rows_to_dicts(self, cursor, rows) -> List[Dict[str, object]]:
        columns = [column[0] for column in cursor.description]
        return [{column: row[idx] for idx, column in enumerate(columns)} for row in rows]

    def _get_known_anomaly_types(self) -> Set[str]:
        cached = self._known_anomaly_types
        if cached is not None:
            return cached

        anomaly_types: Set[str] = set()
        db_instance = self._db_instance
        if not db_instance:
            self._known_anomaly_types = anomaly_types
            return anomaly_types

        try:
            with db_instance._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT detected_anomaly_type FROM stats_abnormal_monthly")
                for row in cursor.fetchall():
                    value = _format_value(row[0])
                    if value:
                        anomaly_types.add(value)
        except Exception as exc:
            logger.info("Unable to load anomaly types for hinting: %s", exc)

        self._known_anomaly_types = anomaly_types
        return anomaly_types

    def _load_state(self) -> Dict[str, Any]:
        if self._state_file.exists():
            try:
                with open(self._state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load RAG state file: {e}. Resetting state.")
        return {}

    def _save_state(self, state: Dict[str, Any]) -> None:
        try:
            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save RAG state file: {e}")

    def _get_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _get_expected_store_config(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "embedding_model": self._embedding_model_name,
            "normalize_embeddings": True,
            "collection_name": self._collection_name,
            "collection_metadata": self._collection_metadata,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_file_size": self.max_file_size,
            "enable_db_ingestion": self._enable_db_ingestion,
            "source_paths": [str(p) for p in self._source_paths],
        }

    def _reset_vector_store(self) -> None:
        try:
            self.chroma_client.delete_collection(self._collection_name)
        except Exception as exc:
            logger.info(
                "Chroma collection '%s' reset skipped (may not exist): %s",
                self._collection_name,
                exc,
            )
        self.collection = self.chroma_client.get_or_create_collection(
            self._collection_name,
            metadata=self._collection_metadata,
        )

    def _ensure_store_matches_config(self) -> None:
        expected = self._get_expected_store_config()
        state = self._load_state()
        if state.get("config") == expected:
            return

        logger.info(
            "RAG store config changed (or missing). Rebuilding Chroma collection '%s'.",
            self._collection_name,
        )
        self._reset_vector_store()
        self._save_state({"config": expected, "files": {}, "database": {}})

    def _sync_database_documents(self) -> None:
        """Incrementally sync database records to vector store."""
        if not self._enable_db_ingestion:
            return

        db_instance = self._db_instance
        if not db_instance:
            # Use INFO so operators notice the missing DB ingestion path.
            logger.info("Skipping MSSQL ingestion: no database instance available.")
            return
        self._db_instance = db_instance

        # Simplified data source config
        data_sources = [
            {
                "query": "SELECT * FROM equipment",
                "table": "equipment",
                "source_tag": "Equipment",
                "id_columns": ["equipment_id"],
                "text_columns": ["name", "equipment_type", "status", "location"],
                "meta_columns": ["equipment_id", "equipment_type"],
            },
            {
                "query": "SELECT TOP 50 * FROM alert_history ORDER BY created_time DESC",
                "table": "alert_history",
                "source_tag": "Alert",
                "id_columns": ["error_id"],
                "text_columns": ["equipment_id", "detected_anomaly_type", "severity_level", "resolution_notes"],
                "meta_columns": ["equipment_id", "is_resolved"],
            },
            {
                "query": "SELECT TOP 100 * FROM error_logs ORDER BY log_date DESC",
                "table": "error_logs",
                "source_tag": "ErrorLog",
                "id_columns": ["log_date", "equipment_id", "error_id"],
                "text_columns": ["log_date", "equipment_id", "detected_anomaly_type", "severity_level", "downtime_sec"],
                "meta_columns": ["equipment_id", "severity_level"],
            },
            {
                "query": (
                    "SELECT *, "
                    "N'月度營運與異常(停機)總覽：查「異常總情況/停機秒數/停機率」用 stats_operational_monthly | "
                    "Total operational & abnormal(downtime) monthly summary' "
                    "as description "
                    "FROM stats_operational_monthly"
                ),
                "table": "stats_operational_monthly",
                "source_tag": "StatsOpMonthly",
                "id_columns": ["equipment_id", "year", "month"],
                "text_columns": ["description", "equipment_id", "year", "month", "total_operation_hrs", "downtime_sec", "downtime_rate_percent"],
                "meta_columns": ["equipment_id", "year", "month"],
            },
            {
                "query": (
                    "SELECT *, "
                    "N'月度異常分項(依 detected_anomaly_type)：查「各類異常/異常分佈」用 stats_abnormal_monthly；"
                    "查「異常總情況」用 stats_operational_monthly | "
                    "Monthly abnormal breakdown by anomaly type' "
                    "as description "
                    "FROM stats_abnormal_monthly"
                ),
                "table": "stats_abnormal_monthly",
                "source_tag": "StatsAbnormalMonthly",
                "id_columns": ["equipment_id", "year", "month", "detected_anomaly_type"],
                "text_columns": ["description", "equipment_id", "year", "month", "detected_anomaly_type", "downtime_sec", "downtime_rate_percent"],
                "meta_columns": ["equipment_id", "year", "month", "detected_anomaly_type"],
            },
            {
                "query": (
                    "SELECT *, "
                    "N'季度營運與異常(停機)總覽：查「異常總情況/停機秒數/停機率」用 stats_operational_quarterly | "
                    "Total operational & abnormal(downtime) quarterly summary' "
                    "as description "
                    "FROM stats_operational_quarterly"
                ),
                "table": "stats_operational_quarterly",
                "source_tag": "StatsOpQuarterly",
                "id_columns": ["equipment_id", "year", "quarter"],
                "text_columns": ["description", "equipment_id", "year", "quarter", "total_operation_hrs", "downtime_sec", "downtime_rate_percent"],
                "meta_columns": ["equipment_id", "year", "quarter"],
            },
            {
                "query": (
                    "SELECT *, "
                    "N'年度營運與異常(停機)總覽：查「異常總情況/停機秒數/停機率」用 stats_operational_yearly | "
                    "Total operational & abnormal(downtime) yearly summary' "
                    "as description "
                    "FROM stats_operational_yearly"
                ),
                "table": "stats_operational_yearly",
                "source_tag": "StatsOpYearly",
                "id_columns": ["equipment_id", "year"],
                "text_columns": ["description", "equipment_id", "year", "total_operation_hrs", "downtime_sec", "downtime_rate_percent"],
                "meta_columns": ["equipment_id", "year"],
            },
            {
                "query": (
                    "SELECT *, "
                    "N'季度異常分項(依 detected_anomaly_type)：查「各類異常/異常分佈」用 stats_abnormal_quarterly；"
                    "查「異常總情況」用 stats_operational_quarterly | "
                    "Quarterly abnormal breakdown by anomaly type' "
                    "as description "
                    "FROM stats_abnormal_quarterly"
                ),
                "table": "stats_abnormal_quarterly",
                "source_tag": "StatsAbnormalQuarterly",
                "id_columns": ["equipment_id", "year", "quarter", "detected_anomaly_type"],
                "text_columns": ["description", "equipment_id", "year", "quarter", "detected_anomaly_type", "downtime_sec", "downtime_rate_percent"],
                "meta_columns": ["equipment_id", "year", "quarter", "detected_anomaly_type"],
            },
            {
                "query": (
                    "SELECT *, "
                    "N'年度異常分項(依 detected_anomaly_type)：查「各類異常/異常分佈」用 stats_abnormal_yearly；"
                    "查「異常總情況」用 stats_operational_yearly | "
                    "Yearly abnormal breakdown by anomaly type' "
                    "as description "
                    "FROM stats_abnormal_yearly"
                ),
                "table": "stats_abnormal_yearly",
                "source_tag": "StatsAbnormalYearly",
                "id_columns": ["equipment_id", "year", "detected_anomaly_type"],
                "text_columns": ["description", "equipment_id", "year", "detected_anomaly_type", "downtime_sec", "downtime_rate_percent"],
                "meta_columns": ["equipment_id", "year", "detected_anomaly_type"],
            },
        ]

        state = self._load_state()
        db_state = state.get("database", {})
        new_db_state = {}
        docs_to_add: List[KnowledgeDocument] = []
        ids_processed: Set[str] = set()

        try:
            with db_instance._get_connection() as conn:
                for source in data_sources:
                    cursor = conn.cursor()
                    try:
                        cursor.execute(source["query"])
                        rows_raw = cursor.fetchall()
                        if not rows_raw:
                            continue

                        row_dicts = self._rows_to_dicts(cursor, rows_raw)

                        for row in row_dicts:
                            doc_id_parts = [
                                str(row.get(col))
                                for col in source["id_columns"]
                                if row.get(col) not in (None, "")
                            ]
                            if not doc_id_parts:
                                continue

                            # Build content
                            content_parts = []
                            for col in source["text_columns"]:
                                val = _format_value(row.get(col))
                                if val:
                                    content_parts.append(f"{col.replace('_', ' ').title()}: {val}")

                            content = "\n".join(content_parts).strip()
                            if not content:
                                continue

                            # Unique ID for this record in DB
                            record_unique_id = f"{source['source_tag']}::{'::'.join(doc_id_parts)}"
                            current_hash = self._get_content_hash(content)

                            new_db_state[record_unique_id] = current_hash
                            ids_processed.add(record_unique_id)

                            # Check if changed
                            if (
                                record_unique_id not in db_state
                                or db_state[record_unique_id] != current_hash
                            ):
                                # Prepare metadata
                                metadata = {
                                    "source": source["source_tag"],
                                    "row_id": "::".join(doc_id_parts),
                                    "origin": "database",
                                    "source_type": "mssql",
                                    "source_table": source["table"],
                                }
                                for col in source["meta_columns"]:
                                    value = _format_value(row.get(col))
                                    if value:
                                        metadata[col] = value

                                docs_to_add.append(
                                    KnowledgeDocument(
                                        doc_id=record_unique_id,
                                        content=content,
                                        metadata=metadata,
                                    )
                                )

                    except pyodbc.Error as e:
                        logger.error("Failed to ingest MSSQL table '%s': %s", source["table"], e)
        except pyodbc.Error as e:
            logger.error("Failed to connect to MSSQL for RAG ingestion: %s", e)
            return

        with self._lock:
            # 1. Remove deleted records
            ids_to_delete = [doc_id for doc_id in db_state if doc_id not in ids_processed]
            if ids_to_delete:
                try:
                    logger.info(f"Removing {len(ids_to_delete)} obsolete database records from Knowledge Base.")
                    self.collection.delete(ids=ids_to_delete)
                except Exception as e:
                    logger.error(f"Failed to delete obsolete DB records: {e}")

            # 2. Upsert new/changed records
            if docs_to_add:
                logger.info(f"Upserting {len(docs_to_add)} new/changed database records.")
                self._add_documents_to_collection(docs_to_add)

            # 3. Update state
            state["database"] = new_db_state
            self._save_state(state)

        logger.info("Database sync complete.")

    @property
    def documents(self) -> List[KnowledgeDocument]:
        """Retrieve all documents from the collection."""
        results = self.collection.get()
        return [
            KnowledgeDocument(doc_id, content, metadata)
            for doc_id, content, metadata in zip(results['ids'], results['documents'], results['metadatas'])
        ]

    @property
    def is_ready(self) -> bool:
        return self.collection.count() > 0

    def refresh(self) -> None:
        """Incrementally refresh the knowledge base."""
        with self._lock:
            logger.info("Scanning for changes in knowledge base...")
            self._sync_documents()
            self._sync_database_documents()
        logger.info("Knowledge base refresh complete.")

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float = 0.3,
        include_db_results: Optional[bool] = None,
        db_top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Search the knowledge base using semantic vector search."""
        if not query.strip() or not self.is_ready:
            return []

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_vector = query_embedding.tolist()
        hints = self._extract_query_hints(query)

        include_db = self._enable_db_ingestion if include_db_results is None else include_db_results
        db_k = db_top_k if db_top_k is not None else self._default_db_top_k

        # Query for filesystem documents
        text_results_raw = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"origin": "text"},
        )

        final_results: List[RetrievalResult] = self._process_query_results(text_results_raw, min_score)

        # Query for database documents if enabled
        if include_db and db_k > 0:
            structured_db_results = self._structured_db_lookup(query_vector, hints)
            if structured_db_results:
                # When we have a precise, structured hit (equipment + time grain),
                # prefer it over broad similarity search to avoid bringing in the wrong stats table.
                final_results.extend(structured_db_results)
            else:
                db_results_raw = self.collection.query(
                    query_embeddings=[query_vector],
                    n_results=db_k,
                    where={"origin": "database"},
                )
                db_results = self._process_query_results(db_results_raw, min_score)

                # If this looks like a "total" stats question, only fall back to abnormal breakdown tables
                # when no operational stats table can be found in the retrieved candidates.
                wants_total = bool(hints.get("wants_total"))
                wants_breakdown = bool(hints.get("wants_breakdown"))
                if wants_total and not wants_breakdown:
                    non_abnormal = []
                    has_operational_stats = False
                    for item in db_results:
                        metadata = item.document.metadata or {}
                        table = (metadata.get("source_table") or "").lower()
                        if table.startswith("stats_abnormal_"):
                            continue
                        if table.startswith("stats_operational_"):
                            has_operational_stats = True
                        non_abnormal.append(item)
                    if has_operational_stats and non_abnormal:
                        db_results = non_abnormal

                final_results.extend(db_results)

        final_results = self._dedupe_results(final_results)
        final_results = self._rerank_results(query, final_results, hints=hints)
        return final_results

    def _extract_query_hints(self, query: str) -> Dict[str, Any]:
        wants_breakdown = any(
            keyword in query
            for keyword in (
                "類型",
                "分佈",
                "分項",
                "細項",
                "明細",
                "各類",
                "各種",
            )
        )
        breakdown_keywords_hit = wants_breakdown
        wants_total = any(
            keyword in query
            for keyword in (
                "總情況",
                "總覽",
                "彙總",
                "總計",
                "總和",
                "整體",
                "概況",
                "概覽",
                "停機率",
                "停機秒數",
                "停機時間",
            )
        )

        equipment_id = None
        equipment_match = re.search(r"(?i)eq\s*(\d{3})", query)
        if equipment_match:
            equipment_id = f"EQ{equipment_match.group(1)}"

        year = None
        year_match = re.search(r"(20\d{2})\s*年", query)
        if year_match:
            year = year_match.group(1)

        month = None
        month_match = re.search(r"(1[0-2]|0?[1-9])\s*月", query)
        if month_match:
            month = str(int(month_match.group(1)))

        quarter = None
        quarter_match = re.search(r"(?:第\s*)?([1-4])\s*季|\bQ([1-4])\b", query, flags=re.IGNORECASE)
        if quarter_match:
            quarter = quarter_match.group(1) or quarter_match.group(2)

        # Fallback parsing: support "2025-05" / "2025/5" when "年/月" is omitted.
        if not (year and month):
            year_month_match = re.search(r"(20\d{2})\s*[-/\.]\s*(1[0-2]|0?[1-9])", query)
            if year_month_match:
                year = year or year_month_match.group(1)
                month = month or str(int(year_month_match.group(2)))

        if not (year and quarter):
            year_quarter_match = re.search(r"(20\d{2})\s*Q([1-4])", query, flags=re.IGNORECASE)
            if year_quarter_match:
                year = year or year_quarter_match.group(1)
                quarter = quarter or year_quarter_match.group(2)

        grain = None
        if year and month:
            grain = "monthly"
        elif year and quarter:
            grain = "quarterly"
        elif year:
            grain = "yearly"

        is_data_query = bool(equipment_id) and bool(year)

        # If the user mentions a concrete anomaly type, prefer the abnormal breakdown tables.
        anomaly_type = None
        filter_anomaly_type = False
        if is_data_query and grain:
            known_types = sorted(self._get_known_anomaly_types(), key=len, reverse=True)
            for known_type in known_types:
                if known_type and known_type in query:
                    anomaly_type = known_type
                    # If no breakdown keywords were used, treat it as a "single anomaly type" query.
                    if not breakdown_keywords_hit:
                        wants_breakdown = True
                        filter_anomaly_type = True
                    break

        # If the user asks about abnormal/downtime without specifying a breakdown,
        # default to the operational (total) tables.
        if (
            is_data_query
            and grain
            and (not wants_breakdown)
            and (not wants_total)
            and any(keyword in query for keyword in ("異常", "停機", "downtime"))
        ):
            wants_total = True

        return {
            "wants_total": wants_total,
            "wants_breakdown": wants_breakdown,
            "anomaly_type": anomaly_type,
            "filter_anomaly_type": filter_anomaly_type,
            "equipment_id": equipment_id,
            "year": year,
            "month": month,
            "quarter": quarter,
            "grain": grain,
            "is_data_query": is_data_query,
        }

    @staticmethod
    def _dedupe_results(results: List[RetrievalResult]) -> List[RetrievalResult]:
        by_id: Dict[str, RetrievalResult] = {}
        for result in results:
            doc_id = result.document.doc_id
            existing = by_id.get(doc_id)
            if existing is None or result.score > existing.score:
                by_id[doc_id] = result
        return list(by_id.values())

    def _structured_db_lookup(self, query_vector: List[float], hints: Dict[str, Any]) -> List[RetrievalResult]:
        """
        When the query contains structured hints (equipment + time grain),
        fetch the most relevant stats rows directly so the correct table is present in context.
        """
        if not hints.get("is_data_query"):
            return []

        equipment_id = hints.get("equipment_id")
        year = hints.get("year")
        month = hints.get("month")
        quarter = hints.get("quarter")
        grain = hints.get("grain")
        wants_total = bool(hints.get("wants_total"))
        wants_breakdown = bool(hints.get("wants_breakdown"))
        anomaly_type = hints.get("anomaly_type")
        filter_anomaly_type = bool(hints.get("filter_anomaly_type"))

        structured_results: List[RetrievalResult] = []

        def query_where(where: Dict[str, Any], n_results: int) -> None:
            raw = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                where=where,
            )
            structured_results.extend(self._process_query_results(raw, min_score=0.0))

        if grain == "monthly" and month and (wants_total or wants_breakdown):
            if wants_total:
                query_where(
                    {
                        "$and": [
                            {"origin": "database"},
                            {"source_table": "stats_operational_monthly"},
                            {"equipment_id": equipment_id},
                            {"year": year},
                            {"month": month},
                        ]
                    },
                    n_results=1,
                )
            if wants_breakdown:
                breakdown_filters = [
                    {"origin": "database"},
                    {"source_table": "stats_abnormal_monthly"},
                    {"equipment_id": equipment_id},
                    {"year": year},
                    {"month": month},
                ]
                if filter_anomaly_type and anomaly_type:
                    breakdown_filters.append({"detected_anomaly_type": anomaly_type})
                query_where(
                    {
                        "$and": breakdown_filters
                    },
                    n_results=1 if (filter_anomaly_type and anomaly_type) else 10,
                )

        if grain == "quarterly" and quarter and (wants_total or wants_breakdown):
            if wants_total:
                query_where(
                    {
                        "$and": [
                            {"origin": "database"},
                            {"source_table": "stats_operational_quarterly"},
                            {"equipment_id": equipment_id},
                            {"year": year},
                            {"quarter": quarter},
                        ]
                    },
                    n_results=1,
                )
            if wants_breakdown:
                breakdown_filters = [
                    {"origin": "database"},
                    {"source_table": "stats_abnormal_quarterly"},
                    {"equipment_id": equipment_id},
                    {"year": year},
                    {"quarter": quarter},
                ]
                if filter_anomaly_type and anomaly_type:
                    breakdown_filters.append({"detected_anomaly_type": anomaly_type})
                query_where(
                    {
                        "$and": breakdown_filters
                    },
                    n_results=1 if (filter_anomaly_type and anomaly_type) else 10,
                )

        if grain == "yearly" and (wants_total or wants_breakdown):
            if wants_total:
                query_where(
                    {
                        "$and": [
                            {"origin": "database"},
                            {"source_table": "stats_operational_yearly"},
                            {"equipment_id": equipment_id},
                            {"year": year},
                        ]
                    },
                    n_results=1,
                )
            if wants_breakdown:
                breakdown_filters = [
                    {"origin": "database"},
                    {"source_table": "stats_abnormal_yearly"},
                    {"equipment_id": equipment_id},
                    {"year": year},
                ]
                if filter_anomaly_type and anomaly_type:
                    breakdown_filters.append({"detected_anomaly_type": anomaly_type})
                query_where(
                    {
                        "$and": breakdown_filters
                    },
                    n_results=1 if (filter_anomaly_type and anomaly_type) else 10,
                )

        return structured_results

    def _rerank_results(
        self, query: str, results: List[RetrievalResult], *, hints: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        hints = hints or self._extract_query_hints(query)
        wants_total = bool(hints.get("wants_total"))
        wants_breakdown = bool(hints.get("wants_breakdown"))
        is_data_query = bool(hints.get("is_data_query"))
        grain = hints.get("grain")

        equipment_id = hints.get("equipment_id")
        year = hints.get("year")
        month = hints.get("month")
        quarter = hints.get("quarter")

        reranked: List[RetrievalResult] = []

        for result in results:
            score = float(result.score)
            metadata = result.document.metadata or {}

            origin = metadata.get("origin")
            if is_data_query:
                if origin == "database":
                    score += 0.12
                elif origin == "text":
                    score -= 0.04

            if origin == "database":
                table = (metadata.get("source_table") or "").lower()

                if grain == "monthly":
                    if table.endswith("_monthly"):
                        score += 0.08
                    elif table.endswith("_quarterly") or table.endswith("_yearly"):
                        score -= 0.04
                elif grain == "quarterly":
                    if table.endswith("_quarterly"):
                        score += 0.08
                    elif table.endswith("_monthly") or table.endswith("_yearly"):
                        score -= 0.04
                elif grain == "yearly":
                    if table.endswith("_yearly"):
                        score += 0.08
                    elif table.endswith("_monthly") or table.endswith("_quarterly"):
                        score -= 0.04

                if wants_total:
                    if table.startswith("stats_operational_"):
                        score += 0.18
                    elif table.startswith("stats_abnormal_") and not wants_breakdown:
                        score -= 0.06
                if wants_breakdown and table.startswith("stats_abnormal_"):
                    score += 0.18

                if equipment_id and (metadata.get("equipment_id") or "").upper() == equipment_id:
                    score += 0.22
                if year and metadata.get("year") == year:
                    score += 0.06
                if month and metadata.get("month") == month:
                    score += 0.06
                if quarter and metadata.get("quarter") == quarter:
                    score += 0.06

            if score < 0.0:
                score = 0.0
            reranked.append(RetrievalResult(document=result.document, score=score))

        def sort_key(item: RetrievalResult) -> tuple:
            metadata = item.document.metadata or {}
            origin = metadata.get("origin")
            table = (metadata.get("source_table") or "").lower() if origin == "database" else ""

            preference = 0
            origin_rank = 1 if origin == "database" else 0
            if origin == "database":
                if wants_total and table.startswith("stats_operational_"):
                    preference += 2
                if wants_breakdown and table.startswith("stats_abnormal_"):
                    preference += 2

            return (float(item.score), preference, origin_rank)

        reranked.sort(key=sort_key, reverse=True)
        return reranked

    def _distance_to_score(self, distance: float) -> float:
        space = None
        try:
            if self.collection.metadata:
                space = self.collection.metadata.get("hnsw:space")
        except Exception:
            space = None

        if space == "cosine":
            score = 1.0 - float(distance)
        else:
            # Fallback: monotonic mapping for L2/IP distances.
            score = 1.0 / (1.0 + float(distance))

        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score

    def _process_query_results(self, results: Dict, min_score: float) -> List[RetrievalResult]:
        """Helper to convert ChromaDB query results to RetrievalResult objects."""
        processed: List[RetrievalResult] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return processed

        for doc_id, content, metadata, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            score = self._distance_to_score(distance)
            logger.info(f"Retrieved doc_id={doc_id} with distance={distance}, score={score}")
            if score >= min_score:
                doc = KnowledgeDocument(doc_id=doc_id, content=content, metadata=metadata)
                processed.append(RetrievalResult(document=doc, score=score))
        return processed

    def _resolve_sources(self, source_paths: Optional[Sequence[os.PathLike[str] | str]]) -> Sequence[Path]:
        if not source_paths:
            book_dir = self.project_root / "src_book"
            defaults = [
                self.project_root / "Documentary.md", self.project_root / "README.md",
                self.project_root / "src", self.project_root / "templates",
            ]
            if book_dir.exists():
                defaults.append(book_dir)
            return tuple(p for p in defaults if p.exists())

        resolved = []
        for raw in source_paths:
            path = Path(raw) if Path(raw).is_absolute() else self.project_root / raw
            if path.exists():
                resolved.append(path)
            else:
                logger.warning("Skipping unavailable RAG source: %s", path)
        return tuple(resolved)

    def _sync_documents(self) -> None:
        """Sync filesystem documents incrementally."""
        state = self._load_state()
        file_state = state.get("files", {})
        new_file_state = {}

        files_processed: Set[str] = set()
        docs_to_add: List[KnowledgeDocument] = []
        files_to_remove_from_db: List[str] = []  # list of relative paths

        for file_path in self._iter_source_files(self._source_paths):
            try:
                relative_path = str(file_path.relative_to(self.project_root))
            except ValueError:
                relative_path = str(file_path)

            files_processed.add(relative_path)

            try:
                stat = file_path.stat()
                mtime = stat.st_mtime
                # Simple change detection: mtime
                # Note: If you want to be more robust against "touch", include size or content hash

                prev_info = file_state.get(relative_path)

                if prev_info and prev_info.get("mtime") == mtime:
                    # File unchanged, keep in state
                    new_file_state[relative_path] = prev_info
                    continue

                # File changed or new
                logger.info(f"Detected change in file: {relative_path}")
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                chunks = [
                    chunk.strip()
                    for chunk in self._split_into_chunks(text.replace("\r\n", "\n"))
                    if chunk.strip()
                ]

                if not chunks:
                    continue

                # If file existed before, we must remove its old chunks first
                if prev_info:
                    files_to_remove_from_db.append(relative_path)

                for idx, chunk in enumerate(chunks, 1):
                    metadata = {
                        "source": relative_path,
                        "chunk_index": str(idx),
                        "chunk_count": str(len(chunks)),
                        "origin": "text",
                        "source_type": "filesystem",
                    }
                    doc_id = f"{relative_path}::chunk-{idx}"
                    docs_to_add.append(KnowledgeDocument(doc_id, chunk, metadata))

                new_file_state[relative_path] = {"mtime": mtime}

            except OSError as e:
                logger.warning("Failed to process file %s: %s", file_path, e)
                continue

        # Identify deleted files
        for old_path in file_state:
            if old_path not in files_processed:
                logger.info(f"File deleted: {old_path}")
                files_to_remove_from_db.append(old_path)

        with self._lock:
            # 1. Remove old chunks for changed/deleted files
            for rel_path in files_to_remove_from_db:
                try:
                    self.collection.delete(where={"source": rel_path})
                except Exception as e:
                    logger.warning(f"Failed to delete chunks for {rel_path}: {e}")

            # 2. Add new chunks
            if docs_to_add:
                logger.info(f"Upserting {len(docs_to_add)} chunks from changed files.")
                self._add_documents_to_collection(docs_to_add)

            # 3. Update state
            state["files"] = new_file_state
            self._save_state(state)

    def _add_documents_to_collection(self, docs: List[KnowledgeDocument], batch_size: int = 128) -> None:
        """Embeds and adds documents to the Chroma collection in batches."""
        if not docs:
            return

        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            contents = [d.content for d in batch]
            ids = [d.doc_id for d in batch]
            metadatas = [d.metadata for d in batch]

            embeddings = self.model.encode(
                contents,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings.cpu().numpy().tolist(),
                metadatas=metadatas,
                documents=contents,
            )

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
            return file_path.stat().st_size <= self.max_file_size
        except OSError:
            return False

    def _split_into_chunks(self, text: str) -> Iterable[str]:
        if len(text) <= self.chunk_size:
            yield text
            return

        start = 0
        while start < len(text):
            end = start + self.chunk_size
            yield text[start:end]
            start += self.chunk_size - self.chunk_overlap


_default_kb: Optional[RAGKnowledgeBase] = None
_default_kb_lock = threading.Lock()


def get_default_knowledge_base() -> RAGKnowledgeBase:
    """Return a lazily instantiated singleton knowledge base."""
    global _default_kb
    if _default_kb is None:
        with _default_kb_lock:
            if _default_kb is None:
                _default_kb = RAGKnowledgeBase()
    return _default_kb
