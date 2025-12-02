"""
Vector-based retrieval-augmented generation utilities for the LINE bot project.
Replaces the original TF-IDF implementation with a sentence-transformer and ChromaDB backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import chromadb
import pyodbc
import torch
from sentence_transformers import SentenceTransformer

from .database import db
from . import database
from .utils import _format_value

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

        # Setup Sentence Transformer model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = embedding_model_name or self.DEFAULT_EMBEDDING_MODEL
        logger.info("Initializing SentenceTransformer model '%s' on device '%s'.", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)

        # Setup ChromaDB
        _chroma_path = chroma_path or self.DEFAULT_CHROMA_PATH
        self._collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        self.chroma_client = chromadb.PersistentClient(path=_chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(self._collection_name)

        self._state_file = Path(_chroma_path) / "rag_state.json"

        # Resolve source paths
        env_sources = os.getenv("RAG_SOURCE_PATHS")
        book_dir = self.project_root / "src_book"
        if source_paths is None:
            if book_dir.exists():
                source_paths = [p for p in book_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
            elif env_sources:
                source_paths = [p for p in env_sources.split(os.pathsep) if p]
        self._source_paths: Sequence[Path] = self._resolve_sources(source_paths)

        self._enable_db_ingestion = enable_db_ingestion and os.getenv(
            "ENABLE_RAG_DB", "true"
        ).lower() not in {"false", "0", "no"}
        self._default_db_top_k = int(os.getenv("RAG_DB_TOP_K", "3"))

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

    def _load_state(self) -> Dict[str, Dict]:
        if self._state_file.exists():
            try:
                with open(self._state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load RAG state file: {e}. Resetting state.")
        return {}

    def _save_state(self, state: Dict[str, Dict]) -> None:
        try:
            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save RAG state file: {e}")

    def _get_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

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
                "text_columns": ["detected_anomaly_type", "severity_level", "resolution_notes"],
                "meta_columns": ["equipment_id", "is_resolved"],
            },
        ]

        state = self._load_state()
        db_state = state.get("database", {})
        new_db_state = {}
        docs_to_add: List[KnowledgeDocument] = []
        ids_processed: Set[str] = set()

        try:
            with db._get_connection() as conn:
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

        query_embedding = self.model.encode(query, convert_to_numpy=True)

        include_db = self._enable_db_ingestion if include_db_results is None else include_db_results
        db_k = db_top_k if db_top_k is not None else self._default_db_top_k

        # Query for filesystem documents
        text_results_raw = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where={"origin": "text"},
        )

        final_results: List[RetrievalResult] = self._process_query_results(text_results_raw, min_score)

        # Query for database documents if enabled
        if include_db and db_k > 0:
            db_results_raw = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=db_k,
                where={"origin": "database"},
            )
            final_results.extend(self._process_query_results(db_results_raw, min_score))

        # Sort by score descending as a final step
        final_results.sort(key=lambda item: item.score, reverse=True)
        return final_results

    def _process_query_results(self, results: Dict, min_score: float) -> List[RetrievalResult]:
        """Helper to convert ChromaDB query results to RetrievalResult objects."""
        processed: List[RetrievalResult] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return processed

        for doc_id, content, metadata, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            # Convert cosine distance to a similarity score (0 to 1, higher is better)
            # Convert L2 distance to a similarity score (0 to 1). This is a simple inversion.
            # A score of 1 is a perfect match (distance 0).
            score = 1.0 / (1.0 + distance)
            logger.info(f"Retrieved doc_id={doc_id} with distance={distance}, score={score}")
            if score >= min_score:
                doc = KnowledgeDocument(doc_id=doc_id, content=content, metadata=metadata)
                processed.append(RetrievalResult(document=doc, score=score))
        return processed

    def _resolve_sources(self, source_paths: Optional[Sequence[os.PathLike[str] | str]]) -> Sequence[Path]:
        if not source_paths:
            defaults = [
                self.project_root / "Documentary.md", self.project_root / "README.md",
                self.project_root / "src", self.project_root / "templates",
            ]
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

            embeddings = self.model.encode(contents, convert_to_tensor=True, show_progress_bar=False)

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
