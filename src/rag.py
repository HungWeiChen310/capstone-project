"""Simple retrieval-augmented generation utilities for the LINE bot project."""
from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


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
    """Loads local project files and exposes a TF-IDF based retriever."""

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
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必須為正整數")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能為負數")
        if chunk_overlap >= chunk_size:
            logger.warning(
                "chunk_overlap (%s) 大於或等於 chunk_size (%s)，將自動調整", chunk_overlap, chunk_size
            )
            chunk_overlap = max(0, chunk_size // 4)

        self.project_root = Path(__file__).resolve().parent.parent
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_size = max_file_size
        self._documents: List[KnowledgeDocument] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_term_frequencies: List[Counter[str]] = []
        self._doc_vectors: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}
        self._lock = threading.Lock()

        env_sources = os.getenv("RAG_SOURCE_PATHS")
        if source_paths is None and env_sources:
            source_paths = [path for path in env_sources.split(os.pathsep) if path]

        self._source_paths: Sequence[Path] = self._resolve_sources(source_paths)
        self._load_documents()

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

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float = 0.05,
    ) -> List[RetrievalResult]:
        query = (query or "").strip()
        if not query or not self._documents:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        query_vector = self._vector_from_tokens(tokens)
        if not query_vector:
            return []

        scores: List[RetrievalResult] = []
        for document, doc_vector in zip(self._documents, self._doc_vectors):
            if not doc_vector:
                continue
            score = sum(query_vector.get(term, 0.0) * weight for term, weight in doc_vector.items())
            if score >= min_score:
                scores.append(RetrievalResult(document=document, score=score))

        scores.sort(key=lambda item: item.score, reverse=True)
        return scores[:top_k]

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
                logger.debug("忽略不存在的 RAG 來源：%s", path)
        return tuple(resolved)

    def _load_documents(self) -> None:
        documents: List[KnowledgeDocument] = []
        doc_tokens: List[List[str]] = []

        for file_path in self._iter_source_files(self._source_paths):
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                logger.warning("讀取檔案 %s 失敗：%s", file_path, exc)
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
                }
                doc_id = f"{relative_path}::chunk-{idx}"
                documents.append(KnowledgeDocument(doc_id=doc_id, content=chunk, metadata=metadata))
                doc_tokens.append(self._tokenize(chunk))

        self._documents = documents
        self._doc_tokens = doc_tokens
        self._rebuild_index()
        if documents:
            logger.info("RAG 知識庫已載入 %s 份文件分片", len(documents))
        else:
            logger.warning("RAG 知識庫未載入任何文件，請確認來源設定")

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
                logger.debug("跳過過大的檔案：%s", file_path)
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
        logger.warning("無法解析整數值 %s，將使用預設值 %s", raw_value, default)
        return default
    return value
