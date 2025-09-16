import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rag import RAGKnowledgeBase  # noqa: E402


@pytest.fixture
def sample_kb(tmp_path):
    doc_path = tmp_path / "knowledge.md"
    doc_path.write_text(
        "本專案使用 Flask 建立 Web API。\n\n"
        "系統也整合 SQL Server 與 OpenAI 服務，現在加入了文件檢索 (RAG)。",
        encoding="utf-8",
    )
    return RAGKnowledgeBase(source_paths=[doc_path], chunk_size=120, chunk_overlap=20)


def test_rag_returns_related_chunks(sample_kb):
    results = sample_kb.search("Flask", top_k=3)
    assert results, "應該能夠找到與 Flask 相關的內容"
    assert any("Flask" in result.document.content for result in results)


def test_rag_handles_unknown_query(sample_kb):
    results = sample_kb.search("Nonexistent topic", top_k=3)
    assert results == []
