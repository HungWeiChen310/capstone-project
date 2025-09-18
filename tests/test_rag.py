import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

os.environ["TESTING"] = "True"

from rag import RAGPipeline, SQLRetriever  # noqa: E402


class _StubDatabase:
    def __init__(self, documents):
        self._documents = documents

    def search_knowledge_documents(
        self,
        query,
        limit,
        table_name="knowledge_documents",
        search_fields=None,
    ):
        query = (query or "").lower()
        matches = []
        for document in self._documents:
            haystack = " ".join(
                str(document.get(field, ""))
                for field in ["title", "content", "tags"]
            )
            if query in haystack.lower():
                matches.append(document)
        return matches[:limit]


def test_sql_retriever_without_database_returns_empty():
    retriever = SQLRetriever(db_instance=None)
    assert retriever.retrieve("any query") == []


def test_rag_pipeline_injects_context_message():
    documents = [
        {
            "title": "Bearing Maintenance Guide",
            "content": (
                "Check the bearing vibration every 4 hours and log the readings."
            ),
            "tags": "maintenance",
            "source": "manual",
            "last_updated": "2024-01-01",
        },
        {
            "title": "Safety Procedures",
            "content": "Always isolate power before touching the spindle housing.",
            "tags": "safety",
            "source": "manual",
            "last_updated": "2024-01-10",
        },
    ]
    retriever = SQLRetriever(db_instance=_StubDatabase(documents))
    pipeline = RAGPipeline(retriever, max_documents=1, max_characters=80)
    base_messages = [
        {"role": "system", "content": "Base instruction"},
        {"role": "user", "content": "How do I monitor bearings?"},
    ]
    augmented = pipeline.inject_context(base_messages, "vibration", language="en")
    assert len(augmented) == len(base_messages) + 1
    assert augmented[1]["role"] == "system"
    assert "knowledge base excerpts" in augmented[1]["content"].lower()
    assert "bearing maintenance guide" in augmented[1]["content"].lower()
