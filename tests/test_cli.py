"""
Tests 96-100 — CLI entry points: ingest.py  &  query.py
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, call


_BASE_ENV = {
    "LLM_PROVIDER": "openai",
    "EMBEDDING_PROVIDER": "openai",
    "VECTOR_DB_PROVIDER": "qdrant",
    "QDRANT_URL": "http://localhost:6333",
    "DATA_DIR": "./data",
    "COLLECTION_NAME": "test",
    "GRAPH_STORE_PROVIDER": "neo4j",
    "NEO4J_URL": "bolt://localhost:7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "ENTITY_EXTRACTION_PROVIDER": "llm",
    "RETRIEVAL_MODE": "vector",
}


# ---------------------------------------------------------------------------
# Test 96 — ingest.py (no --file) calls ingest_directory()
# ---------------------------------------------------------------------------
def test_ingest_cli_no_file_calls_ingest_directory():
    mock_client = MagicMock()
    with patch.dict(os.environ, _BASE_ENV, clear=True), \
         patch("sys.argv", ["ingest.py"]), \
         patch("sdk.RAGClient", return_value=mock_client):
        import importlib
        import ingest as ingest_mod
        importlib.reload(ingest_mod)
        ingest_mod.main()
        mock_client.ingest_directory.assert_called_once()
        mock_client.ingest_file.assert_not_called()


# ---------------------------------------------------------------------------
# Test 97 — ingest.py --file passes path to ingest_file()
# ---------------------------------------------------------------------------
def test_ingest_cli_with_file_calls_ingest_file():
    mock_client = MagicMock()
    with patch.dict(os.environ, _BASE_ENV, clear=True), \
         patch("sys.argv", ["ingest.py", "--file", "./data/report.pdf"]), \
         patch("sdk.RAGClient", return_value=mock_client):
        import importlib
        import ingest as ingest_mod
        importlib.reload(ingest_mod)
        ingest_mod.main()
        mock_client.ingest_file.assert_called_once_with("./data/report.pdf")
        mock_client.ingest_directory.assert_not_called()


# ---------------------------------------------------------------------------
# Test 98 — ingest.py --enable-graph sets enable_graph=True on RAGClient
# ---------------------------------------------------------------------------
def test_ingest_cli_enable_graph_flag():
    captured = {}

    def fake_rag_client(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch.dict(os.environ, _BASE_ENV, clear=True), \
         patch("sys.argv", ["ingest.py", "--enable-graph"]), \
         patch("sdk.RAGClient", side_effect=fake_rag_client):
        import importlib
        import ingest as ingest_mod
        importlib.reload(ingest_mod)
        ingest_mod.main()
    assert captured.get("enable_graph") is True


# ---------------------------------------------------------------------------
# Test 99 — query.py passes question to client.query()
# ---------------------------------------------------------------------------
def test_query_cli_passes_question_to_client():
    mock_client = MagicMock()
    mock_client.query.return_value = "An answer"
    with patch.dict(os.environ, _BASE_ENV, clear=True), \
         patch("sys.argv", ["query.py", "What is GraphRAG?"]), \
         patch("sdk.RAGClient", return_value=mock_client):
        import importlib
        import query as query_mod
        importlib.reload(query_mod)
        query_mod.main()
        mock_client.query.assert_called_once_with("What is GraphRAG?")


# ---------------------------------------------------------------------------
# Test 100 — query.py --retrieval-mode passed to RAGClient constructor
# ---------------------------------------------------------------------------
def test_query_cli_retrieval_mode_flag():
    captured = {}

    def fake_rag_client(**kwargs):
        captured.update(kwargs)
        m = MagicMock()
        m.query.return_value = "ok"
        return m

    with patch.dict(os.environ, _BASE_ENV, clear=True), \
         patch("sys.argv", ["query.py", "Some question", "--retrieval-mode", "hybrid"]), \
         patch("sdk.RAGClient", side_effect=fake_rag_client):
        import importlib
        import query as query_mod
        importlib.reload(query_mod)
        query_mod.main()
    assert captured.get("retrieval_mode") == "hybrid"

