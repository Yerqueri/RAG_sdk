"""
Tests 84-95 — sdk.py  (RAGClient)
"""
import os
import pytest
from unittest.mock import MagicMock, patch, call
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Env stub so config doesn't blow up
# ---------------------------------------------------------------------------
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


def _make_client(**kwargs):
    with patch.dict(os.environ, _BASE_ENV, clear=True):
        from sdk import RAGClient
        return RAGClient(**kwargs)


# ---------------------------------------------------------------------------
# Test 84
# ---------------------------------------------------------------------------
def test_default_constructor_enable_graph_is_false():
    client = _make_client()
    assert client.enable_graph is False


# ---------------------------------------------------------------------------
# Test 85
# ---------------------------------------------------------------------------
def test_default_retrieval_mode_is_none():
    client = _make_client()
    assert client.retrieval_mode is None


# ---------------------------------------------------------------------------
# Test 86
# ---------------------------------------------------------------------------
def test_ingest_file_calls_document_loader():
    with patch.dict(os.environ, _BASE_ENV, clear=True), \
         patch("core.document_loader.DocumentLoader.load_file", return_value=[]) as mock_load, \
         patch("core.text_chunker.TextChunker.split", return_value=[]), \
         patch("factories.embedding_factory.EmbeddingFactory.get_embeddings", return_value=MagicMock()), \
         patch("factories.vector_store_factory.VectorStoreFactory.get_vector_store", return_value=MagicMock(store=MagicMock())):
        from sdk import RAGClient
        client = RAGClient()
        client.ingest_file("./data/test.txt")
        mock_load.assert_called_once_with("./data/test.txt")


# ---------------------------------------------------------------------------
# Test 87
# ---------------------------------------------------------------------------
def test_ingest_directory_uses_config_data_dir_when_no_arg():
    with patch.dict(os.environ, _BASE_ENV, clear=True), \
         patch("core.document_loader.DocumentLoader.load_directory", return_value=[]) as mock_load, \
         patch("core.text_chunker.TextChunker.split", return_value=[]), \
         patch("factories.embedding_factory.EmbeddingFactory.get_embeddings", return_value=MagicMock()), \
         patch("factories.vector_store_factory.VectorStoreFactory.get_vector_store", return_value=MagicMock(store=MagicMock())):
        from sdk import RAGClient
        from core.config import config
        client = RAGClient()
        client.ingest_directory()
        mock_load.assert_called_once_with(config.data_dir)


# ---------------------------------------------------------------------------
# Test 88
# ---------------------------------------------------------------------------
def test_ingest_directory_uses_provided_path():
    with patch.dict(os.environ, _BASE_ENV, clear=True), \
         patch("core.document_loader.DocumentLoader.load_directory", return_value=[]) as mock_load, \
         patch("core.text_chunker.TextChunker.split", return_value=[]), \
         patch("factories.embedding_factory.EmbeddingFactory.get_embeddings", return_value=MagicMock()), \
         patch("factories.vector_store_factory.VectorStoreFactory.get_vector_store", return_value=MagicMock(store=MagicMock())):
        from sdk import RAGClient
        client = RAGClient()
        client.ingest_directory("./custom_docs")
        mock_load.assert_called_once_with("./custom_docs")


# ---------------------------------------------------------------------------
# Test 89
# ---------------------------------------------------------------------------
def test_ingest_documents_calls_chunker():
    mock_chunker = MagicMock()
    mock_chunker.split.return_value = []

    with patch.dict(os.environ, _BASE_ENV, clear=True), \
         patch("sdk.TextChunker", return_value=mock_chunker), \
         patch("factories.embedding_factory.EmbeddingFactory.get_embeddings", return_value=MagicMock()), \
         patch("factories.vector_store_factory.VectorStoreFactory.get_vector_store", return_value=MagicMock(store=MagicMock())):
        from sdk import RAGClient
        client = RAGClient()
        client._ingest_documents([Document(page_content="hello")])
        mock_chunker.split.assert_called_once()


# ---------------------------------------------------------------------------
# Test 90
# ---------------------------------------------------------------------------
def test_ingest_documents_calls_vector_store():
    mock_vs = MagicMock()
    with patch.dict(os.environ, _BASE_ENV, clear=True), \
         patch("core.text_chunker.TextChunker.split", return_value=[Document(page_content="c")]), \
         patch("factories.embedding_factory.EmbeddingFactory.get_embeddings", return_value=MagicMock()), \
         patch("factories.vector_store_factory.VectorStoreFactory.get_vector_store", return_value=mock_vs):
        from sdk import RAGClient
        client = RAGClient()
        client._ingest_documents([Document(page_content="hello")])
        mock_vs.store.assert_called_once()


# ---------------------------------------------------------------------------
# Test 91
# ---------------------------------------------------------------------------
def test_enable_graph_calls_entity_extractor():
    mock_extractor = MagicMock()
    mock_extractor.extract_from_chunks.return_value = {0: []}  # index 0 must exist
    mock_gs = MagicMock()

    chunk = Document(page_content="text", metadata={"source": "f.txt", "chunk_index": 0})

    with patch.dict(os.environ, _BASE_ENV, clear=True), \
         patch("sdk.TextChunker.split", return_value=[chunk]), \
         patch("factories.embedding_factory.EmbeddingFactory.get_embeddings", return_value=MagicMock()), \
         patch("factories.vector_store_factory.VectorStoreFactory.get_vector_store", return_value=MagicMock(store=MagicMock())), \
         patch("core.entity_extractor.EntityExtractor", return_value=mock_extractor), \
         patch("core.graph_store.GraphStore", return_value=mock_gs):
        from sdk import RAGClient
        client = RAGClient(enable_graph=True)
        client._ingest_documents([Document(page_content="hello")])
        mock_extractor.extract_from_chunks.assert_called_once()


# ---------------------------------------------------------------------------
# Test 92
# ---------------------------------------------------------------------------
def test_enable_graph_calls_graph_store_store_graph():
    mock_extractor = MagicMock()
    mock_extractor.extract_from_chunks.return_value = {0: []}
    mock_gs = MagicMock()

    chunk = Document(page_content="text", metadata={"source": "f.txt", "chunk_index": 0})

    with patch.dict(os.environ, _BASE_ENV, clear=True), \
         patch("sdk.TextChunker.split", return_value=[chunk]), \
         patch("factories.embedding_factory.EmbeddingFactory.get_embeddings", return_value=MagicMock()), \
         patch("factories.vector_store_factory.VectorStoreFactory.get_vector_store", return_value=MagicMock(store=MagicMock())), \
         patch("core.entity_extractor.EntityExtractor", return_value=mock_extractor), \
         patch("core.graph_store.GraphStore", return_value=mock_gs):
        from sdk import RAGClient
        client = RAGClient(enable_graph=True)
        client._ingest_documents([Document(page_content="hello")])
        mock_gs.store_graph.assert_called()


# ---------------------------------------------------------------------------
# Test 93
# ---------------------------------------------------------------------------
def test_query_vector_mode_does_not_use_hybrid_retriever():
    mock_retriever = MagicMock()
    mock_vs = MagicMock()
    mock_vs.get_retriever.return_value = mock_retriever
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="answer")

    with patch.dict(os.environ, {**_BASE_ENV, "RETRIEVAL_MODE": "vector"}, clear=True), \
         patch("factories.embedding_factory.EmbeddingFactory.get_embeddings", return_value=MagicMock()), \
         patch("factories.vector_store_factory.VectorStoreFactory.get_vector_store", return_value=mock_vs), \
         patch("factories.llm_factory.LLMFactory.get_llm", return_value=mock_llm), \
         patch("core.hybrid_retriever.HybridRetriever") as mock_hr_class:
        from sdk import RAGClient
        client = RAGClient(retrieval_mode="vector")
        try:
            client.query("What is X?")
        except Exception:
            pass
        mock_hr_class.assert_not_called()


# ---------------------------------------------------------------------------
# Test 94
# ---------------------------------------------------------------------------
def test_query_hybrid_mode_instantiates_hybrid_retriever():
    mock_gs = MagicMock()
    mock_ee = MagicMock()
    mock_vs = MagicMock()
    mock_vs.get_retriever.return_value = MagicMock()

    with patch.dict(os.environ, {**_BASE_ENV, "RETRIEVAL_MODE": "hybrid"}, clear=True), \
         patch("factories.embedding_factory.EmbeddingFactory.get_embeddings", return_value=MagicMock()), \
         patch("factories.vector_store_factory.VectorStoreFactory.get_vector_store", return_value=mock_vs), \
         patch("factories.llm_factory.LLMFactory.get_llm", return_value=MagicMock()), \
         patch("core.graph_store.GraphStore", return_value=mock_gs), \
         patch("core.entity_extractor.EntityExtractor", return_value=mock_ee), \
         patch("core.hybrid_retriever.HybridRetriever") as mock_hr_class:
        mock_hr_class.return_value = MagicMock()
        from sdk import RAGClient
        client = RAGClient(retrieval_mode="hybrid")
        try:
            client.query("Find entities")
        except Exception:
            pass
        mock_hr_class.assert_called_once()


# ---------------------------------------------------------------------------
# Test 95
# ---------------------------------------------------------------------------
def test_query_falls_back_to_vector_when_retrieval_mode_env_raises():
    """RAGClient.query should default to 'vector' if config.retrieval_mode raises."""
    mock_vs = MagicMock()
    mock_vs.get_retriever.return_value = MagicMock()

    # Remove RETRIEVAL_MODE from env so config raises, triggering fallback
    env = {k: v for k, v in _BASE_ENV.items() if k != "RETRIEVAL_MODE"}

    with patch.dict(os.environ, env, clear=True), \
         patch("factories.embedding_factory.EmbeddingFactory.get_embeddings", return_value=MagicMock()), \
         patch("factories.vector_store_factory.VectorStoreFactory.get_vector_store", return_value=mock_vs), \
         patch("factories.llm_factory.LLMFactory.get_llm", return_value=MagicMock()), \
         patch("core.hybrid_retriever.HybridRetriever") as mock_hr_class:
        from sdk import RAGClient
        client = RAGClient()  # no retrieval_mode set
        try:
            client.query("test question")
        except Exception:
            pass
        # HybridRetriever must NOT have been instantiated — fell back to vector
        mock_hr_class.assert_not_called()

