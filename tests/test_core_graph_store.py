"""
Tests 67-71 — core/graph_store.py
"""
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def _make_graph_store():
    mock_strategy = MagicMock()
    # Patch where GraphStoreFactory is imported inside core.graph_store
    with patch("core.graph_store.GraphStoreFactory.get_graph_store",
               return_value=mock_strategy):
        from rag_sdk.core.graph_store import GraphStore
        gs = GraphStore(provider="neo4j")
    return gs, mock_strategy


# ---------------------------------------------------------------------------
# Test 67
# ---------------------------------------------------------------------------
def test_graph_store_store_graph_delegates():
    gs, mock_strategy = _make_graph_store()
    chunks = [Document(page_content="chunk content")]
    entities = {0: [{"entity": "Alice", "type": "PERSON", "relations": []}]}
    gs.store_graph("doc1", chunks, entities)
    mock_strategy.store_graph.assert_called_once_with("doc1", chunks, entities)


# ---------------------------------------------------------------------------
# Test 68
# ---------------------------------------------------------------------------
def test_graph_store_get_related_chunks_delegates():
    gs, mock_strategy = _make_graph_store()
    mock_strategy.get_related_chunks.return_value = ["content A"]
    result = gs.get_related_chunks(["Alice"], k=4)
    mock_strategy.get_related_chunks.assert_called_once_with(["Alice"], k=4)
    assert result == ["content A"]


# ---------------------------------------------------------------------------
# Test 69
# ---------------------------------------------------------------------------
def test_graph_store_close_delegates():
    gs, mock_strategy = _make_graph_store()
    gs.close()
    mock_strategy.close.assert_called_once()


# ---------------------------------------------------------------------------
# Test 70
# ---------------------------------------------------------------------------
def test_graph_store_init_uses_factory():
    mock_strategy = MagicMock()
    with patch("core.graph_store.GraphStoreFactory.get_graph_store",
               return_value=mock_strategy) as mock_factory:
        from rag_sdk.core.graph_store import GraphStore
        GraphStore(provider="neo4j")
        mock_factory.assert_called_once_with(provider="neo4j")


# ---------------------------------------------------------------------------
# Test 71
# ---------------------------------------------------------------------------
def test_graph_store_provider_override_passed_to_factory():
    mock_strategy = MagicMock()
    with patch("core.graph_store.GraphStoreFactory.get_graph_store",
               return_value=mock_strategy) as mock_factory:
        from rag_sdk.core.graph_store import GraphStore
        GraphStore(provider="custom_graph")
        mock_factory.assert_called_once_with(provider="custom_graph")

