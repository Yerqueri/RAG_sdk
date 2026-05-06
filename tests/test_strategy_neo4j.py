"""
Tests 34-48 — strategies/graph_store/neo4j_strategy.py

The Neo4j driver is fully mocked; no running Neo4j instance is required.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_neo4j_mock():
    """Build a mock neo4j module with a fake GraphDatabase.driver."""
    mock_neo4j = MagicMock()
    mock_driver = MagicMock()
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    mock_driver.session.return_value = session
    mock_neo4j.GraphDatabase.driver.return_value = mock_driver
    return mock_neo4j, mock_driver, session


def _make_neo4j_strategy():
    """Build a Neo4jStrategy with a fully mocked neo4j driver."""
    mock_neo4j, mock_driver, session = _make_neo4j_mock()
    with patch.dict(sys.modules, {"neo4j": mock_neo4j}):
        import importlib
        import strategies.graph_store.neo4j_strategy as mod
        importlib.reload(mod)
        strategy = mod.Neo4jStrategy(
            url="bolt://localhost:7687",
            username="neo4j",
            password="password",
        )
    # Swap the session on the real driver mock so calls after construction work
    strategy.driver = mock_driver
    return strategy, session


def _chunk(content: str, idx: int = 0) -> Document:
    return Document(page_content=content, metadata={"chunk_index": idx})


# ---------------------------------------------------------------------------
# Test 34
# ---------------------------------------------------------------------------
def test_neo4j_init_creates_driver():
    mock_neo4j, mock_driver, session = _make_neo4j_mock()
    with patch.dict(sys.modules, {"neo4j": mock_neo4j}):
        import importlib
        import strategies.graph_store.neo4j_strategy as mod
        importlib.reload(mod)
        mod.Neo4jStrategy("bolt://localhost:7687", "neo4j", "password")
    mock_neo4j.GraphDatabase.driver.assert_called_once_with(
        "bolt://localhost:7687", auth=("neo4j", "password")
    )


# ---------------------------------------------------------------------------
# Test 35
# ---------------------------------------------------------------------------
def test_neo4j_init_creates_constraints():
    _, _, session = _make_neo4j_mock()
    strategy, session = _make_neo4j_strategy()
    # _create_constraints runs session.run 3 times
    assert session.run.call_count >= 3


# ---------------------------------------------------------------------------
# Test 36
# ---------------------------------------------------------------------------
def test_store_graph_creates_document_node():
    strategy, session = _make_neo4j_strategy()
    session.run.reset_mock()
    strategy.store_graph("doc1", [_chunk("hello")], {0: []})
    calls_text = " ".join(str(c) for c in session.run.call_args_list)
    assert "Document" in calls_text or "MERGE" in calls_text


# ---------------------------------------------------------------------------
# Test 37
# ---------------------------------------------------------------------------
def test_store_graph_creates_chunk_nodes():
    strategy, session = _make_neo4j_strategy()
    session.run.reset_mock()
    strategy.store_graph("docX", [_chunk("chunk A"), _chunk("chunk B")], {})
    calls_text = " ".join(str(c) for c in session.run.call_args_list)
    assert "Chunk" in calls_text


# ---------------------------------------------------------------------------
# Test 38
# ---------------------------------------------------------------------------
def test_store_graph_links_document_to_chunk():
    strategy, session = _make_neo4j_strategy()
    session.run.reset_mock()
    strategy.store_graph("docY", [_chunk("text")], {})
    calls_text = " ".join(str(c) for c in session.run.call_args_list)
    assert "CONTAINS" in calls_text


# ---------------------------------------------------------------------------
# Test 39
# ---------------------------------------------------------------------------
def test_store_graph_creates_entity_nodes():
    strategy, session = _make_neo4j_strategy()
    session.run.reset_mock()
    entities = {0: [{"entity": "Alice", "type": "PERSON", "relations": []}]}
    strategy.store_graph("doc1", [_chunk("Alice lives here")], entities)
    calls_text = " ".join(str(c) for c in session.run.call_args_list)
    assert "Entity" in calls_text


# ---------------------------------------------------------------------------
# Test 40
# ---------------------------------------------------------------------------
def test_store_graph_creates_mentions_edges():
    strategy, session = _make_neo4j_strategy()
    session.run.reset_mock()
    entities = {0: [{"entity": "Bob", "type": "PERSON", "relations": []}]}
    strategy.store_graph("doc2", [_chunk("Bob is here")], entities)
    calls_text = " ".join(str(c) for c in session.run.call_args_list)
    assert "MENTIONS" in calls_text


# ---------------------------------------------------------------------------
# Test 41
# ---------------------------------------------------------------------------
def test_store_graph_creates_related_to_edges():
    strategy, session = _make_neo4j_strategy()
    session.run.reset_mock()
    entities = {0: [{
        "entity": "Acme", "type": "ORG",
        "relations": [{"target": "CEO", "relation": "HAS"}],
    }]}
    strategy.store_graph("doc3", [_chunk("Acme has a CEO")], entities)
    calls_text = " ".join(str(c) for c in session.run.call_args_list)
    assert "RELATED_TO" in calls_text


# ---------------------------------------------------------------------------
# Test 42
# ---------------------------------------------------------------------------
def test_store_graph_creates_next_edges_between_chunks():
    strategy, session = _make_neo4j_strategy()
    session.run.reset_mock()
    chunks = [_chunk("first"), _chunk("second"), _chunk("third")]
    strategy.store_graph("docN", chunks, {})
    calls_text = " ".join(str(c) for c in session.run.call_args_list)
    assert "NEXT" in calls_text


# ---------------------------------------------------------------------------
# Test 43
# ---------------------------------------------------------------------------
def test_store_graph_skips_empty_entity_name():
    strategy, session = _make_neo4j_strategy()
    session.run.reset_mock()
    entities = {0: [{"entity": "", "type": "OTHER", "relations": []}]}
    strategy.store_graph("doc4", [_chunk("text")], entities)
    calls_text = " ".join(str(c) for c in session.run.call_args_list)
    assert "MENTIONS" not in calls_text


# ---------------------------------------------------------------------------
# Test 44
# ---------------------------------------------------------------------------
def test_store_graph_handles_no_entities():
    strategy, session = _make_neo4j_strategy()
    session.run.reset_mock()
    strategy.store_graph("doc5", [_chunk("some text")], {})


# ---------------------------------------------------------------------------
# Test 45
# ---------------------------------------------------------------------------
def test_store_graph_handles_multiple_chunks():
    strategy, session = _make_neo4j_strategy()
    session.run.reset_mock()
    chunks = [_chunk(f"text {i}") for i in range(5)]
    strategy.store_graph("doc6", chunks, {})
    calls_text = " ".join(str(c) for c in session.run.call_args_list)
    assert "NEXT" in calls_text


# ---------------------------------------------------------------------------
# Test 46
# ---------------------------------------------------------------------------
def test_get_related_chunks_runs_query():
    strategy, session = _make_neo4j_strategy()
    mock_record = {"content": "chunk content"}
    session.run.return_value = iter([mock_record])
    result = strategy.get_related_chunks(["Alice"], k=3)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Test 47
# ---------------------------------------------------------------------------
def test_get_related_chunks_empty_names_returns_empty():
    strategy, _ = _make_neo4j_strategy()
    result = strategy.get_related_chunks([], k=5)
    assert result == []


# ---------------------------------------------------------------------------
# Test 48
# ---------------------------------------------------------------------------
def test_get_related_chunks_respects_k():
    strategy, session = _make_neo4j_strategy()
    session.run.return_value = iter([])
    strategy.get_related_chunks(["entity"], k=7)
    last_call = session.run.call_args
    assert last_call.kwargs.get("k") == 7 or (last_call.args and 7 in last_call.args)


# ---------------------------------------------------------------------------
# Test 48b — close()
# ---------------------------------------------------------------------------
def test_neo4j_close_calls_driver_close():
    strategy, _ = _make_neo4j_strategy()
    strategy.close()
    strategy.driver.close.assert_called_once()

