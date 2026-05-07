"""
Tests 72-83 — core/hybrid_retriever.py
"""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from rag_sdk.core.hybrid_retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_manager():
    return MagicMock(spec=CallbackManagerForRetrieverRun)


def _make_vector_retriever(docs=None):
    mock = MagicMock()
    mock.invoke.return_value = docs or []
    return mock


def _make_graph_store(chunks=None):
    mock = MagicMock()
    mock.get_related_chunks.return_value = chunks or []
    return mock


def _make_entity_extractor(entities=None):
    mock = MagicMock()
    mock.extract.return_value = entities or []
    return mock


# ---------------------------------------------------------------------------
# Test 72
# ---------------------------------------------------------------------------
def test_vector_only_no_graph_store_returns_vector_docs():
    docs = [Document(page_content="doc A")]
    retriever = HybridRetriever(
        vector_retriever=_make_vector_retriever(docs),
        graph_store=None,
        entity_extractor=None,
    )
    result = retriever._get_relevant_documents("query", run_manager=_run_manager())
    assert result == docs


# ---------------------------------------------------------------------------
# Test 73
# ---------------------------------------------------------------------------
def test_vector_docs_returned_in_results():
    d1 = Document(page_content="result one")
    d2 = Document(page_content="result two")
    retriever = HybridRetriever(
        vector_retriever=_make_vector_retriever([d1, d2]),
        graph_store=None,
        entity_extractor=None,
    )
    result = retriever._get_relevant_documents("q", run_manager=_run_manager())
    assert len(result) == 2
    assert result[0].page_content == "result one"


# ---------------------------------------------------------------------------
# Test 74
# ---------------------------------------------------------------------------
def test_graph_only_skips_vector_when_vector_retriever_is_none():
    gs = _make_graph_store(["graph chunk"])
    ee = _make_entity_extractor([{"entity": "Alice", "type": "PERSON"}])
    retriever = HybridRetriever(
        vector_retriever=None,
        graph_store=gs,
        entity_extractor=ee,
    )
    result = retriever._get_relevant_documents("find Alice", run_manager=_run_manager())
    assert any(d.page_content == "graph chunk" for d in result)


# ---------------------------------------------------------------------------
# Test 75
# ---------------------------------------------------------------------------
def test_hybrid_deduplicates_by_content():
    shared_content = "shared chunk"
    d = Document(page_content=shared_content)
    gs = _make_graph_store([shared_content])
    ee = _make_entity_extractor([{"entity": "X"}])
    retriever = HybridRetriever(
        vector_retriever=_make_vector_retriever([d]),
        graph_store=gs,
        entity_extractor=ee,
    )
    result = retriever._get_relevant_documents("q", run_manager=_run_manager())
    contents = [r.page_content for r in result]
    assert contents.count(shared_content) == 1


# ---------------------------------------------------------------------------
# Test 76
# ---------------------------------------------------------------------------
def test_hybrid_vector_results_come_first():
    vdoc = Document(page_content="vector doc")
    gcontent = "graph doc"
    gs = _make_graph_store([gcontent])
    ee = _make_entity_extractor([{"entity": "E"}])
    retriever = HybridRetriever(
        vector_retriever=_make_vector_retriever([vdoc]),
        graph_store=gs,
        entity_extractor=ee,
    )
    result = retriever._get_relevant_documents("q", run_manager=_run_manager())
    assert result[0].page_content == "vector doc"


# ---------------------------------------------------------------------------
# Test 77
# ---------------------------------------------------------------------------
def test_graph_result_wrapped_in_document():
    gs = _make_graph_store(["graph text"])
    ee = _make_entity_extractor([{"entity": "Ent"}])
    retriever = HybridRetriever(
        vector_retriever=None,
        graph_store=gs,
        entity_extractor=ee,
    )
    result = retriever._get_relevant_documents("q", run_manager=_run_manager())
    assert isinstance(result[0], Document)
    assert result[0].metadata.get("source") == "knowledge_graph"


# ---------------------------------------------------------------------------
# Test 78
# ---------------------------------------------------------------------------
def test_graph_skipped_when_no_entity_extractor():
    gs = _make_graph_store(["should not appear"])
    retriever = HybridRetriever(
        vector_retriever=None,
        graph_store=gs,
        entity_extractor=None,   # <-- no extractor
    )
    result = retriever._get_relevant_documents("q", run_manager=_run_manager())
    assert result == []
    gs.get_related_chunks.assert_not_called()


# ---------------------------------------------------------------------------
# Test 79
# ---------------------------------------------------------------------------
def test_graph_skipped_when_graph_store_is_none():
    ee = _make_entity_extractor([{"entity": "E"}])
    retriever = HybridRetriever(
        vector_retriever=None,
        graph_store=None,        # <-- no graph store
        entity_extractor=ee,
    )
    result = retriever._get_relevant_documents("q", run_manager=_run_manager())
    assert result == []
    ee.extract.assert_not_called()


# ---------------------------------------------------------------------------
# Test 80
# ---------------------------------------------------------------------------
def test_vector_failure_does_not_raise():
    mock_vr = MagicMock()
    mock_vr.invoke.side_effect = RuntimeError("vector DB down")
    retriever = HybridRetriever(
        vector_retriever=mock_vr,
        graph_store=None,
        entity_extractor=None,
    )
    # Should not raise — failure is caught internally
    result = retriever._get_relevant_documents("q", run_manager=_run_manager())
    assert result == []


# ---------------------------------------------------------------------------
# Test 81
# ---------------------------------------------------------------------------
def test_graph_failure_does_not_raise():
    gs = MagicMock()
    gs.get_related_chunks.side_effect = RuntimeError("Neo4j down")
    ee = _make_entity_extractor([{"entity": "E"}])
    retriever = HybridRetriever(
        vector_retriever=None,
        graph_store=gs,
        entity_extractor=ee,
    )
    result = retriever._get_relevant_documents("q", run_manager=_run_manager())
    assert result == []


# ---------------------------------------------------------------------------
# Test 82
# ---------------------------------------------------------------------------
def test_empty_entity_extraction_skips_graph_query():
    gs = _make_graph_store(["should not appear"])
    ee = _make_entity_extractor([])   # empty — no entities found
    retriever = HybridRetriever(
        vector_retriever=None,
        graph_store=gs,
        entity_extractor=ee,
    )
    result = retriever._get_relevant_documents("q", run_manager=_run_manager())
    gs.get_related_chunks.assert_not_called()
    assert result == []


# ---------------------------------------------------------------------------
# Test 83
# ---------------------------------------------------------------------------
def test_k_passed_to_get_related_chunks():
    gs = _make_graph_store([])
    ee = _make_entity_extractor([{"entity": "Alice"}])
    retriever = HybridRetriever(
        vector_retriever=None,
        graph_store=gs,
        entity_extractor=ee,
        k=7,
    )
    retriever._get_relevant_documents("query", run_manager=_run_manager())
    gs.get_related_chunks.assert_called_once_with(["Alice"], k=7)

