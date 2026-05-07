"""
Tests 62-66 — core/entity_extractor.py
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def _make_extractor(return_value=None):
    """Return an EntityExtractor whose underlying strategy is fully mocked."""
    if return_value is None:
        return_value = []
    mock_strategy = MagicMock()
    mock_strategy.extract.return_value = return_value

    # Patch where EntityExtractionFactory is used inside core.entity_extractor
    with patch("core.entity_extractor.EntityExtractionFactory.get_extractor",
               return_value=mock_strategy):
        from rag_sdk.core.entity_extractor import EntityExtractor
        extractor = EntityExtractor(provider="llm")
    return extractor, mock_strategy


# ---------------------------------------------------------------------------
# Test 62
# ---------------------------------------------------------------------------
def test_extract_delegates_to_strategy():
    expected = [{"entity": "Alice", "type": "PERSON", "relations": []}]
    extractor, mock_strategy = _make_extractor(return_value=expected)
    result = extractor.extract("Alice was here.")
    mock_strategy.extract.assert_called_once_with("Alice was here.")
    assert result == expected


# ---------------------------------------------------------------------------
# Test 63
# ---------------------------------------------------------------------------
def test_extract_from_chunks_returns_dict():
    extractor, _ = _make_extractor([{"entity": "X", "type": "OTHER", "relations": []}])
    chunks = [Document(page_content="text one"), Document(page_content="text two")]
    result = extractor.extract_from_chunks(chunks)
    assert isinstance(result, dict)
    assert set(result.keys()) == {0, 1}


# ---------------------------------------------------------------------------
# Test 64
# ---------------------------------------------------------------------------
def test_extract_from_chunks_correct_index_mapping():
    extractor, mock_strategy = _make_extractor()
    chunks = [
        Document(page_content="alpha"),
        Document(page_content="beta"),
        Document(page_content="gamma"),
    ]
    extractor.extract_from_chunks(chunks)
    calls = [c.args[0] for c in mock_strategy.extract.call_args_list]
    assert calls == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# Test 65
# ---------------------------------------------------------------------------
def test_extract_from_chunks_empty_list():
    extractor, mock_strategy = _make_extractor()
    result = extractor.extract_from_chunks([])
    assert result == {}
    mock_strategy.extract.assert_not_called()


# ---------------------------------------------------------------------------
# Test 66
# ---------------------------------------------------------------------------
def test_extract_from_chunks_calls_strategy_with_chunk_content():
    extractor, mock_strategy = _make_extractor()
    chunks = [Document(page_content="hello world", metadata={"source": "f.txt"})]
    extractor.extract_from_chunks(chunks)
    mock_strategy.extract.assert_called_once_with("hello world")
