"""
Tests 11-25 — strategies/entity_extraction/llm_entity_extraction_strategy.py
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from strategies.entity_extraction.llm_entity_extraction_strategy import (
    LLMEntityExtractionStrategy,
)


# ---------------------------------------------------------------------------
# Shared fixture: strategy with a mock LLM
# ---------------------------------------------------------------------------

def _make_strategy(response_content: str = "[]") -> LLMEntityExtractionStrategy:
    mock_llm = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = response_content
    mock_llm.invoke.return_value = mock_msg

    # LLMFactory is imported lazily inside __init__; patch at its definition site
    with patch("factories.llm_factory.LLMFactory.get_llm", return_value=mock_llm):
        strategy = LLMEntityExtractionStrategy(llm_provider="openai")
    return strategy


# ---------------------------------------------------------------------------
# Test 11
# ---------------------------------------------------------------------------
def test_extract_empty_string_returns_empty_list():
    s = _make_strategy()
    assert s.extract("") == []


# ---------------------------------------------------------------------------
# Test 12
# ---------------------------------------------------------------------------
def test_extract_whitespace_only_returns_empty_list():
    s = _make_strategy()
    assert s.extract("   \n\t  ") == []


# ---------------------------------------------------------------------------
# Test 13
# ---------------------------------------------------------------------------
def test_extract_valid_json():
    payload = json.dumps([{"entity": "Alice", "type": "PERSON", "relations": []}])
    s = _make_strategy(payload)
    result = s.extract("Alice works here.")
    assert len(result) == 1
    assert result[0]["entity"] == "Alice"


# ---------------------------------------------------------------------------
# Test 14
# ---------------------------------------------------------------------------
def test_extract_strips_backtick_markdown_fence():
    payload = "```\n[{\"entity\": \"Bob\", \"type\": \"PERSON\", \"relations\": []}]\n```"
    s = _make_strategy(payload)
    result = s.extract("Bob is here.")
    assert result[0]["entity"] == "Bob"


# ---------------------------------------------------------------------------
# Test 15
# ---------------------------------------------------------------------------
def test_extract_strips_json_labeled_markdown_fence():
    payload = "```json\n[{\"entity\": \"Eve\", \"type\": \"PERSON\", \"relations\": []}]\n```"
    s = _make_strategy(payload)
    result = s.extract("Eve was mentioned.")
    assert result[0]["entity"] == "Eve"


# ---------------------------------------------------------------------------
# Test 16
# ---------------------------------------------------------------------------
def test_extract_fallback_bracket_search():
    payload = 'Some preamble. [{"entity": "Org", "type": "ORG", "relations": []}] trailing'
    s = _make_strategy(payload)
    result = s.extract("Org is a company.")
    assert result[0]["type"] == "ORG"


# ---------------------------------------------------------------------------
# Test 17
# ---------------------------------------------------------------------------
def test_extract_handles_non_list_json():
    """JSON object (not array) must return []."""
    payload = '{"entity": "Bad", "type": "OTHER"}'
    s = _make_strategy(payload)
    result = s.extract("text")
    assert result == []


# ---------------------------------------------------------------------------
# Test 18
# ---------------------------------------------------------------------------
def test_extract_handles_malformed_json():
    payload = "not json at all"
    s = _make_strategy(payload)
    result = s.extract("text")
    assert result == []


# ---------------------------------------------------------------------------
# Test 19
# ---------------------------------------------------------------------------
def test_extract_llm_exception_returns_empty():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("LLM connection failed")
    with patch("factories.llm_factory.LLMFactory.get_llm", return_value=mock_llm):
        s = LLMEntityExtractionStrategy(llm_provider="openai")
    result = s.extract("some text")
    assert result == []


# ---------------------------------------------------------------------------
# Test 20
# ---------------------------------------------------------------------------
def test_extract_result_has_correct_shape():
    payload = json.dumps([{
        "entity": "Apple",
        "type": "ORG",
        "relations": [{"target": "Tim Cook", "relation": "CEO_OF"}],
    }])
    s = _make_strategy(payload)
    result = s.extract("Apple is led by Tim Cook.")
    assert result[0]["entity"] == "Apple"
    assert result[0]["type"] == "ORG"
    assert result[0]["relations"][0]["target"] == "Tim Cook"


# ---------------------------------------------------------------------------
# Test 21
# ---------------------------------------------------------------------------
def test_parse_json_valid_array():
    raw = '[{"entity": "X", "type": "OTHER", "relations": []}]'
    result = LLMEntityExtractionStrategy._parse_json(raw)
    assert isinstance(result, list)
    assert result[0]["entity"] == "X"


# ---------------------------------------------------------------------------
# Test 22
# ---------------------------------------------------------------------------
def test_parse_json_with_plain_fence():
    raw = "```\n[{\"entity\": \"Y\", \"type\": \"PERSON\", \"relations\": []}]\n```"
    result = LLMEntityExtractionStrategy._parse_json(raw)
    assert result[0]["entity"] == "Y"


# ---------------------------------------------------------------------------
# Test 23
# ---------------------------------------------------------------------------
def test_parse_json_bracket_fallback():
    raw = 'here is the result: [{"entity": "Z", "type": "ORG", "relations": []}] done.'
    result = LLMEntityExtractionStrategy._parse_json(raw)
    assert result[0]["entity"] == "Z"


# ---------------------------------------------------------------------------
# Test 24
# ---------------------------------------------------------------------------
def test_parse_json_empty_string():
    result = LLMEntityExtractionStrategy._parse_json("")
    assert result == []


# ---------------------------------------------------------------------------
# Test 25
# ---------------------------------------------------------------------------
def test_parse_json_non_array_dict_returns_empty():
    raw = '{"key": "value"}'
    result = LLMEntityExtractionStrategy._parse_json(raw)
    assert result == []

