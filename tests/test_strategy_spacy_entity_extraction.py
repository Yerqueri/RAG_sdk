"""
Tests 26-33 — strategies/entity_extraction/spacy_entity_extraction_strategy.py

All tests use a lightweight mock of spaCy so the real model is never downloaded.
"""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helper: build a mock spaCy doc with synthetic entities
# ---------------------------------------------------------------------------

def _mock_spacy(entities: list):
    """
    entities: list of (text, label_) tuples.
    Returns a patched `spacy` module whose `load()` returns a callable nlp.
    """
    mock_ent_objects = []
    for text, label in entities:
        e = MagicMock()
        e.text = text
        e.label_ = label
        mock_ent_objects.append(e)

    mock_doc = MagicMock()
    mock_doc.ents = mock_ent_objects

    mock_nlp = MagicMock(return_value=mock_doc)

    mock_spacy_module = MagicMock()
    mock_spacy_module.load.return_value = mock_nlp
    return mock_spacy_module


def _make_strategy(entities=None):
    if entities is None:
        entities = []
    mock_spacy = _mock_spacy(entities)
    with patch.dict("sys.modules", {"spacy": mock_spacy}):
        from rag_sdk.strategies.entity_extraction.spacy_entity_extraction_strategy import (
            SpacyEntityExtractionStrategy,
        )
        return SpacyEntityExtractionStrategy.__new__(SpacyEntityExtractionStrategy), mock_spacy


def _build(entities=None):
    """Build a strategy with mocked spaCy, return the strategy."""
    if entities is None:
        entities = []
    mock_spacy = _mock_spacy(entities)
    with patch.dict("sys.modules", {"spacy": mock_spacy}):
        # Must import inside the patch context so spacy is the mock
        import importlib
        import rag_sdk.strategies.entity_extraction.spacy_entity_extraction_strategy as mod
        importlib.reload(mod)
        s = mod.SpacyEntityExtractionStrategy.__new__(mod.SpacyEntityExtractionStrategy)
        s.nlp = mock_spacy.load.return_value
        return s


# ---------------------------------------------------------------------------
# Test 26
# ---------------------------------------------------------------------------
def test_spacy_extract_empty_string_returns_empty():
    s = _build()
    assert s.extract("") == []


# ---------------------------------------------------------------------------
# Test 27
# ---------------------------------------------------------------------------
def test_spacy_extract_whitespace_returns_empty():
    s = _build()
    assert s.extract("   ") == []


# ---------------------------------------------------------------------------
# Test 28
# ---------------------------------------------------------------------------
def test_spacy_extract_person_entity():
    s = _build([("Barack Obama", "PERSON")])
    result = s.extract("Barack Obama was president.")
    assert len(result) == 1
    assert result[0]["entity"] == "Barack Obama"
    assert result[0]["type"] == "PERSON"


# ---------------------------------------------------------------------------
# Test 29
# ---------------------------------------------------------------------------
def test_spacy_extract_org_entity():
    s = _build([("OpenAI", "ORG")])
    result = s.extract("OpenAI built GPT.")
    assert result[0]["type"] == "ORG"


# ---------------------------------------------------------------------------
# Test 30
# ---------------------------------------------------------------------------
def test_spacy_extract_gpe_maps_to_location():
    s = _build([("London", "GPE")])
    result = s.extract("London is a city.")
    assert result[0]["type"] == "LOCATION"


# ---------------------------------------------------------------------------
# Test 31
# ---------------------------------------------------------------------------
def test_spacy_extract_deduplicates():
    s = _build([("Alice", "PERSON"), ("Alice", "PERSON")])
    result = s.extract("Alice met Alice.")
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Test 32
# ---------------------------------------------------------------------------
def test_spacy_extract_unknown_label_maps_to_other():
    s = _build([("X42", "CARDINAL")])
    result = s.extract("X42 is a number.")
    assert result[0]["type"] == "OTHER"


# ---------------------------------------------------------------------------
# Test 33
# ---------------------------------------------------------------------------
def test_spacy_import_error_on_missing_spacy():
    import sys
    # Temporarily remove spacy from sys.modules
    with patch.dict("sys.modules", {"spacy": None}):
        import importlib
        import rag_sdk.strategies.entity_extraction.spacy_entity_extraction_strategy as mod
        importlib.reload(mod)
        with pytest.raises(ImportError, match="spaCy is not installed"):
            mod.SpacyEntityExtractionStrategy()

