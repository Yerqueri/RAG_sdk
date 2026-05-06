"""
Tests 49-61 — factories/graph_store_factory.py  &  factories/entity_extraction_factory.py
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch


# ===========================================================================
# Helpers
# ===========================================================================

def _neo4j_sys_mock():
    """Return a sys.modules patch that provides a stub neo4j module."""
    mock_neo4j = MagicMock()
    mock_driver_instance = MagicMock()
    sess = MagicMock()
    sess.__enter__ = MagicMock(return_value=sess)
    sess.__exit__ = MagicMock(return_value=False)
    mock_driver_instance.session.return_value = sess
    mock_neo4j.GraphDatabase.driver.return_value = mock_driver_instance
    return mock_neo4j


def _graph_env(**extra):
    base = {
        "GRAPH_STORE_PROVIDER": "neo4j",
        "NEO4J_URL": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "password",
        "LLM_PROVIDER": "openai",
        "EMBEDDING_PROVIDER": "openai",
        "VECTOR_DB_PROVIDER": "qdrant",
        "QDRANT_URL": "http://localhost:6333",
        "DATA_DIR": "./data",
        "COLLECTION_NAME": "test",
    }
    base.update(extra)
    return base


def _entity_env(**extra):
    base = {
        "ENTITY_EXTRACTION_PROVIDER": "llm",
        "LLM_PROVIDER": "openai",
        "EMBEDDING_PROVIDER": "openai",
        "VECTOR_DB_PROVIDER": "qdrant",
        "QDRANT_URL": "http://localhost:6333",
        "DATA_DIR": "./data",
        "COLLECTION_NAME": "test",
    }
    base.update(extra)
    return base


# ===========================================================================
# GraphStoreFactory
# ===========================================================================

# Test 49
def test_graph_factory_returns_neo4j_strategy():
    mock_neo4j = _neo4j_sys_mock()
    with patch.dict(os.environ, _graph_env(), clear=True), \
         patch.dict(sys.modules, {"neo4j": mock_neo4j}):
        import importlib
        import strategies.graph_store.neo4j_strategy as nmod
        importlib.reload(nmod)
        import factories.graph_store_factory as gfmod
        importlib.reload(gfmod)
        result = gfmod.GraphStoreFactory.get_graph_store(provider="neo4j")
        assert isinstance(result, nmod.Neo4jStrategy)


# Test 50
def test_graph_factory_unknown_provider_raises():
    mock_neo4j = _neo4j_sys_mock()
    with patch.dict(os.environ, _graph_env(), clear=True), \
         patch.dict(sys.modules, {"neo4j": mock_neo4j}):
        import importlib
        import factories.graph_store_factory as gfmod
        importlib.reload(gfmod)
        with pytest.raises(ValueError, match="Unsupported GRAPH_STORE_PROVIDER"):
            gfmod.GraphStoreFactory.get_graph_store(provider="postgres")


# Test 51
def test_graph_factory_uses_config_default_when_no_provider():
    mock_neo4j = _neo4j_sys_mock()
    with patch.dict(os.environ, _graph_env(GRAPH_STORE_PROVIDER="neo4j"), clear=True), \
         patch.dict(sys.modules, {"neo4j": mock_neo4j}):
        import importlib
        import strategies.graph_store.neo4j_strategy as nmod
        importlib.reload(nmod)
        import factories.graph_store_factory as gfmod
        importlib.reload(gfmod)
        result = gfmod.GraphStoreFactory.get_graph_store()
        assert isinstance(result, nmod.Neo4jStrategy)


# Test 52
def test_graph_factory_passes_correct_url():
    mock_neo4j = _neo4j_sys_mock()
    with patch.dict(os.environ, _graph_env(NEO4J_URL="bolt://myhost:7687"), clear=True), \
         patch.dict(sys.modules, {"neo4j": mock_neo4j}):
        import importlib
        import strategies.graph_store.neo4j_strategy as nmod
        importlib.reload(nmod)
        import factories.graph_store_factory as gfmod
        importlib.reload(gfmod)
        gfmod.GraphStoreFactory.get_graph_store(provider="neo4j")
    call_args = mock_neo4j.GraphDatabase.driver.call_args
    assert call_args.args[0] == "bolt://myhost:7687"


# Test 53
def test_graph_factory_passes_correct_username():
    mock_neo4j = _neo4j_sys_mock()
    with patch.dict(os.environ, _graph_env(NEO4J_USERNAME="admin"), clear=True), \
         patch.dict(sys.modules, {"neo4j": mock_neo4j}):
        import importlib
        import strategies.graph_store.neo4j_strategy as nmod
        importlib.reload(nmod)
        import factories.graph_store_factory as gfmod
        importlib.reload(gfmod)
        gfmod.GraphStoreFactory.get_graph_store(provider="neo4j")
    call_args = mock_neo4j.GraphDatabase.driver.call_args
    assert call_args.kwargs["auth"][0] == "admin"


# Test 54
def test_graph_factory_passes_correct_password():
    mock_neo4j = _neo4j_sys_mock()
    with patch.dict(os.environ, _graph_env(NEO4J_PASSWORD="secret123"), clear=True), \
         patch.dict(sys.modules, {"neo4j": mock_neo4j}):
        import importlib
        import strategies.graph_store.neo4j_strategy as nmod
        importlib.reload(nmod)
        import factories.graph_store_factory as gfmod
        importlib.reload(gfmod)
        gfmod.GraphStoreFactory.get_graph_store(provider="neo4j")
    call_args = mock_neo4j.GraphDatabase.driver.call_args
    assert call_args.kwargs["auth"][1] == "secret123"


# ===========================================================================
# EntityExtractionFactory
# ===========================================================================

# Test 55
def test_entity_factory_returns_llm_strategy():
    mock_llm = MagicMock()
    with patch.dict(os.environ, _entity_env(), clear=True), \
         patch("factories.llm_factory.LLMFactory.get_llm", return_value=mock_llm):
        from factories.entity_extraction_factory import EntityExtractionFactory
        from strategies.entity_extraction.llm_entity_extraction_strategy import LLMEntityExtractionStrategy
        result = EntityExtractionFactory.get_extractor(provider="llm")
        assert isinstance(result, LLMEntityExtractionStrategy)


# Test 56
def test_entity_factory_returns_spacy_strategy():
    mock_spacy = MagicMock()
    mock_spacy.load.return_value = MagicMock()
    with patch.dict(os.environ, _entity_env(), clear=True), \
         patch.dict(sys.modules, {"spacy": mock_spacy}):
        import importlib
        import strategies.entity_extraction.spacy_entity_extraction_strategy as smod
        importlib.reload(smod)
        from factories.entity_extraction_factory import EntityExtractionFactory
        result = EntityExtractionFactory.get_extractor(provider="spacy")
        assert isinstance(result, smod.SpacyEntityExtractionStrategy)


# Test 57
def test_entity_factory_unknown_provider_raises():
    with patch.dict(os.environ, _entity_env(), clear=True):
        from factories.entity_extraction_factory import EntityExtractionFactory
        with pytest.raises(ValueError, match="Unsupported ENTITY_EXTRACTION_PROVIDER"):
            EntityExtractionFactory.get_extractor(provider="bert")


# Test 58
def test_entity_factory_uses_config_default():
    mock_llm = MagicMock()
    with patch.dict(os.environ, _entity_env(ENTITY_EXTRACTION_PROVIDER="llm"), clear=True), \
         patch("factories.llm_factory.LLMFactory.get_llm", return_value=mock_llm):
        from factories.entity_extraction_factory import EntityExtractionFactory
        from strategies.entity_extraction.llm_entity_extraction_strategy import LLMEntityExtractionStrategy
        result = EntityExtractionFactory.get_extractor()
        assert isinstance(result, LLMEntityExtractionStrategy)


# Test 59
def test_entity_factory_passes_llm_provider():
    captured = {}

    def fake_get_llm(provider=None):
        captured["provider"] = provider
        return MagicMock()

    with patch.dict(os.environ, _entity_env(), clear=True), \
         patch("factories.llm_factory.LLMFactory.get_llm", side_effect=fake_get_llm):
        from factories.entity_extraction_factory import EntityExtractionFactory
        EntityExtractionFactory.get_extractor(provider="llm", llm_provider="anthropic")
    assert captured["provider"] == "anthropic"


# Test 60
def test_entity_factory_llm_strategy_has_extract_method():
    mock_llm = MagicMock()
    with patch.dict(os.environ, _entity_env(), clear=True), \
         patch("factories.llm_factory.LLMFactory.get_llm", return_value=mock_llm):
        from factories.entity_extraction_factory import EntityExtractionFactory
        result = EntityExtractionFactory.get_extractor(provider="llm")
        assert callable(getattr(result, "extract", None))


# Test 61
def test_entity_factory_spacy_strategy_has_extract_method():
    mock_spacy = MagicMock()
    mock_spacy.load.return_value = MagicMock()
    with patch.dict(os.environ, _entity_env(), clear=True), \
         patch.dict(sys.modules, {"spacy": mock_spacy}):
        import importlib
        import strategies.entity_extraction.spacy_entity_extraction_strategy as smod
        importlib.reload(smod)
        from factories.entity_extraction_factory import EntityExtractionFactory
        result = EntityExtractionFactory.get_extractor(provider="spacy")
        assert callable(getattr(result, "extract", None))

