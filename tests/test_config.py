"""
Tests 1-10 — core/config.py
"""
import os
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_BASE_ENV = {
    "NEO4J_URL": "bolt://localhost:7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "ENTITY_EXTRACTION_PROVIDER": "llm",
    "RETRIEVAL_MODE": "vector",
    "GRAPH_STORE_PROVIDER": "neo4j",
    "LLM_PROVIDER": "openai",
    "EMBEDDING_PROVIDER": "openai",
    "VECTOR_DB_PROVIDER": "qdrant",
    "QDRANT_URL": "http://localhost:6333",
    "DATA_DIR": "./data",
    "COLLECTION_NAME": "test",
}


def _cfg_prop(prop: str, **env_overrides):
    """
    Return the value of Config.<prop> evaluated inside a patched env.
    This ensures os.getenv() sees our overrides when the property is actually called.
    """
    env = {**_BASE_ENV, **env_overrides}
    with patch.dict(os.environ, env, clear=True):
        from core.config import Config
        return getattr(Config(), prop)


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------
def test_neo4j_url_default():
    assert _cfg_prop("neo4j_url") == "bolt://localhost:7687"


# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------
def test_neo4j_username_default():
    assert _cfg_prop("neo4j_username") == "neo4j"


# ---------------------------------------------------------------------------
# Test 3
# ---------------------------------------------------------------------------
def test_neo4j_password_default():
    assert _cfg_prop("neo4j_password") == "password"


# ---------------------------------------------------------------------------
# Test 4
# ---------------------------------------------------------------------------
def test_entity_extraction_provider_default():
    assert _cfg_prop("entity_extraction_provider") == "llm"


# ---------------------------------------------------------------------------
# Test 5
# ---------------------------------------------------------------------------
def test_retrieval_mode_default():
    assert _cfg_prop("retrieval_mode") == "vector"


# ---------------------------------------------------------------------------
# Test 6
# ---------------------------------------------------------------------------
def test_retrieval_mode_from_env():
    assert _cfg_prop("retrieval_mode", RETRIEVAL_MODE="hybrid") == "hybrid"


# ---------------------------------------------------------------------------
# Test 7
# ---------------------------------------------------------------------------
def test_graph_store_provider_default():
    assert _cfg_prop("graph_store_provider") == "neo4j"


# ---------------------------------------------------------------------------
# Test 8
# ---------------------------------------------------------------------------
def test_graph_store_provider_from_env():
    assert _cfg_prop("graph_store_provider", GRAPH_STORE_PROVIDER="neo4j") == "neo4j"


# ---------------------------------------------------------------------------
# Test 9
# ---------------------------------------------------------------------------
def test_missing_required_var_raises():
    """Config.get_env_var should raise ValueError when a required var is absent."""
    with patch.dict(os.environ, {}, clear=True):
        from core.config import Config
        cfg = Config()
        with pytest.raises(ValueError, match="Missing required environment variable"):
            _ = cfg.llm_provider


# ---------------------------------------------------------------------------
# Test 10
# ---------------------------------------------------------------------------
def test_config_lowercases_retrieval_mode():
    assert _cfg_prop("retrieval_mode", RETRIEVAL_MODE="HYBRID") == "hybrid"
