#!/usr/bin/env python3
"""
live_test.py — Manual end-to-end integration test suite for the Knowledge Graph RAG pipeline.

This file is intentionally OUTSIDE the tests/ directory and is NOT collected by pytest.
Run it manually:

    python live_test.py
    python live_test.py --section neo4j          # only Neo4j tests
    python live_test.py --section qdrant         # only Qdrant tests
    python live_test.py --section entity         # only entity extraction tests
    python live_test.py --section hybrid         # only hybrid retriever tests
    python live_test.py --section sdk            # only full SDK pipeline tests
    python live_test.py --section graph_schema   # only graph schema verification
    python live_test.py --section cleanup        # only cleanup
    python live_test.py --section all            # everything (default)

Requirements:
  - docker-compose up -d   (starts Qdrant + Neo4j)
  - .env file populated with at least one LLM + embedding provider
  - pip install -r requirements.txt
"""

# ── stdlib ────────────────────────────────────────────────────────────────── #
import argparse
import os
import sys
import time
import textwrap
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, List, Optional

# ── project root on path ─────────────────────────────────────────────────── #
sys.path.insert(0, os.path.dirname(__file__))

# ── ANSI colours ─────────────────────────────────────────────────────────── #
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

def _c(colour: str, text: str) -> str:
    return f"{colour}{text}{_RESET}"

# ── Test result bookkeeping ───────────────────────────────────────────────── #

@dataclass
class LTResult:
    name: str
    section: str
    passed: bool
    duration: float
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: str = ""

@dataclass
class Suite:
    results: List[LTResult] = field(default_factory=list)

    def record(self, r: LTResult):
        self.results.append(r)
        status = (
            _c(_YELLOW, "SKIP") if r.skipped else
            _c(_GREEN,  "PASS") if r.passed else
            _c(_RED,    "FAIL")
        )
        timing = _c(_DIM, f"({r.duration:.2f}s)")
        line = f"  {status}  {r.name} {timing}"
        if r.skipped:
            line += _c(_DIM, f"  ← {r.skip_reason}")
        elif not r.passed and r.error:
            short = r.error.splitlines()[-1] if r.error else ""
            line += _c(_RED, f"  ← {short}")
        print(line)

    def summary(self):
        total   = len(self.results)
        passed  = sum(1 for r in self.results if r.passed and not r.skipped)
        failed  = sum(1 for r in self.results if not r.passed and not r.skipped)
        skipped = sum(1 for r in self.results if r.skipped)
        total_t = sum(r.duration for r in self.results)

        print()
        print(_c(_BOLD, "─" * 60))
        print(_c(_BOLD, "LIVE TEST SUMMARY"))
        print(_c(_BOLD, "─" * 60))
        print(f"  Total   : {total}")
        print(f"  {_c(_GREEN, 'Passed')}  : {passed}")
        print(f"  {_c(_RED,   'Failed')}  : {failed}")
        print(f"  {_c(_YELLOW,'Skipped')}: {skipped}")
        print(f"  Time    : {total_t:.2f}s")
        print(_c(_BOLD, "─" * 60))

        if failed:
            print(_c(_RED + _BOLD, "\nFAILED TESTS:"))
            for r in self.results:
                if not r.passed and not r.skipped:
                    print(f"\n  ● {r.section} / {r.name}")
                    if r.error:
                        for line in textwrap.indent(r.error, "    ").splitlines():
                            print(_c(_RED, line))
        return failed == 0


_suite = Suite()


# ── Test decorator / runner ───────────────────────────────────────────────── #

@contextmanager
def _measuring():
    t0 = time.monotonic()
    yield
    return time.monotonic() - t0


def run_test(
    section: str,
    name: str,
    fn: Callable,
    skip_if: bool = False,
    skip_reason: str = "",
):
    if skip_if:
        r = LTResult(name=name, section=section, passed=True,
                     duration=0.0, skipped=True, skip_reason=skip_reason)
        _suite.record(r)
        return

    t0 = time.monotonic()
    try:
        fn()
        duration = time.monotonic() - t0
        _suite.record(LTResult(name=name, section=section, passed=True, duration=duration))
    except Exception:
        duration = time.monotonic() - t0
        _suite.record(LTResult(
            name=name, section=section, passed=False,
            duration=duration, error=traceback.format_exc(),
        ))


# ── Service availability probes ───────────────────────────────────────────── #

def _neo4j_available(url: str, user: str, pw: str) -> bool:
    try:
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(url, auth=(user, pw))
        with d.session() as s:
            s.run("RETURN 1")
        d.close()
        return True
    except Exception:
        return False


def _qdrant_available(url: str) -> bool:
    try:
        from qdrant_client import QdrantClient
        QdrantClient(url=url).get_collections()
        return True
    except Exception:
        return False


def _ollama_available(model: str = "nomic-embed-text") -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        tags = [m["name"].split(":")[0] for m in r.json().get("models", [])]
        return model.split(":")[0] in tags
    except Exception:
        return False


def _spacy_available() -> bool:
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


def _openai_key_present() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _google_key_present() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY"))


def _anthropic_key_present() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ── Shared test corpus ───────────────────────────────────────────────────── #

_CORPUS = [
    {
        "source": "live_test::acme_annual_report",
        "content": (
            "Acme Corporation was founded in 1990 by Jane Smith and John Doe. "
            "The company is headquartered in San Francisco, California. "
            "Jane Smith serves as the Chief Executive Officer and John Doe is the Chief Technology Officer. "
            "Acme Corporation develops enterprise software, including its flagship product AcmeOS. "
            "In fiscal year 2024, Acme Corporation reported revenue of $2.4 billion, "
            "a 15 percent increase from the prior year. "
            "The company employs over 12,000 people globally."
        ),
    },
    {
        "source": "live_test::climate_report",
        "content": (
            "The Global Climate Initiative (GCI) published its 2024 annual report. "
            "Dr. Maria Lopez, Director of the GCI, stated that carbon emissions fell by 8 percent "
            "compared to 2023. Key contributors include renewable energy adoption in Germany, "
            "India, and Brazil. The United Nations praised the GCI report and called on "
            "member states to accelerate green energy transitions. "
            "GCI's headquarters are located in Geneva, Switzerland."
        ),
    },
    {
        "source": "live_test::tech_acquisitions",
        "content": (
            "TechVentures Inc. acquired DataSpark Ltd. for $450 million in March 2024. "
            "TechVentures Inc. is led by CEO Robert Chen. DataSpark Ltd. was co-founded "
            "by Alice Wang and Bob Kumar. The acquisition strengthens TechVentures Inc.'s "
            "AI analytics portfolio. DataSpark Ltd. operates offices in London, Tokyo, and Austin. "
            "Robert Chen stated the deal is expected to close regulatory review by Q3 2024."
        ),
    },
]

_TEST_COLLECTION = "live_test_collection"
_TEST_NEO4J_LABEL = "LiveTestDoc"


# ═══════════════════════════════════════════════════════════════════════════ #
#  SECTION 1 — Neo4j live tests                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def run_neo4j_section(neo4j_url, neo4j_user, neo4j_pw, available):
    from langchain_core.documents import Document

    section = "Neo4j"
    print(f"\n{_c(_BOLD + _CYAN, '── Section 1: Neo4j Graph Store ──')}")

    # ── 1.1 Connection ───────────────────────────────────────────────────── #
    def test_connection():
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pw))
        with d.session() as s:
            result = s.run("RETURN 'hello' AS msg")
            assert result.single()["msg"] == "hello"
        d.close()

    run_test(section, "1.01  Neo4j driver connects successfully", test_connection,
             skip_if=not available, skip_reason="Neo4j not reachable")

    # ── 1.2 Strategy instantiation ───────────────────────────────────────── #
    def test_strategy_init():
        from rag_sdk.strategies.graph_store.neo4j_strategy import Neo4jStrategy
        s = Neo4jStrategy(neo4j_url, neo4j_user, neo4j_pw)
        s.close()

    run_test(section, "1.02  Neo4jStrategy instantiates and creates constraints", test_strategy_init,
             skip_if=not available, skip_reason="Neo4j not reachable")

    # ── 1.3 store_graph — single chunk, single entity ────────────────────── #
    def test_store_single_chunk():
        from rag_sdk.strategies.graph_store.neo4j_strategy import Neo4jStrategy
        s = Neo4jStrategy(neo4j_url, neo4j_user, neo4j_pw)
        chunk = Document(page_content="Acme Corporation was founded by Jane Smith.",
                         metadata={"source": "live_test::single"})
        entities = {0: [{"entity": "Jane Smith", "type": "PERSON", "relations": []},
                        {"entity": "Acme Corporation", "type": "ORG",
                         "relations": [{"target": "Jane Smith", "relation": "FOUNDED_BY"}]}]}
        s.store_graph("live_test::single", [chunk], entities)
        s.close()

    run_test(section, "1.03  store_graph writes single chunk + entities", test_store_single_chunk,
             skip_if=not available, skip_reason="Neo4j not reachable")

    # ── 1.4 store_graph — multiple chunks + NEXT edges ───────────────────── #
    def test_store_multiple_chunks():
        from rag_sdk.strategies.graph_store.neo4j_strategy import Neo4jStrategy
        s = Neo4jStrategy(neo4j_url, neo4j_user, neo4j_pw)
        chunks = [Document(page_content=c["content"], metadata={"source": c["source"]})
                  for c in _CORPUS]
        entities = {
            0: [{"entity": "Acme Corporation", "type": "ORG",
                 "relations": [{"target": "Jane Smith", "relation": "FOUNDED_BY"}]},
                {"entity": "Jane Smith", "type": "PERSON", "relations": []},
                {"entity": "San Francisco", "type": "LOCATION", "relations": []}],
            1: [{"entity": "Global Climate Initiative", "type": "ORG", "relations": []},
                {"entity": "Dr. Maria Lopez", "type": "PERSON", "relations": []},
                {"entity": "United Nations", "type": "ORG", "relations": []}],
            2: [{"entity": "TechVentures Inc.", "type": "ORG",
                 "relations": [{"target": "DataSpark Ltd.", "relation": "ACQUIRED"}]},
                {"entity": "DataSpark Ltd.", "type": "ORG", "relations": []},
                {"entity": "Robert Chen", "type": "PERSON", "relations": []}],
        }
        s.store_graph("live_test::corpus", chunks, entities)
        s.close()

    run_test(section, "1.04  store_graph writes 3-chunk corpus with NEXT chain", test_store_multiple_chunks,
             skip_if=not available, skip_reason="Neo4j not reachable")

    # ── 1.5 get_related_chunks — exact match ────────────────────────────── #
    def test_get_related_exact():
        from rag_sdk.strategies.graph_store.neo4j_strategy import Neo4jStrategy
        s = Neo4jStrategy(neo4j_url, neo4j_user, neo4j_pw)
        results = s.get_related_chunks(["Acme Corporation"], k=5)
        s.close()
        assert len(results) > 0, "Expected at least one chunk for 'Acme Corporation'"
        assert any("Acme" in r for r in results), "Expected Acme content in results"

    run_test(section, "1.05  get_related_chunks returns correct chunk for 'Acme Corporation'",
             test_get_related_exact, skip_if=not available, skip_reason="Neo4j not reachable")

    # ── 1.6 get_related_chunks — case-insensitive ────────────────────────── #
    def test_get_related_case():
        from rag_sdk.strategies.graph_store.neo4j_strategy import Neo4jStrategy
        s = Neo4jStrategy(neo4j_url, neo4j_user, neo4j_pw)
        upper = s.get_related_chunks(["ACME CORPORATION"], k=5)
        lower = s.get_related_chunks(["acme corporation"], k=5)
        s.close()
        assert set(upper) == set(lower), "Case-insensitive lookup should return same results"

    run_test(section, "1.06  get_related_chunks is case-insensitive",
             test_get_related_case, skip_if=not available, skip_reason="Neo4j not reachable")

    # ── 1.7 get_related_chunks — k limit respected ───────────────────────── #
    def test_get_related_k_limit():
        from rag_sdk.strategies.graph_store.neo4j_strategy import Neo4jStrategy
        s = Neo4jStrategy(neo4j_url, neo4j_user, neo4j_pw)
        results = s.get_related_chunks(["Acme Corporation", "Jane Smith",
                                        "TechVentures Inc.", "Robert Chen"], k=2)
        s.close()
        assert len(results) <= 2, f"Expected ≤ 2 results, got {len(results)}"

    run_test(section, "1.07  get_related_chunks respects k limit",
             test_get_related_k_limit, skip_if=not available, skip_reason="Neo4j not reachable")

    # ── 1.8 get_related_chunks — empty returns [] ────────────────────────── #
    def test_get_related_empty():
        from rag_sdk.strategies.graph_store.neo4j_strategy import Neo4jStrategy
        s = Neo4jStrategy(neo4j_url, neo4j_user, neo4j_pw)
        results = s.get_related_chunks([], k=10)
        s.close()
        assert results == []

    run_test(section, "1.08  get_related_chunks([]) returns empty list",
             test_get_related_empty, skip_if=not available, skip_reason="Neo4j not reachable")

    # ── 1.9 get_related_chunks — unknown entity returns [] ───────────────── #
    def test_get_related_unknown():
        from rag_sdk.strategies.graph_store.neo4j_strategy import Neo4jStrategy
        s = Neo4jStrategy(neo4j_url, neo4j_user, neo4j_pw)
        results = s.get_related_chunks(["XYZZY_ENTITY_THAT_DOES_NOT_EXIST_12345"], k=5)
        s.close()
        assert results == [], f"Expected [], got {results}"

    run_test(section, "1.09  get_related_chunks unknown entity returns []",
             test_get_related_unknown, skip_if=not available, skip_reason="Neo4j not reachable")

    # ── 1.10 Verify Document → Chunk → Entity path in graph ─────────────── #
    def test_graph_schema_paths():
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pw))
        with d.session() as s:
            # Document → Chunk edge
            r1 = s.run(
                "MATCH (d:Document)-[:CONTAINS]->(c:Chunk) "
                "WHERE d.id STARTS WITH 'live_test' RETURN COUNT(c) AS cnt"
            ).single()["cnt"]
            assert r1 > 0, "Expected Document-[:CONTAINS]->Chunk edges"

            # Chunk → Entity edge
            r2 = s.run(
                "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) "
                "WHERE c.id STARTS WITH 'live_test' RETURN COUNT(e) AS cnt"
            ).single()["cnt"]
            assert r2 > 0, "Expected Chunk-[:MENTIONS]->Entity edges"

            # Entity → Entity RELATED_TO
            r3 = s.run(
                "MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) RETURN COUNT(r) AS cnt"
            ).single()["cnt"]
            assert r3 > 0, "Expected Entity-[:RELATED_TO]->Entity edges"

            # NEXT chain
            r4 = s.run(
                "MATCH (a:Chunk)-[:NEXT]->(b:Chunk) "
                "WHERE a.id STARTS WITH 'live_test' RETURN COUNT(*) AS cnt"
            ).single()["cnt"]
            assert r4 > 0, "Expected Chunk-[:NEXT]->Chunk edges"
        d.close()

    run_test(section, "1.10  Graph schema: Document→Chunk→Entity + NEXT chain verified",
             test_graph_schema_paths, skip_if=not available, skip_reason="Neo4j not reachable")

    # ── 1.11 RELATED_TO edge carries relation property ───────────────────── #
    def test_related_to_property():
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pw))
        with d.session() as s:
            result = s.run(
                "MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) "
                "WHERE r.relation IS NOT NULL RETURN r.relation LIMIT 1"
            ).single()
            assert result is not None, "Expected at least one RELATED_TO with a relation property"
        d.close()

    run_test(section, "1.11  RELATED_TO edges carry 'relation' property",
             test_related_to_property, skip_if=not available, skip_reason="Neo4j not reachable")

    # ── 1.12 GraphStore facade delegates correctly ───────────────────────── #
    def test_graph_store_facade():
        from rag_sdk.core.graph_store import GraphStore
        from langchain_core.documents import Document
        os.environ.setdefault("GRAPH_STORE_PROVIDER", "neo4j")
        gs = GraphStore()
        chunk = Document(page_content="Facade test text.", metadata={"source": "live_test::facade"})
        gs.store_graph("live_test::facade", [chunk], {0: [{"entity": "FacadeEntity", "type": "OTHER", "relations": []}]})
        results = gs.get_related_chunks(["FacadeEntity"], k=3)
        gs.close()
        assert any("Facade" in r for r in results)

    run_test(section, "1.12  GraphStore facade: store + retrieve round-trip",
             test_graph_store_facade, skip_if=not available, skip_reason="Neo4j not reachable")


# ═══════════════════════════════════════════════════════════════════════════ #
#  SECTION 2 — Qdrant live tests                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

def run_qdrant_section(qdrant_url, qdrant_available, embed_available, embed_provider):
    section = "Qdrant"
    skip = not qdrant_available
    skip_embed = not embed_available
    print(f"\n{_c(_BOLD + _CYAN, '── Section 2: Qdrant Vector Store ──')}")

    # ── 2.1 Qdrant HTTP health check ─────────────────────────────────────── #
    def test_qdrant_health():
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url)
        collections = client.get_collections()
        assert collections is not None

    run_test(section, "2.01  Qdrant HTTP endpoint reachable",
             test_qdrant_health, skip_if=skip, skip_reason="Qdrant not reachable")

    # ── 2.2 Create and delete a collection ───────────────────────────────── #
    def test_create_delete_collection():
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams
        client = QdrantClient(url=qdrant_url)
        name = "live_test_temp_collection"
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
        names = [c.name for c in client.get_collections().collections]
        assert name in names
        client.delete_collection(name)
        names2 = [c.name for c in client.get_collections().collections]
        assert name not in names2

    run_test(section, "2.02  Qdrant: create and delete collection",
             test_create_delete_collection, skip_if=skip, skip_reason="Qdrant not reachable")

    # ── 2.3 QdrantStrategy store + retrieve ──────────────────────────────── #
    def test_qdrant_strategy_store():
        from langchain_core.documents import Document
        from rag_sdk.factories.embedding_factory import EmbeddingFactory
        from rag_sdk.factories.vector_store_factory import VectorStoreFactory
        os.environ["COLLECTION_NAME"] = _TEST_COLLECTION
        embeddings = EmbeddingFactory.get_embeddings(provider=embed_provider)
        vs = VectorStoreFactory.get_vector_store(provider="qdrant")
        docs = [Document(page_content=c["content"], metadata={"source": c["source"]})
                for c in _CORPUS]
        vs.store(docs, embeddings)

    run_test(section, "2.03  QdrantStrategy: store 3 documents with real embeddings",
             test_qdrant_strategy_store,
             skip_if=(skip or skip_embed),
             skip_reason="Qdrant or embedding model not available")

    # ── 2.4 Retrieve relevant chunks ─────────────────────────────────────── #
    def test_qdrant_retrieval():
        from rag_sdk.factories.embedding_factory import EmbeddingFactory
        from rag_sdk.factories.vector_store_factory import VectorStoreFactory
        os.environ["COLLECTION_NAME"] = _TEST_COLLECTION
        embeddings = EmbeddingFactory.get_embeddings(provider=embed_provider)
        vs = VectorStoreFactory.get_vector_store(provider="qdrant")
        retriever = vs.get_retriever(embeddings=embeddings, k=2)
        docs = retriever.invoke("Who is the CEO of Acme Corporation?")
        assert len(docs) > 0, "Expected at least one retrieved document"
        combined = " ".join(d.page_content for d in docs)
        assert "Acme" in combined or "Jane" in combined, \
            "Expected Acme-related content in retrieval results"

    run_test(section, "2.04  QdrantStrategy: similarity search returns relevant results",
             test_qdrant_retrieval,
             skip_if=(skip or skip_embed),
             skip_reason="Qdrant or embedding model not available")

    # ── 2.5 k parameter is respected ─────────────────────────────────────── #
    def test_qdrant_k_limit():
        from rag_sdk.factories.embedding_factory import EmbeddingFactory
        from rag_sdk.factories.vector_store_factory import VectorStoreFactory
        os.environ["COLLECTION_NAME"] = _TEST_COLLECTION
        embeddings = EmbeddingFactory.get_embeddings(provider=embed_provider)
        vs = VectorStoreFactory.get_vector_store(provider="qdrant")
        retriever = vs.get_retriever(embeddings=embeddings, k=1)
        docs = retriever.invoke("climate change")
        assert len(docs) <= 1, f"Expected ≤1 result, got {len(docs)}"

    run_test(section, "2.05  QdrantStrategy: k=1 returns at most 1 result",
             test_qdrant_k_limit,
             skip_if=(skip or skip_embed),
             skip_reason="Qdrant or embedding model not available")


# ═══════════════════════════════════════════════════════════════════════════ #
#  SECTION 3 — Entity extraction live tests                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

def run_entity_section(llm_available, llm_provider, spacy_available):
    section = "EntityExtraction"
    print(f"\n{_c(_BOLD + _CYAN, '── Section 3: Entity Extraction ──')}")

    sample_text = _CORPUS[0]["content"]  # Acme Corp text

    # ── 3.1 spaCy extracts known entities ────────────────────────────────── #
    def test_spacy_extracts_person():
        from rag_sdk.strategies.entity_extraction.spacy_entity_extraction_strategy import SpacyEntityExtractionStrategy
        s = SpacyEntityExtractionStrategy()
        results = s.extract(sample_text)
        names = [e["entity"] for e in results]
        # Should find at least one PERSON (Jane Smith / John Doe)
        types = [e["type"] for e in results]
        assert "PERSON" in types, f"Expected PERSON entity; got: {results}"

    run_test(section, "3.01  spaCy extracts PERSON entities from Acme text",
             test_spacy_extracts_person,
             skip_if=not spacy_available, skip_reason="spaCy en_core_web_sm not installed")

    # ── 3.2 spaCy extracts ORG ────────────────────────────────────────────── #
    def test_spacy_extracts_org():
        from rag_sdk.strategies.entity_extraction.spacy_entity_extraction_strategy import SpacyEntityExtractionStrategy
        s = SpacyEntityExtractionStrategy()
        results = s.extract(sample_text)
        types = [e["type"] for e in results]
        assert "ORG" in types, f"Expected ORG entity; got: {results}"

    run_test(section, "3.02  spaCy extracts ORG entities from Acme text",
             test_spacy_extracts_org,
             skip_if=not spacy_available, skip_reason="spaCy en_core_web_sm not installed")

    # ── 3.3 spaCy extracts LOCATION ──────────────────────────────────────── #
    def test_spacy_extracts_location():
        from rag_sdk.strategies.entity_extraction.spacy_entity_extraction_strategy import SpacyEntityExtractionStrategy
        s = SpacyEntityExtractionStrategy()
        results = s.extract(sample_text)
        types = [e["type"] for e in results]
        assert "LOCATION" in types, f"Expected LOCATION entity; got: {results}"

    run_test(section, "3.03  spaCy extracts LOCATION entities (GPE→LOCATION)",
             test_spacy_extracts_location,
             skip_if=not spacy_available, skip_reason="spaCy en_core_web_sm not installed")

    # ── 3.4 spaCy empty text returns [] ──────────────────────────────────── #
    def test_spacy_empty():
        from rag_sdk.strategies.entity_extraction.spacy_entity_extraction_strategy import SpacyEntityExtractionStrategy
        s = SpacyEntityExtractionStrategy()
        assert s.extract("") == []
        assert s.extract("   ") == []

    run_test(section, "3.04  spaCy returns [] for empty/whitespace input",
             test_spacy_empty,
             skip_if=not spacy_available, skip_reason="spaCy en_core_web_sm not installed")

    # ── 3.5 spaCy deduplicates ────────────────────────────────────────────── #
    def test_spacy_no_duplicates():
        from rag_sdk.strategies.entity_extraction.spacy_entity_extraction_strategy import SpacyEntityExtractionStrategy
        s = SpacyEntityExtractionStrategy()
        text = "Apple Apple Apple Apple Microsoft Microsoft"
        results = s.extract(text)
        names = [e["entity"] for e in results]
        assert len(names) == len(set(names)), "Duplicate entities found"

    run_test(section, "3.05  spaCy deduplicates repeated entity mentions",
             test_spacy_no_duplicates,
             skip_if=not spacy_available, skip_reason="spaCy en_core_web_sm not installed")

    # ── 3.6 LLM entity extraction returns list ────────────────────────────── #
    def test_llm_returns_list():
        from rag_sdk.strategies.entity_extraction.llm_entity_extraction_strategy import LLMEntityExtractionStrategy
        s = LLMEntityExtractionStrategy(llm_provider=llm_provider)
        results = s.extract(sample_text)
        assert isinstance(results, list), f"Expected list, got {type(results)}"

    run_test(section, "3.06  LLM extractor returns a list",
             test_llm_returns_list,
             skip_if=not llm_available, skip_reason=f"LLM provider '{llm_provider}' not configured")

    # ── 3.7 LLM extracts at least one entity ─────────────────────────────── #
    def test_llm_extracts_entities():
        from rag_sdk.strategies.entity_extraction.llm_entity_extraction_strategy import LLMEntityExtractionStrategy
        s = LLMEntityExtractionStrategy(llm_provider=llm_provider)
        results = s.extract(sample_text)
        assert len(results) > 0, "LLM returned no entities from Acme Corp text"

    run_test(section, "3.07  LLM extracts ≥1 entity from Acme text",
             test_llm_extracts_entities,
             skip_if=not llm_available, skip_reason=f"LLM provider '{llm_provider}' not configured")

    # ── 3.8 LLM result items have required keys ───────────────────────────── #
    def test_llm_entity_shape():
        from rag_sdk.strategies.entity_extraction.llm_entity_extraction_strategy import LLMEntityExtractionStrategy
        s = LLMEntityExtractionStrategy(llm_provider=llm_provider)
        results = s.extract(sample_text)
        for item in results:
            assert "entity" in item, f"Missing 'entity' key: {item}"
            assert "type" in item,   f"Missing 'type' key: {item}"
            assert "relations" in item, f"Missing 'relations' key: {item}"
            assert isinstance(item["relations"], list), f"'relations' must be list: {item}"

    run_test(section, "3.08  LLM entity items have entity/type/relations keys",
             test_llm_entity_shape,
             skip_if=not llm_available, skip_reason=f"LLM provider '{llm_provider}' not configured")

    # ── 3.9 LLM extracts relationships ───────────────────────────────────── #
    def test_llm_extracts_relations():
        from rag_sdk.strategies.entity_extraction.llm_entity_extraction_strategy import LLMEntityExtractionStrategy
        s = LLMEntityExtractionStrategy(llm_provider=llm_provider)
        # Text with an obvious relationship
        text = ("Acme Corporation is led by CEO Jane Smith, "
                "who previously worked at MegaCorp.")
        results = s.extract(text)
        has_relation = any(item.get("relations") for item in results)
        assert has_relation, f"Expected at least one relationship; got: {results}"

    run_test(section, "3.09  LLM extractor captures entity relationships",
             test_llm_extracts_relations,
             skip_if=not llm_available, skip_reason=f"LLM provider '{llm_provider}' not configured")

    # ── 3.10 LLM empty text returns [] ───────────────────────────────────── #
    def test_llm_empty_text():
        from rag_sdk.strategies.entity_extraction.llm_entity_extraction_strategy import LLMEntityExtractionStrategy
        s = LLMEntityExtractionStrategy(llm_provider=llm_provider)
        assert s.extract("") == []
        assert s.extract("  \n\t  ") == []

    run_test(section, "3.10  LLM extractor returns [] for empty text (no API call)",
             test_llm_empty_text,
             skip_if=not llm_available, skip_reason=f"LLM provider '{llm_provider}' not configured")

    # ── 3.11 EntityExtractor facade — extract_from_chunks ────────────────── #
    def test_entity_extractor_facade():
        from langchain_core.documents import Document
        from rag_sdk.core.entity_extractor import EntityExtractor
        extractor = EntityExtractor(provider="spacy")
        chunks = [Document(page_content=c["content"]) for c in _CORPUS]
        results = extractor.extract_from_chunks(chunks)
        assert set(results.keys()) == {0, 1, 2}, "Expected keys 0, 1, 2"
        for idx, entities in results.items():
            assert isinstance(entities, list), f"Chunk {idx} should map to a list"

    run_test(section, "3.11  EntityExtractor.extract_from_chunks returns keyed dict",
             test_entity_extractor_facade,
             skip_if=not spacy_available, skip_reason="spaCy en_core_web_sm not installed")


# ═══════════════════════════════════════════════════════════════════════════ #
#  SECTION 4 — HybridRetriever live tests                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

def run_hybrid_section(neo4j_available, qdrant_available, embed_available,
                        neo4j_url, neo4j_user, neo4j_pw, qdrant_url, embed_provider, spacy_available):
    section = "HybridRetriever"
    print(f"\n{_c(_BOLD + _CYAN, '── Section 4: HybridRetriever ──')}")

    both_available = neo4j_available and qdrant_available and embed_available and spacy_available
    skip_reason = "Neo4j, Qdrant, embedding, or spaCy not available"

    # ── 4.1 Vector-only mode ─────────────────────────────────────────────── #
    def test_hybrid_vector_only():
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document
        from rag_sdk.core.hybrid_retriever import HybridRetriever
        from rag_sdk.factories.embedding_factory import EmbeddingFactory
        from rag_sdk.factories.vector_store_factory import VectorStoreFactory

        os.environ["COLLECTION_NAME"] = _TEST_COLLECTION
        embeddings = EmbeddingFactory.get_embeddings(provider=embed_provider)
        vs = VectorStoreFactory.get_vector_store(provider="qdrant")
        vector_retriever = vs.get_retriever(embeddings=embeddings, k=2)

        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            graph_store=None,
            entity_extractor=None,
            k=2,
        )
        docs = retriever.invoke("Who founded Acme Corporation?")
        assert len(docs) > 0, "Expected vector results"

    run_test(section, "4.01  HybridRetriever: vector-only mode returns docs",
             test_hybrid_vector_only,
             skip_if=not (qdrant_available and embed_available),
             skip_reason="Qdrant or embedding not available")

    # ── 4.2 Graph-only mode ──────────────────────────────────────────────── #
    def test_hybrid_graph_only():
        from rag_sdk.core.hybrid_retriever import HybridRetriever
        from rag_sdk.core.graph_store import GraphStore
        from rag_sdk.core.entity_extractor import EntityExtractor

        os.environ.setdefault("GRAPH_STORE_PROVIDER", "neo4j")
        gs = GraphStore()
        extractor = EntityExtractor(provider="spacy")

        retriever = HybridRetriever(
            vector_retriever=None,
            graph_store=gs,
            entity_extractor=extractor,
            k=3,
        )
        docs = retriever.invoke("What did Acme Corporation build?")
        gs.close()
        # May return 0 if spaCy doesn't find "Acme Corporation" in query — that's ok
        assert isinstance(docs, list)

    run_test(section, "4.02  HybridRetriever: graph-only mode returns list",
             test_hybrid_graph_only,
             skip_if=not (neo4j_available and spacy_available),
             skip_reason="Neo4j or spaCy not available")

    # ── 4.3 Hybrid mode — union of vector + graph ────────────────────────── #
    def test_hybrid_combined():
        from rag_sdk.core.hybrid_retriever import HybridRetriever
        from rag_sdk.core.graph_store import GraphStore
        from rag_sdk.core.entity_extractor import EntityExtractor
        from rag_sdk.factories.embedding_factory import EmbeddingFactory
        from rag_sdk.factories.vector_store_factory import VectorStoreFactory

        os.environ["COLLECTION_NAME"] = _TEST_COLLECTION
        os.environ.setdefault("GRAPH_STORE_PROVIDER", "neo4j")
        embeddings = EmbeddingFactory.get_embeddings(provider=embed_provider)
        vs = VectorStoreFactory.get_vector_store(provider="qdrant")
        vector_retriever = vs.get_retriever(embeddings=embeddings, k=2)
        gs = GraphStore()
        extractor = EntityExtractor(provider="spacy")

        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            graph_store=gs,
            entity_extractor=extractor,
            k=3,
        )
        docs = retriever.invoke("Who leads Acme Corporation?")
        gs.close()
        assert len(docs) > 0, "Hybrid mode should return at least 1 document"

    run_test(section, "4.03  HybridRetriever: hybrid mode returns ≥1 document",
             test_hybrid_combined, skip_if=not both_available, skip_reason=skip_reason)

    # ── 4.4 No duplicate content across vector + graph results ───────────── #
    def test_hybrid_no_duplicates():
        from rag_sdk.core.hybrid_retriever import HybridRetriever
        from rag_sdk.core.graph_store import GraphStore
        from rag_sdk.core.entity_extractor import EntityExtractor
        from rag_sdk.factories.embedding_factory import EmbeddingFactory
        from rag_sdk.factories.vector_store_factory import VectorStoreFactory

        os.environ["COLLECTION_NAME"] = _TEST_COLLECTION
        embeddings = EmbeddingFactory.get_embeddings(provider=embed_provider)
        vs = VectorStoreFactory.get_vector_store(provider="qdrant")
        vector_retriever = vs.get_retriever(embeddings=embeddings, k=3)
        gs = GraphStore()
        extractor = EntityExtractor(provider="spacy")

        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            graph_store=gs,
            entity_extractor=extractor,
            k=5,
        )
        docs = retriever.invoke("Tell me about corporations and their founders.")
        gs.close()
        contents = [d.page_content for d in docs]
        assert len(contents) == len(set(contents)), "Duplicate page_content found in hybrid results"

    run_test(section, "4.04  HybridRetriever: no duplicate content in results",
             test_hybrid_no_duplicates, skip_if=not both_available, skip_reason=skip_reason)

    # ── 4.5 Graph results tagged with source=knowledge_graph ─────────────── #
    def test_hybrid_graph_source_tag():
        from rag_sdk.core.hybrid_retriever import HybridRetriever
        from rag_sdk.core.graph_store import GraphStore
        from rag_sdk.core.entity_extractor import EntityExtractor

        gs = GraphStore()
        extractor = EntityExtractor(provider="spacy")

        retriever = HybridRetriever(
            vector_retriever=None,
            graph_store=gs,
            entity_extractor=extractor,
            k=5,
        )
        docs = retriever.invoke("Acme Corporation TechVentures Robert Chen")
        gs.close()
        for doc in docs:
            assert doc.metadata.get("source") == "knowledge_graph", \
                f"Graph-only doc should have source=knowledge_graph; got {doc.metadata}"

    run_test(section, "4.05  HybridRetriever: graph-sourced docs tagged 'knowledge_graph'",
             test_hybrid_graph_source_tag,
             skip_if=not (neo4j_available and spacy_available),
             skip_reason="Neo4j or spaCy not available")


# ═══════════════════════════════════════════════════════════════════════════ #
#  SECTION 5 — Full SDK pipeline tests                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def run_sdk_section(neo4j_available, qdrant_available, embed_available,
                     llm_available, llm_provider, embed_provider, spacy_available):
    section = "SDK"
    print(f"\n{_c(_BOLD + _CYAN, '── Section 5: Full RAGClient SDK Pipeline ──')}")

    full_available = neo4j_available and qdrant_available and embed_available and llm_available
    skip_reason = "Neo4j, Qdrant, embedding, or LLM provider not fully configured"

    # ── 5.1 RAGClient default constructor ────────────────────────────────── #
    def test_client_instantiates():
        from rag_sdk.sdk import RAGClient
        client = RAGClient()
        assert client.enable_graph is False
        assert client.retrieval_mode is None

    run_test(section, "5.01  RAGClient instantiates with defaults", test_client_instantiates)

    # ── 5.2 RAGClient graph constructor ──────────────────────────────────── #
    def test_client_graph_flags():
        from rag_sdk.sdk import RAGClient
        client = RAGClient(
            enable_graph=True,
            retrieval_mode="hybrid",
            entity_extraction_provider="spacy",
        )
        assert client.enable_graph is True
        assert client.retrieval_mode == "hybrid"
        assert client.entity_extraction_provider == "spacy"

    run_test(section, "5.02  RAGClient sets graph flags correctly", test_client_graph_flags)

    # ── 5.3 Vector-only ingest of in-memory text ──────────────────────────── #
    def test_vector_ingest():
        import tempfile, textwrap
        from rag_sdk.sdk import RAGClient
        os.environ["COLLECTION_NAME"] = _TEST_COLLECTION
        # Write corpus to temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, doc in enumerate(_CORPUS):
                path = os.path.join(tmpdir, f"doc_{i}.txt")
                with open(path, "w") as f:
                    f.write(doc["content"])
            client = RAGClient(
                embedding_provider=embed_provider,
                vector_db_provider="qdrant",
            )
            client.ingest_directory(tmpdir)

    run_test(section, "5.03  RAGClient.ingest_directory vector-only (no graph)",
             test_vector_ingest,
             skip_if=not (qdrant_available and embed_available),
             skip_reason="Qdrant or embedding not available")

    # ── 5.4 Graph-augmented ingest with spaCy ────────────────────────────── #
    def test_graph_ingest_spacy():
        import tempfile
        from rag_sdk.sdk import RAGClient
        os.environ["COLLECTION_NAME"] = _TEST_COLLECTION
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, doc in enumerate(_CORPUS):
                path = os.path.join(tmpdir, f"doc_{i}.txt")
                with open(path, "w") as f:
                    f.write(doc["content"])
            client = RAGClient(
                embedding_provider=embed_provider,
                vector_db_provider="qdrant",
                enable_graph=True,
                entity_extraction_provider="spacy",
            )
            client.ingest_directory(tmpdir)

    run_test(section, "5.04  RAGClient.ingest_directory with graph=True + spaCy extractor",
             test_graph_ingest_spacy,
             skip_if=not (neo4j_available and qdrant_available and embed_available and spacy_available),
             skip_reason="Neo4j, Qdrant, embedding, or spaCy not fully available")

    # ── 5.5 Vector-only query returns string answer ───────────────────────── #
    def test_vector_query():
        from rag_sdk.sdk import RAGClient
        os.environ["COLLECTION_NAME"] = _TEST_COLLECTION
        client = RAGClient(
            llm_provider=llm_provider,
            embedding_provider=embed_provider,
            vector_db_provider="qdrant",
            retrieval_mode="vector",
        )
        answer = client.query("Who are the founders of Acme Corporation?")
        assert isinstance(answer, str) and len(answer) > 5, \
            f"Expected non-empty string answer; got: {answer!r}"
        print(f"\n       [vector answer] {answer[:120]}")

    run_test(section, "5.05  RAGClient.query vector-only returns non-empty answer",
             test_vector_query, skip_if=not full_available, skip_reason=skip_reason)

    # ── 5.6 Hybrid query returns string answer ────────────────────────────── #
    def test_hybrid_query():
        from rag_sdk.sdk import RAGClient
        os.environ["COLLECTION_NAME"] = _TEST_COLLECTION
        client = RAGClient(
            llm_provider=llm_provider,
            embedding_provider=embed_provider,
            vector_db_provider="qdrant",
            entity_extraction_provider="spacy",
            retrieval_mode="hybrid",
        )
        answer = client.query("What company did TechVentures Inc. acquire?")
        assert isinstance(answer, str) and len(answer) > 5, \
            f"Expected non-empty string answer; got: {answer!r}"
        print(f"\n       [hybrid answer] {answer[:120]}")

    run_test(section, "5.06  RAGClient.query hybrid mode returns non-empty answer",
             test_hybrid_query,
             skip_if=not (full_available and neo4j_available and spacy_available),
             skip_reason=skip_reason + " / spaCy")

    # ── 5.7 Graph-only query ──────────────────────────────────────────────── #
    def test_graph_only_query():
        from rag_sdk.sdk import RAGClient
        os.environ["COLLECTION_NAME"] = _TEST_COLLECTION
        client = RAGClient(
            llm_provider=llm_provider,
            embedding_provider=embed_provider,
            vector_db_provider="qdrant",
            entity_extraction_provider="spacy",
            retrieval_mode="graph",
        )
        answer = client.query("Who is Dr. Maria Lopez?")
        assert isinstance(answer, str), "Expected string answer"
        print(f"\n       [graph answer]  {answer[:120]}")

    run_test(section, "5.07  RAGClient.query graph-only mode returns string answer",
             test_graph_only_query,
             skip_if=not (full_available and neo4j_available and spacy_available),
             skip_reason=skip_reason + " / spaCy")

    # ── 5.8 Provider override: switch LLM mid-session ────────────────────── #
    def test_provider_override():
        from rag_sdk.sdk import RAGClient
        # Two clients with different explicit providers — both must instantiate cleanly
        os.environ["COLLECTION_NAME"] = _TEST_COLLECTION
        c1 = RAGClient(embedding_provider=embed_provider, vector_db_provider="qdrant", retrieval_mode="vector")
        c2 = RAGClient(embedding_provider=embed_provider, vector_db_provider="qdrant", retrieval_mode="vector",
                       enable_graph=True, entity_extraction_provider="spacy")
        assert c1.retrieval_mode == "vector"
        assert c2.enable_graph is True

    run_test(section, "5.08  RAGClient: different provider configs co-exist",
             test_provider_override)

    # ── 5.9 ingest_file with a single .txt file ───────────────────────────── #
    def test_ingest_single_file():
        import tempfile
        from rag_sdk.sdk import RAGClient
        os.environ["COLLECTION_NAME"] = _TEST_COLLECTION
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write(_CORPUS[2]["content"])
            path = f.name
        try:
            client = RAGClient(
                embedding_provider=embed_provider,
                vector_db_provider="qdrant",
            )
            client.ingest_file(path)
        finally:
            os.unlink(path)

    run_test(section, "5.09  RAGClient.ingest_file ingests a single .txt file",
             test_ingest_single_file,
             skip_if=not (qdrant_available and embed_available),
             skip_reason="Qdrant or embedding not available")

    # ── 5.10 Missing file raises FileNotFoundError ─────────────────────────── #
    def test_missing_file_raises():
        from rag_sdk.sdk import RAGClient
        client = RAGClient()
        try:
            client.ingest_file("/tmp/this_file_does_not_exist_live_test.txt")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass  # expected

    run_test(section, "5.10  RAGClient.ingest_file raises FileNotFoundError for missing file",
             test_missing_file_raises)


# ═══════════════════════════════════════════════════════════════════════════ #
#  SECTION 6 — Graph schema deep verification                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

def run_graph_schema_section(neo4j_available, neo4j_url, neo4j_user, neo4j_pw):
    section = "GraphSchema"
    print(f"\n{_c(_BOLD + _CYAN, '── Section 6: Graph Schema Deep Verification ──')}")

    # ── 6.1 All required node labels exist ──────────────────────────────── #
    def test_node_labels():
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pw))
        with d.session() as s:
            labels = {r["label"] for r in s.run("CALL db.labels() YIELD label")}
        d.close()
        for required in ("Document", "Chunk", "Entity"):
            assert required in labels, f"Missing node label: {required}"

    run_test(section, "6.01  Graph has Document, Chunk, Entity node labels",
             test_node_labels, skip_if=not neo4j_available, skip_reason="Neo4j not reachable")

    # ── 6.2 All required relationship types exist ─────────────────────────── #
    def test_rel_types():
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pw))
        with d.session() as s:
            types = {r["relationshipType"] for r in
                     s.run("CALL db.relationshipTypes() YIELD relationshipType")}
        d.close()
        for required in ("CONTAINS", "MENTIONS", "RELATED_TO", "NEXT"):
            assert required in types, f"Missing relationship type: {required}"

    run_test(section, "6.02  Graph has CONTAINS, MENTIONS, RELATED_TO, NEXT rel types",
             test_rel_types, skip_if=not neo4j_available, skip_reason="Neo4j not reachable")

    # ── 6.3 Unique constraints are in place ───────────────────────────────── #
    def test_constraints():
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pw))
        with d.session() as s:
            constraints = [r.data() for r in s.run("SHOW CONSTRAINTS")]
        d.close()
        constraint_text = str(constraints).lower()
        assert "entity" in constraint_text, "Expected Entity uniqueness constraint"
        assert "chunk" in constraint_text,  "Expected Chunk uniqueness constraint"

    run_test(section, "6.03  Uniqueness constraints exist for Entity and Chunk",
             test_constraints, skip_if=not neo4j_available, skip_reason="Neo4j not reachable")

    # ── 6.4 Every Chunk belongs to at least one Document ─────────────────── #
    def test_all_chunks_have_document():
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pw))
        with d.session() as s:
            orphans = s.run(
                "MATCH (c:Chunk) WHERE NOT ()-[:CONTAINS]->(c) RETURN COUNT(c) AS cnt"
            ).single()["cnt"]
        d.close()
        assert orphans == 0, f"{orphans} orphaned Chunk nodes have no parent Document"

    run_test(section, "6.04  All Chunk nodes have a parent Document",
             test_all_chunks_have_document, skip_if=not neo4j_available, skip_reason="Neo4j not reachable")

    # ── 6.5 Multi-hop: Document → Chunk → Entity traversal works ─────────── #
    def test_multi_hop_traversal():
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pw))
        with d.session() as s:
            result = s.run(
                "MATCH (d:Document)-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity) "
                "RETURN d.id AS doc, e.name AS entity LIMIT 5"
            ).data()
        d.close()
        assert len(result) > 0, "Expected at least 1 Document→Chunk→Entity path"

    run_test(section, "6.05  Multi-hop traversal Document→Chunk→Entity succeeds",
             test_multi_hop_traversal, skip_if=not neo4j_available, skip_reason="Neo4j not reachable")

    # ── 6.6 Entity types are in the canonical set ─────────────────────────── #
    def test_entity_types_canonical():
        from neo4j import GraphDatabase
        canonical = {"PERSON", "ORG", "LOCATION", "CONCEPT", "PRODUCT", "EVENT", "OTHER"}
        d = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pw))
        with d.session() as s:
            types = {r["type"] for r in
                     s.run("MATCH (e:Entity) WHERE e.type IS NOT NULL "
                            "RETURN DISTINCT e.type AS type")}
        d.close()
        unknown = types - canonical
        assert not unknown, f"Unexpected entity types: {unknown}"

    run_test(section, "6.06  All Entity.type values are in canonical set",
             test_entity_types_canonical, skip_if=not neo4j_available, skip_reason="Neo4j not reachable")

    # ── 6.7 NEXT chain is contiguous (no skip) for a multi-chunk doc ─────── #
    def test_next_chain_contiguous():
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pw))
        with d.session() as s:
            # Find a document with multiple chunks and verify NEXT indices are sequential
            result = s.run(
                "MATCH (d:Document)-[:CONTAINS]->(c:Chunk) "
                "WITH d, collect(c.chunk_index) AS indices "
                "WHERE size(indices) > 1 "
                "RETURN d.id AS doc, indices LIMIT 1"
            ).single()
            if result:
                indices = sorted(result["indices"])
                expected = list(range(len(indices)))
                assert indices == expected, \
                    f"Chunk indices not contiguous: {indices}"
        d.close()

    run_test(section, "6.07  NEXT chain chunk indices are contiguous (0,1,2,...)",
             test_next_chain_contiguous, skip_if=not neo4j_available, skip_reason="Neo4j not reachable")


# ═══════════════════════════════════════════════════════════════════════════ #
#  SECTION 7 — Performance benchmarks                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

def run_perf_section(neo4j_available, neo4j_url, neo4j_user, neo4j_pw,
                      qdrant_available, embed_available, embed_provider):
    section = "Performance"
    print(f"\n{_c(_BOLD + _CYAN, '── Section 7: Performance Benchmarks ──')}")

    # ── 7.1 Neo4j round-trip latency < 2 s for simple MATCH ─────────────── #
    def test_neo4j_latency():
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pw))
        t0 = time.monotonic()
        for _ in range(10):
            with d.session() as s:
                s.run("MATCH (e:Entity) RETURN e LIMIT 5").data()
        elapsed = time.monotonic() - t0
        d.close()
        avg = elapsed / 10
        assert avg < 2.0, f"Neo4j avg query time {avg:.3f}s > 2s threshold"
        print(f"\n       avg Neo4j MATCH latency: {avg*1000:.1f} ms")

    run_test(section, "7.01  Neo4j: avg MATCH latency < 2 s over 10 queries",
             test_neo4j_latency, skip_if=not neo4j_available, skip_reason="Neo4j not reachable")

    # ── 7.2 Qdrant vector search latency < 3 s ───────────────────────────── #
    def test_qdrant_latency():
        from rag_sdk.factories.embedding_factory import EmbeddingFactory
        from rag_sdk.factories.vector_store_factory import VectorStoreFactory
        os.environ["COLLECTION_NAME"] = _TEST_COLLECTION
        embeddings = EmbeddingFactory.get_embeddings(provider=embed_provider)
        vs = VectorStoreFactory.get_vector_store(provider="qdrant")
        retriever = vs.get_retriever(embeddings=embeddings, k=3)

        t0 = time.monotonic()
        for q in ["CEO", "acquisition", "climate", "corporation", "Geneva"]:
            retriever.invoke(q)
        elapsed = time.monotonic() - t0
        avg = elapsed / 5
        assert avg < 3.0, f"Qdrant avg search time {avg:.3f}s > 3s threshold"
        print(f"\n       avg Qdrant search latency: {avg*1000:.1f} ms")

    run_test(section, "7.02  Qdrant: avg vector search latency < 3 s over 5 queries",
             test_qdrant_latency,
             skip_if=not (qdrant_available and embed_available),
             skip_reason="Qdrant or embedding not available")

    # ── 7.3 Entity extraction (spaCy) processes 3 docs in < 5 s ─────────── #
    def test_spacy_throughput():
        from langchain_core.documents import Document
        from rag_sdk.core.entity_extractor import EntityExtractor
        extractor = EntityExtractor(provider="spacy")
        chunks = [Document(page_content=c["content"]) for c in _CORPUS]
        t0 = time.monotonic()
        extractor.extract_from_chunks(chunks)
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0, f"spaCy extraction of 3 docs took {elapsed:.2f}s > 5s threshold"
        print(f"\n       spaCy extraction of 3 docs: {elapsed*1000:.0f} ms")

    try:
        import spacy; spacy.load("en_core_web_sm")
        _spacy_ok = True
    except Exception:
        _spacy_ok = False

    run_test(section, "7.03  spaCy: 3-doc entity extraction completes in < 5 s",
             test_spacy_throughput, skip_if=not _spacy_ok, skip_reason="spaCy not installed")


# ═══════════════════════════════════════════════════════════════════════════ #
#  SECTION 8 — Cleanup                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def run_cleanup_section(neo4j_available, neo4j_url, neo4j_user, neo4j_pw,
                         qdrant_available, qdrant_url):
    section = "Cleanup"
    print(f"\n{_c(_BOLD + _CYAN, '── Section 8: Cleanup ──')}")

    # ── 8.1 Delete live_test nodes from Neo4j ─────────────────────────────── #
    def test_neo4j_cleanup():
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pw))
        with d.session() as s:
            # Remove all chunks created by live tests
            s.run(
                "MATCH (d:Document) WHERE d.id STARTS WITH 'live_test' "
                "OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk) "
                "DETACH DELETE d, c"
            )
            # Remove entities that are now orphaned (no MENTIONS edges)
            s.run(
                "MATCH (e:Entity) WHERE NOT ()-[:MENTIONS]->(e) DETACH DELETE e"
            )
        d.close()

    run_test(section, "8.01  Neo4j: delete all live_test Documents, Chunks, orphan Entities",
             test_neo4j_cleanup, skip_if=not neo4j_available, skip_reason="Neo4j not reachable")

    # ── 8.2 Verify Neo4j live_test nodes are gone ─────────────────────────── #
    def test_neo4j_cleanup_verified():
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pw))
        with d.session() as s:
            count = s.run(
                "MATCH (d:Document) WHERE d.id STARTS WITH 'live_test' RETURN COUNT(d) AS cnt"
            ).single()["cnt"]
        d.close()
        assert count == 0, f"Expected 0 live_test Documents after cleanup, found {count}"

    run_test(section, "8.02  Neo4j: verify no live_test Documents remain",
             test_neo4j_cleanup_verified, skip_if=not neo4j_available, skip_reason="Neo4j not reachable")

    # ── 8.3 Delete Qdrant test collection ────────────────────────────────── #
    def test_qdrant_cleanup():
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url)
        names = [c.name for c in client.get_collections().collections]
        if _TEST_COLLECTION in names:
            client.delete_collection(_TEST_COLLECTION)
        names2 = [c.name for c in client.get_collections().collections]
        assert _TEST_COLLECTION not in names2

    run_test(section, "8.03  Qdrant: delete live_test_collection",
             test_qdrant_cleanup, skip_if=not qdrant_available, skip_reason="Qdrant not reachable")


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main entry point                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def main():
    parser = argparse.ArgumentParser(
        description="Manual live integration tests for the Knowledge Graph RAG pipeline."
    )
    parser.add_argument(
        "--section",
        default="all",
        choices=["all", "neo4j", "qdrant", "entity", "hybrid", "sdk", "graph_schema", "perf", "cleanup"],
        help="Which section to run (default: all)",
    )
    args = parser.parse_args()

    # ── Load .env ─────────────────────────────────────────────────────────── #
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # ── Read config ───────────────────────────────────────────────────────── #
    NEO4J_URL  = os.environ.get("NEO4J_URL",      "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USERNAME",  "neo4j")
    NEO4J_PW   = os.environ.get("NEO4J_PASSWORD",  "password")
    QDRANT_URL = os.environ.get("QDRANT_URL",      "http://localhost:6333")
    EMBED_PROV = os.environ.get("EMBEDDING_PROVIDER", "ollama")
    LLM_PROV   = os.environ.get("LLM_PROVIDER",    "gemini")

    # ── Probe availability ─────────────────────────────────────────────────── #
    print(_c(_BOLD, "\nProbing service availability…"))
    neo4j_ok   = _neo4j_available(NEO4J_URL, NEO4J_USER, NEO4J_PW)
    qdrant_ok  = _qdrant_available(QDRANT_URL)
    ollama_ok  = _ollama_available() if EMBED_PROV == "ollama" else False
    spacy_ok   = _spacy_available()

    # Embedding availability — fall back to Gemini when Ollama is configured
    # but not reachable, so embedding-dependent tests can still run.
    if EMBED_PROV == "ollama":
        if ollama_ok:
            embed_ok = True
        elif _google_key_present():
            EMBED_PROV = "gemini"
            embed_ok = True
            print(_c(_YELLOW,
                     "  ! Ollama unreachable — falling back to Gemini for embeddings"))
        else:
            embed_ok = False
    elif EMBED_PROV == "openai":
        embed_ok = _openai_key_present()
    elif EMBED_PROV == "gemini":
        embed_ok = _google_key_present()
    else:
        embed_ok = False

    # LLM availability
    if LLM_PROV == "openai":
        llm_ok = _openai_key_present()
    elif LLM_PROV in ("gemini",):
        llm_ok = _google_key_present()
    elif LLM_PROV == "anthropic":
        llm_ok = _anthropic_key_present()
    else:
        llm_ok = False

    def _status(ok): return _c(_GREEN, "✓") if ok else _c(_RED, "✗")

    print(f"  {_status(neo4j_ok)}  Neo4j        {NEO4J_URL}")
    print(f"  {_status(qdrant_ok)}  Qdrant       {QDRANT_URL}")
    print(f"  {_status(embed_ok)}  Embeddings   provider={EMBED_PROV}")
    print(f"  {_status(llm_ok)}  LLM          provider={LLM_PROV}")
    print(f"  {_status(spacy_ok)}  spaCy        en_core_web_sm")

    s = args.section

    if s in ("all", "neo4j"):
        run_neo4j_section(NEO4J_URL, NEO4J_USER, NEO4J_PW, neo4j_ok)

    if s in ("all", "qdrant"):
        run_qdrant_section(QDRANT_URL, qdrant_ok, embed_ok, EMBED_PROV)

    if s in ("all", "entity"):
        run_entity_section(llm_ok, LLM_PROV, spacy_ok)

    if s in ("all", "hybrid"):
        run_hybrid_section(
            neo4j_ok, qdrant_ok, embed_ok,
            NEO4J_URL, NEO4J_USER, NEO4J_PW,
            QDRANT_URL, EMBED_PROV, spacy_ok,
        )

    if s in ("all", "sdk"):
        run_sdk_section(
            neo4j_ok, qdrant_ok, embed_ok, llm_ok,
            LLM_PROV, EMBED_PROV, spacy_ok,
        )

    if s in ("all", "graph_schema"):
        run_graph_schema_section(neo4j_ok, NEO4J_URL, NEO4J_USER, NEO4J_PW)

    if s in ("all", "perf"):
        run_perf_section(
            neo4j_ok, NEO4J_URL, NEO4J_USER, NEO4J_PW,
            qdrant_ok, embed_ok, EMBED_PROV,
        )

    if s in ("all", "cleanup"):
        run_cleanup_section(neo4j_ok, NEO4J_URL, NEO4J_USER, NEO4J_PW,
                             qdrant_ok, QDRANT_URL)

    ok = _suite.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

