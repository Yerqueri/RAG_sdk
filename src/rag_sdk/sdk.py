from collections import defaultdict
from typing import Optional

from rag_sdk.core.config import config
from rag_sdk.core.document_loader import DocumentLoader
from rag_sdk.core.text_chunker import TextChunker
from rag_sdk.factories.embedding_factory import EmbeddingFactory
from rag_sdk.factories.llm_factory import LLMFactory
from rag_sdk.factories.vector_store_factory import VectorStoreFactory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RAGClient:
    def __init__(
        self,
        llm_provider: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        vector_db_provider: Optional[str] = None,
        # ── Graph / Hybrid options ──────────────────────────────────── #
        enable_graph: bool = False,
        retrieval_mode: Optional[str] = None,          # "vector" | "graph" | "hybrid"
        entity_extraction_provider: Optional[str] = None,  # "llm" | "spacy"
        graph_store_provider: Optional[str] = None,    # "neo4j"
    ):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.vector_db_provider = vector_db_provider
        self.enable_graph = enable_graph
        self.retrieval_mode = retrieval_mode
        self.entity_extraction_provider = entity_extraction_provider
        self.graph_store_provider = graph_store_provider

    # ------------------------------------------------------------------ #
    #  Ingestion                                                           #
    # ------------------------------------------------------------------ #

    def ingest_file(self, file_path: str):
        print(f"Ingesting single file: {file_path}")
        docs = DocumentLoader.load_file(file_path)
        self._ingest_documents(docs)

    def ingest_directory(self, dir_path: Optional[str] = None):
        target_dir = dir_path or config.data_dir
        print(f"Ingesting directory: {target_dir}")
        docs = DocumentLoader.load_directory(target_dir)
        self._ingest_documents(docs)

    def _ingest_documents(self, documents):
        chunker = TextChunker()
        chunks = chunker.split(documents)

        # Stamp each chunk with a stable index so the graph can reference it
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        print("Initializing Embeddings API...")
        embeddings = EmbeddingFactory.get_embeddings(provider=self.embedding_provider)

        print("Initializing Vector DB...")
        vector_store = VectorStoreFactory.get_vector_store(provider=self.vector_db_provider)
        vector_store.store(chunks, embeddings)

        # ── Optional graph ingestion ─────────────────────────────────── #
        if self.enable_graph:
            print("\nExtracting entities and building knowledge graph...")
            from rag_sdk.core.entity_extractor import EntityExtractor
            from rag_sdk.core.graph_store import GraphStore

            extractor = EntityExtractor(
                provider=self.entity_extraction_provider,
                llm_provider=self.llm_provider,
            )
            entities_per_chunk = extractor.extract_from_chunks(chunks)

            # Group chunks by source document
            by_source: dict = defaultdict(list)
            by_source_idx: dict = defaultdict(list)
            for i, chunk in enumerate(chunks):
                src = chunk.metadata.get("source", "unknown_document")
                by_source[src].append(chunk)
                by_source_idx[src].append(i)

            graph_store = GraphStore(provider=self.graph_store_provider)
            for source, src_chunks in by_source.items():
                # Re-map global indices to per-document indices
                src_entities = {
                    j: entities_per_chunk[global_idx]
                    for j, global_idx in enumerate(by_source_idx[source])
                }
                graph_store.store_graph(source, src_chunks, src_entities)

            graph_store.close()
            print("Knowledge graph built successfully!\n")

    # ------------------------------------------------------------------ #
    #  Query                                                               #
    # ------------------------------------------------------------------ #

    def query(self, text: str) -> str:
        # Resolve retrieval mode: explicit param > .env > "vector"
        mode = self.retrieval_mode
        if mode is None:
            try:
                mode = config.retrieval_mode
            except ValueError:
                mode = "vector"

        print("Initializing Embeddings API and connecting to Vector DB...")
        embeddings = EmbeddingFactory.get_embeddings(provider=self.embedding_provider)
        vector_store = VectorStoreFactory.get_vector_store(provider=self.vector_db_provider)

        # ── Build retriever ──────────────────────────────────────────── #
        if mode in ("graph", "hybrid"):
            from rag_sdk.core.entity_extractor import EntityExtractor
            from rag_sdk.core.graph_store import GraphStore
            from rag_sdk.core.hybrid_retriever import HybridRetriever

            graph_store = GraphStore(provider=self.graph_store_provider)
            entity_extractor = EntityExtractor(
                provider=self.entity_extraction_provider,
                llm_provider=self.llm_provider,
            )

            if mode == "hybrid":
                vector_retriever = vector_store.get_retriever(embeddings=embeddings, k=3)
            else:
                vector_retriever = None  # graph-only

            retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                graph_store=graph_store,
                entity_extractor=entity_extractor,
                k=5,
            )
        else:
            retriever = vector_store.get_retriever(embeddings=embeddings, k=3)

        # ── LLM chain (unchanged) ────────────────────────────────────── #
        print("Initializing Generation API...")
        llm = LLMFactory.get_llm(provider=self.llm_provider)

        template = """Answer the question based ONLY on the following context:

{context}

Question: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print(f"\n--- Generating Answer [{mode} retrieval] for: '{text}' ---")
        return rag_chain.invoke(text)
