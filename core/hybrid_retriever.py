from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict


class HybridRetriever(BaseRetriever):
    """
    Fuses vector similarity search with knowledge-graph traversal.

    Retrieval modes
    ───────────────
    * **hybrid** (default) – vector results first, then graph-only additions.
    * **graph**            – set ``vector_retriever=None`` to skip vector search.
    * **vector**           – set ``graph_store=None`` to skip graph traversal.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_retriever: Optional[Any] = None   # LangChain BaseRetriever or None
    graph_store: Optional[Any] = None        # core.graph_store.GraphStore or None
    entity_extractor: Optional[Any] = None   # core.entity_extractor.EntityExtractor or None
    k: int = 3

    # ------------------------------------------------------------------ #

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        docs: List[Document] = []
        seen_content: set = set()

        # ── 1. Vector search ──────────────────────────────────────────── #
        if self.vector_retriever is not None:
            try:
                vector_docs = self.vector_retriever.invoke(query)
                for doc in vector_docs:
                    if doc.page_content not in seen_content:
                        docs.append(doc)
                        seen_content.add(doc.page_content)
            except Exception as exc:
                print(f"[HybridRetriever] Vector search failed: {exc}")

        # ── 2. Graph search ───────────────────────────────────────────── #
        if self.graph_store is not None and self.entity_extractor is not None:
            try:
                entities = self.entity_extractor.extract(query)
                entity_names = [e["entity"] for e in entities if e.get("entity")]
                if entity_names:
                    print(
                        f"[HybridRetriever] Graph search for entities: {entity_names}"
                    )
                    graph_chunks = self.graph_store.get_related_chunks(
                        entity_names, k=self.k
                    )
                    for content in graph_chunks:
                        if content and content not in seen_content:
                            docs.append(
                                Document(
                                    page_content=content,
                                    metadata={"source": "knowledge_graph"},
                                )
                            )
                            seen_content.add(content)
            except Exception as exc:
                print(f"[HybridRetriever] Graph search failed: {exc}")

        return docs

