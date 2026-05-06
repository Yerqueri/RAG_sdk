from typing import Dict, List, Optional

from langchain_core.documents import Document

from factories.graph_store_factory import GraphStoreFactory
from strategies.graph_store.base_graph_strategy import BaseGraphStrategy


class GraphStore:
    """
    Thin facade over a :class:`BaseGraphStrategy` that mirrors the API of
    ``core/vector_store.py`` for consistency.
    """

    def __init__(self, provider: Optional[str] = None):
        self.strategy: BaseGraphStrategy = GraphStoreFactory.get_graph_store(
            provider=provider
        )

    def store_graph(
        self,
        doc_id: str,
        chunks: List[Document],
        entities_per_chunk: Dict[int, List[dict]],
    ) -> None:
        self.strategy.store_graph(doc_id, chunks, entities_per_chunk)

    def get_related_chunks(
        self, entity_names: List[str], k: int = 5
    ) -> List[str]:
        return self.strategy.get_related_chunks(entity_names, k=k)

    def close(self) -> None:
        self.strategy.close()

