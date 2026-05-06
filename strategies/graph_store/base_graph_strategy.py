from abc import ABC, abstractmethod
from typing import Dict, List
from langchain_core.documents import Document


class BaseGraphStrategy(ABC):
    @abstractmethod
    def store_graph(
        self,
        doc_id: str,
        chunks: List[Document],
        entities_per_chunk: Dict[int, List[dict]],
    ) -> None:
        """
        Persist a document's chunk/entity graph.

        :param doc_id: Unique identifier for the source document (e.g. file path).
        :param chunks: Ordered list of LangChain Document chunks.
        :param entities_per_chunk: Mapping of chunk index → list of entity dicts.
            Each entity dict has the shape:
            {
                "entity": str,
                "type": str,
                "relations": [{"target": str, "relation": str}, ...]
            }
        """
        pass

    @abstractmethod
    def get_related_chunks(self, entity_names: List[str], k: int = 5) -> List[str]:
        """
        Return up to k chunk content strings whose graph nodes mention any of
        the supplied entity names.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release any underlying connections."""
        pass

