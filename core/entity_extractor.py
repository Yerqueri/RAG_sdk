from typing import Dict, List, Optional

from langchain_core.documents import Document

from factories.entity_extraction_factory import EntityExtractionFactory
from strategies.entity_extraction.base_entity_extraction_strategy import (
    BaseEntityExtractionStrategy,
)


class EntityExtractor:
    """
    Wraps an :class:`BaseEntityExtractionStrategy` with higher-level helpers
    used by the ingestion and query pipelines.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ):
        self.strategy: BaseEntityExtractionStrategy = (
            EntityExtractionFactory.get_extractor(
                provider=provider, llm_provider=llm_provider
            )
        )

    def extract(self, text: str) -> List[dict]:
        """Extract entities from a single piece of text."""
        return self.strategy.extract(text)

    def extract_from_chunks(
        self, chunks: List[Document]
    ) -> Dict[int, List[dict]]:
        """
        Extract entities from every chunk in *chunks*.

        Returns a dict mapping chunk index → list of entity dicts.
        """
        results: Dict[int, List[dict]] = {}
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            print(f"  Extracting entities from chunk {i + 1}/{total}...")
            results[i] = self.strategy.extract(chunk.page_content)
        return results

