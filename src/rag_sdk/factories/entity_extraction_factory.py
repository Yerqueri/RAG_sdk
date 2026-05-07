from rag_sdk.core.config import config
from rag_sdk.strategies.entity_extraction.base_entity_extraction_strategy import BaseEntityExtractionStrategy


class EntityExtractionFactory:
    @staticmethod
    def get_extractor(
        provider: str = None,
        llm_provider: str = None,
    ) -> BaseEntityExtractionStrategy:
        provider = provider or config.entity_extraction_provider

        if provider == "llm":
            from rag_sdk.strategies.entity_extraction.llm_entity_extraction_strategy import (
                LLMEntityExtractionStrategy,
            )
            return LLMEntityExtractionStrategy(llm_provider=llm_provider)

        elif provider == "spacy":
            from rag_sdk.strategies.entity_extraction.spacy_entity_extraction_strategy import (
                SpacyEntityExtractionStrategy,
            )
            return SpacyEntityExtractionStrategy()

        else:
            raise ValueError(
                f"Unsupported ENTITY_EXTRACTION_PROVIDER: '{provider}'. "
                "Use 'llm' or 'spacy'."
            )

