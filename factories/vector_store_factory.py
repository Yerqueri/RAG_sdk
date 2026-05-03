from strategies.vector_store.base_vector_strategy import BaseVectorStrategy
from strategies.vector_store.qdrant_strategy import QdrantStrategy
from strategies.vector_store.chroma_strategy import ChromaStrategy
from core.config import config

class VectorStoreFactory:
    @staticmethod
    def get_vector_store(provider: str = None) -> BaseVectorStrategy:
        provider = provider or config.vector_db_provider
        
        if provider == "qdrant":
            return QdrantStrategy(url=config.qdrant_url, collection_name=config.collection_name)
        elif provider == "chroma":
            return ChromaStrategy(persist_dir=config.chroma_persist_dir, collection_name=config.collection_name)
        else:
            raise ValueError(f"Unsupported VECTOR_DB_PROVIDER: {provider}. Use 'qdrant' or 'chroma'.")
