from core.config import config
from strategies.graph_store.base_graph_strategy import BaseGraphStrategy
from strategies.graph_store.neo4j_strategy import Neo4jStrategy


class GraphStoreFactory:
    @staticmethod
    def get_graph_store(provider: str = None) -> BaseGraphStrategy:
        provider = provider or config.graph_store_provider

        if provider == "neo4j":
            return Neo4jStrategy(
                url=config.neo4j_url,
                username=config.neo4j_username,
                password=config.neo4j_password,
            )
        else:
            raise ValueError(
                f"Unsupported GRAPH_STORE_PROVIDER: '{provider}'. Currently supported: 'neo4j'."
            )

