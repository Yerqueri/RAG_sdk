from typing import Dict, List

from langchain_core.documents import Document
from neo4j import GraphDatabase

from .base_graph_strategy import BaseGraphStrategy


class Neo4jStrategy(BaseGraphStrategy):
    """
    Stores and retrieves a knowledge graph in Neo4j.

    Graph schema
    ─────────────
    Nodes  : (:Document {id, source}), (:Chunk {id, content, chunk_index}),
             (:Entity {name, type})
    Edges  : (Document)-[:CONTAINS]->(Chunk)
             (Chunk)-[:MENTIONS]->(Entity)
             (Entity)-[:RELATED_TO {relation}]->(Entity)
             (Chunk)-[:NEXT]->(Chunk)   ← sequential order
    """

    def __init__(self, url: str, username: str, password: str):
        self.driver = GraphDatabase.driver(url, auth=(username, password))
        self._create_constraints()

    # ------------------------------------------------------------------ #
    #  Schema bootstrap                                                    #
    # ------------------------------------------------------------------ #

    def _create_constraints(self) -> None:
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT document_id IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.id IS UNIQUE"
            )

    # ------------------------------------------------------------------ #
    #  Ingestion                                                           #
    # ------------------------------------------------------------------ #

    def store_graph(
        self,
        doc_id: str,
        chunks: List[Document],
        entities_per_chunk: Dict[int, List[dict]],
    ) -> None:
        with self.driver.session() as session:
            # Upsert Document node
            session.run(
                "MERGE (d:Document {id: $id}) SET d.source = $source",
                id=doc_id,
                source=doc_id,
            )

            prev_chunk_id = None
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}::chunk_{idx}"

                # Upsert Chunk node and link to Document
                session.run(
                    """
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.content = $content, c.chunk_index = $idx
                    WITH c
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:CONTAINS]->(c)
                    """,
                    chunk_id=chunk_id,
                    content=chunk.page_content,
                    idx=idx,
                    doc_id=doc_id,
                )

                # Sequential NEXT edge
                if prev_chunk_id:
                    session.run(
                        """
                        MATCH (prev:Chunk {id: $prev_id}), (curr:Chunk {id: $curr_id})
                        MERGE (prev)-[:NEXT]->(curr)
                        """,
                        prev_id=prev_chunk_id,
                        curr_id=chunk_id,
                    )
                prev_chunk_id = chunk_id

                # Entities for this chunk
                for entity_dict in entities_per_chunk.get(idx, []):
                    entity_name = entity_dict.get("entity", "").strip()
                    entity_type = entity_dict.get("type", "OTHER").strip()
                    if not entity_name:
                        continue

                    # Upsert Entity and MENTIONS edge
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type
                        WITH e
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:MENTIONS]->(e)
                        """,
                        name=entity_name,
                        type=entity_type,
                        chunk_id=chunk_id,
                    )

                    # RELATED_TO edges
                    for rel in entity_dict.get("relations", []):
                        target = rel.get("target", "").strip()
                        relation = rel.get("relation", "RELATED_TO").strip()
                        if not target:
                            continue
                        session.run(
                            """
                            MERGE (src:Entity {name: $src})
                            MERGE (tgt:Entity {name: $tgt})
                            MERGE (src)-[:RELATED_TO {relation: $relation}]->(tgt)
                            """,
                            src=entity_name,
                            tgt=target,
                            relation=relation,
                        )

        print(f"Stored graph for '{doc_id}': {len(chunks)} chunks.")

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def get_related_chunks(self, entity_names: List[str], k: int = 5) -> List[str]:
        if not entity_names:
            return []
        names_lower = [n.lower() for n in entity_names]
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE toLower(e.name) IN $names
                RETURN DISTINCT c.content AS content
                LIMIT $k
                """,
                names=names_lower,
                k=k,
            )
            return [record["content"] for record in result if record["content"]]

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self.driver.close()

