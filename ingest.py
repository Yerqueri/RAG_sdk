import argparse
from rag_sdk.sdk import RAGClient


def main():
    parser = argparse.ArgumentParser(
        description="Ingest files into the Knowledge Graph RAG pipeline."
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Optional: path to a single file. If omitted, ingests the default data directory.",
    )
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        default=False,
        help="Extract entities and build a Neo4j knowledge graph alongside the vector store.",
    )
    parser.add_argument(
        "--entity-provider",
        type=str,
        default=None,
        choices=["llm", "spacy"],
        help="Entity extraction backend (default: from ENTITY_EXTRACTION_PROVIDER env var → 'llm').",
    )
    args = parser.parse_args()

    client = RAGClient(
        enable_graph=args.enable_graph,
        entity_extraction_provider=args.entity_provider,
    )

    if args.file:
        client.ingest_file(args.file)
    else:
        client.ingest_directory()


if __name__ == "__main__":
    main()
