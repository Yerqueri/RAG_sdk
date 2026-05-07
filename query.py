import argparse
from rag_sdk.sdk import RAGClient


def main():
    parser = argparse.ArgumentParser(
        description="Query the Knowledge Graph RAG pipeline."
    )
    parser.add_argument("query", type=str, help="The question you want to ask.")
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default=None,
        choices=["vector", "graph", "hybrid"],
        help=(
            "Retrieval strategy: 'vector' (default), 'graph' (graph-only), "
            "or 'hybrid' (vector + graph). Overrides RETRIEVAL_MODE env var."
        ),
    )
    args = parser.parse_args()

    client = RAGClient(retrieval_mode=args.retrieval_mode)
    response = client.query(args.query)
    print(f"\nAnswer: {response}")


if __name__ == "__main__":
    main()
