import argparse
from sdk import RAGClient

def main():
    parser = argparse.ArgumentParser(description="Ingest files into the Hybrid RAG pipeline.")
    parser.add_argument("--file", type=str, help="Optional: Path to a single file to ingest. If not provided, ingests the default data directory.")
    args = parser.parse_args()

    client = RAGClient()
    if args.file:
        client.ingest_file(args.file)
    else:
        client.ingest_directory()

if __name__ == "__main__":
    main()
