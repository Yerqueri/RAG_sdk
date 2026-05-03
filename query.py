import argparse
from sdk import RAGClient

def main():
    parser = argparse.ArgumentParser(description="Query the Hybrid RAG pipeline.")
    parser.add_argument("query", type=str, help="The question you want to ask.")
    args = parser.parse_args()

    client = RAGClient()
    response = client.query(args.query)
    print(f"\nAnswer: {response}")

if __name__ == "__main__":
    main()
