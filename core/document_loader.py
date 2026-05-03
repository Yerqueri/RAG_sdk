import os
from typing import List
from langchain_core.documents import Document
from factories.loader_factory import LoaderFactory

class DocumentLoader:
    @staticmethod
    def load_file(file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Loading file: {file_path}")
        strategy = LoaderFactory.get_loader(file_path)
        return strategy.load()

    @staticmethod
    def load_directory(data_dir: str) -> List[Document]:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            raise FileNotFoundError(f"Created directory {data_dir}. Please place files inside and try again.")
        
        documents = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    docs = DocumentLoader.load_file(file_path)
                    documents.extend(docs)
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
        
        if not documents:
            raise ValueError(f"No valid documents found in {data_dir}.")
            
        print(f"Total documents loaded from directory: {len(documents)}")
        return documents
