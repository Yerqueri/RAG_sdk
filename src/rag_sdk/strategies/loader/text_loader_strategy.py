from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from .base_loader_strategy import BaseLoaderStrategy

class TextLoaderStrategy(BaseLoaderStrategy):
    def load(self) -> List[Document]:
        loader = TextLoader(self.file_path)
        return loader.load()
