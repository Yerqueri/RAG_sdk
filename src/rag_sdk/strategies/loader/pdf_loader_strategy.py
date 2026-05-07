from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from .base_loader_strategy import BaseLoaderStrategy

class PDFLoaderStrategy(BaseLoaderStrategy):
    def load(self) -> List[Document]:
        loader = PyPDFLoader(self.file_path)
        return loader.load()
