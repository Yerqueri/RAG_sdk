from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredExcelLoader
from .base_loader_strategy import BaseLoaderStrategy

class ExcelLoaderStrategy(BaseLoaderStrategy):
    def load(self) -> List[Document]:
        loader = UnstructuredExcelLoader(self.file_path, mode="elements")
        return loader.load()
