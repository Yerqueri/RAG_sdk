from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseLoaderStrategy(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def load(self) -> List[Document]:
        pass
