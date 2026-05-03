import os
from strategies.loader.base_loader_strategy import BaseLoaderStrategy
from strategies.loader.text_loader_strategy import TextLoaderStrategy
from strategies.loader.pdf_loader_strategy import PDFLoaderStrategy
from strategies.loader.excel_loader_strategy import ExcelLoaderStrategy

class LoaderFactory:
    @staticmethod
    def get_loader(file_path: str) -> BaseLoaderStrategy:
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == ".txt":
            return TextLoaderStrategy(file_path)
        elif ext == ".pdf":
            return PDFLoaderStrategy(file_path)
        elif ext in [".xlsx", ".xls"]:
            return ExcelLoaderStrategy(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext} for {file_path}")
