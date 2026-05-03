from typing import Optional
from core.config import config
from core.document_loader import DocumentLoader
from core.text_chunker import TextChunker
from factories.embedding_factory import EmbeddingFactory
from factories.llm_factory import LLMFactory
from factories.vector_store_factory import VectorStoreFactory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGClient:
    def __init__(
        self, 
        llm_provider: Optional[str] = None, 
        embedding_provider: Optional[str] = None, 
        vector_db_provider: Optional[str] = None
    ):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.vector_db_provider = vector_db_provider

    def ingest_file(self, file_path: str):
        print(f"Ingesting single file: {file_path}")
        docs = DocumentLoader.load_file(file_path)
        self._ingest_documents(docs)

    def ingest_directory(self, dir_path: Optional[str] = None):
        target_dir = dir_path or config.data_dir
        print(f"Ingesting directory: {target_dir}")
        docs = DocumentLoader.load_directory(target_dir)
        self._ingest_documents(docs)

    def _ingest_documents(self, documents):
        chunker = TextChunker()
        chunks = chunker.split(documents)

        print(f"Initializing Embeddings API...")
        embeddings = EmbeddingFactory.get_embeddings(provider=self.embedding_provider)

        print("Initializing Vector DB...")
        vector_store = VectorStoreFactory.get_vector_store(provider=self.vector_db_provider)
        
        vector_store.store(chunks, embeddings)

    def query(self, text: str) -> str:
        print("Initializing Embeddings API and connecting to Vector DB...")
        embeddings = EmbeddingFactory.get_embeddings(provider=self.embedding_provider)
        vector_store = VectorStoreFactory.get_vector_store(provider=self.vector_db_provider)
        retriever = vector_store.get_retriever(embeddings=embeddings, k=3)

        print("Initializing Generation API...")
        llm = LLMFactory.get_llm(provider=self.llm_provider)

        template = """Answer the question based ONLY on the following context:

{context}

Question: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print(f"\n--- Generating Answer for: '{text}' ---")
        return rag_chain.invoke(text)
