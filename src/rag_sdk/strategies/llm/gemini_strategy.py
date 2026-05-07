from .base_llm_strategy import BaseLLMStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_sdk.core.config import config

class GeminiStrategy(BaseLLMStrategy):
    def get_llm(self) -> BaseChatModel:
        return ChatGoogleGenerativeAI(model=config.gemini_llm_model, temperature=0)
