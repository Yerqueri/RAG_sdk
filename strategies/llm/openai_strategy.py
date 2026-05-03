from .base_llm_strategy import BaseLLMStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from core.config import config

class OpenAIStrategy(BaseLLMStrategy):
    def get_llm(self) -> BaseChatModel:
        return ChatOpenAI(model=config.openai_llm_model, temperature=0)
