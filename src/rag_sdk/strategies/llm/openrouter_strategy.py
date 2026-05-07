from .base_llm_strategy import BaseLLMStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from rag_sdk.core.config import config

class OpenRouterStrategy(BaseLLMStrategy):
    def get_llm(self) -> BaseChatModel:
        return ChatOpenAI(
            model=config.openrouter_llm_model, 
            temperature=0,
            openai_api_key=config.get_env_var("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )
