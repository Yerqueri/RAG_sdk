from .base_llm_strategy import BaseLLMStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from rag_sdk.core.config import config

class AnthropicStrategy(BaseLLMStrategy):
    def get_llm(self) -> BaseChatModel:
        return ChatAnthropic(model=config.anthropic_llm_model, temperature=0)
