from langchain_core.language_models.chat_models import BaseChatModel
from strategies.llm.openai_strategy import OpenAIStrategy
from strategies.llm.gemini_strategy import GeminiStrategy
from strategies.llm.anthropic_strategy import AnthropicStrategy
from strategies.llm.openrouter_strategy import OpenRouterStrategy
from core.config import config

class LLMFactory:
    @staticmethod
    def get_llm(provider: str = None) -> BaseChatModel:
        provider = provider or config.llm_provider
        
        if provider == "openai":
            return OpenAIStrategy().get_llm()
        elif provider == "gemini":
            return GeminiStrategy().get_llm()
        elif provider == "anthropic":
            return AnthropicStrategy().get_llm()
        elif provider == "openrouter":
            return OpenRouterStrategy().get_llm()
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}. Use 'openai', 'gemini', 'anthropic', or 'openrouter'.")
