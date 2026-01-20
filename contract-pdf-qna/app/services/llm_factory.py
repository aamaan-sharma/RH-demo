"""LLM factory for creating and caching LLM instances."""
from typing import Optional, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config.settings import settings


class LLMFactory:
    """Factory for creating and caching LLM instances."""
    
    def __init__(self):
        """Initialize LLM factory with caching."""
        self._chat_llm_cache: Dict[str, ChatOpenAI] = {}
        self._embedding_cache: Optional[OpenAIEmbeddings] = None
    
    def create_chat_llm(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        cache_key: Optional[str] = None
    ) -> ChatOpenAI:
        """Create or retrieve cached ChatOpenAI instance.
        
        Args:
            model: Model name (defaults to gpt-4o)
            temperature: Temperature setting
            max_tokens: Maximum tokens
            timeout: Request timeout in seconds
            cache_key: Optional cache key for this configuration
            
        Returns:
            ChatOpenAI instance
        """
        if model is None:
            model = "gpt-4o"
        
        # Create cache key if not provided
        if cache_key is None:
            cache_key = f"{model}_{temperature}_{max_tokens}_{timeout}"
        
        # Return cached instance if available
        if cache_key in self._chat_llm_cache:
            return self._chat_llm_cache[cache_key]
        
        # Create new instance
        kwargs = {
            "temperature": temperature,
            "model": model,
            "openai_api_key": settings.OPENAI_API_KEY,
        }
        
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        
        if timeout is not None:
            kwargs["timeout"] = timeout
        
        llm = ChatOpenAI(**kwargs)
        
        # Cache the instance
        self._chat_llm_cache[cache_key] = llm
        
        return llm
    
    def create_embedding(self, model: str = "text-embedding-ada-002") -> OpenAIEmbeddings:
        """Create or retrieve cached OpenAIEmbeddings instance.
        
        Args:
            model: Embedding model name
            
        Returns:
            OpenAIEmbeddings instance
        """
        if self._embedding_cache is None:
            self._embedding_cache = OpenAIEmbeddings(
                model=model,
                openai_api_key=settings.OPENAI_API_KEY
            )
        
        return self._embedding_cache
    
    def create_standalone_llm(self, model: Optional[str] = None) -> ChatOpenAI:
        """Create LLM for standalone question rephrasing.
        
        Args:
            model: Model name (defaults to fine-tuned model)
            
        Returns:
            ChatOpenAI instance
        """
        if model is None:
            model = "ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA"
        
        return self.create_chat_llm(
            model=model,
            temperature=0.0,
            cache_key=f"standalone_{model}"
        )
    
    def create_intent_llm(self) -> ChatOpenAI:
        """Create LLM for intent detection.
        
        Returns:
            ChatOpenAI instance configured for intent detection
        """
        return self.create_chat_llm(
            model=settings.MODEL_INTENT,
            temperature=0.0,
            max_tokens=200,
            timeout=10.0,
            cache_key="intent"
        )
    
    def create_suggest_llm(self) -> ChatOpenAI:
        """Create LLM for suggestion generation.
        
        Returns:
            ChatOpenAI instance configured for suggestions
        """
        return self.create_chat_llm(
            model=settings.MODEL_SUGGEST,
            temperature=0.0,
            max_tokens=500,
            timeout=15.0,
            cache_key="suggest"
        )
    
    def create_diagnostics_llm(self) -> ChatOpenAI:
        """Create LLM for diagnostics.
        
        Returns:
            ChatOpenAI instance configured for diagnostics
        """
        return self.create_chat_llm(
            model=settings.MODEL_SUGGEST,
            temperature=0.2,
            max_tokens=300,
            timeout=10.0,
            cache_key="diagnostics"
        )


# Global factory instance
_llm_factory: Optional[LLMFactory] = None


def get_llm_factory() -> LLMFactory:
    """Get or create global LLM factory instance.
    
    Returns:
        LLMFactory instance
    """
    global _llm_factory
    if _llm_factory is None:
        _llm_factory = LLMFactory()
    return _llm_factory
