"""
Абстрактный слой для работы с различными LLM провайдерами.
Поддерживает OpenRouter и локальную модель через Ollama.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


class LLMClient(ABC):
    """Абстрактный класс клиента LLM."""
    
    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, Any]], 
                            tools: Optional[List[Dict[str, Any]]] = None, 
                            max_tokens: int = 1024) -> Dict[str, Any]:
        """Выполняет запрос к LLM."""
        pass
    
    @abstractmethod
    def supports_tools(self) -> bool:
        """Поддерживает ли клиент function calling (tools)."""
        pass


class OpenRouterClient(LLMClient):
    """Клиент для работы с OpenRouter."""
    
    def __init__(self, model: str = "openrouter/auto", api_key: Optional[str] = None):
        """
        Инициализация клиента OpenRouter.
        
        Args:
            model: Название модели OpenRouter
            api_key: API ключ OpenRouter (если None, берётся из окружения)
        """
        self.model = model
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY не найден в окружении")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0,
            max_retries=2
        )
    
    async def chat_completion(self, messages: List[Dict[str, Any]], 
                            tools: Optional[List[Dict[str, Any]]] = None, 
                            max_tokens: int = 1024) -> Dict[str, Any]:
        """Выполняет запрос к OpenRouter."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                max_tokens=max_tokens
            )
            
            return {
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls,
                "model": response.model,
                "usage": response.usage.dict() if response.usage else None
            }
        except Exception as e:
            return {
                "content": f"Ошибка при вызове OpenRouter: {str(e)}",
                "tool_calls": None,
                "model": self.model,
                "error": str(e)
            }
    
    def supports_tools(self) -> bool:
        """OpenRouter поддерживает function calling."""
        return True


class OllamaClient(LLMClient):
    """Клиент для работы с локальной моделью через Ollama."""
    
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_URL):
        """
        Инициализация клиента Ollama.
        
        Args:
            model: Название модели Ollama (например, "llama3.2:3b")
            base_url: URL сервера Ollama
        """
        self.model = model
        self.base_url = base_url
        
        # Ollama предоставляет OpenAI-совместимый API через /v1
        self.client = AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="ollama",  # Ollama не требует реального ключа
            timeout=120.0
        )
    
    async def chat_completion(self, messages: List[Dict[str, Any]], 
                            tools: Optional[List[Dict[str, Any]]] = None, 
                            max_tokens: int = 1024) -> Dict[str, Any]:
        """Выполняет запрос к локальной модели через Ollama."""
        try:
            # Ollama не поддерживает tools, поэтому игнорируем их
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )
            
            return {
                "content": response.choices[0].message.content,
                "tool_calls": None,  # Ollama не возвращает tool_calls
                "model": response.model,
                "usage": response.usage.dict() if response.usage else None
            }
        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg or "Failed to connect" in error_msg:
                error_msg = "Не удалось подключиться к серверу Ollama. Убедитесь, что сервер запущен."
            
            return {
                "content": f"⚠️ Ошибка при вызове локальной модели: {error_msg}",
                "tool_calls": None,
                "model": self.model,
                "error": error_msg
            }
    
    def supports_tools(self) -> bool:
        """Ollama не поддерживает function calling."""
        return False


def create_llm_client(model: str, use_local: bool = False) -> LLMClient:
    """
    Фабрика для создания клиента LLM.
    
    Args:
        model: Название модели
        use_local: Использовать локальную модель Ollama
        
    Returns:
        LLMClient: Экземпляр клиента LLM
    """
    if use_local:
        return OllamaClient(model=model)
    else:
        return OpenRouterClient(model=model)