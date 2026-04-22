"""
Основной LLM агент для логистического ассистента.
Объединяет MCP инструменты, RAG поиск и управление диалогами.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from openai import AsyncOpenAI

from utils.config import settings
from utils.async_helpers import run_in_background
from services.mcp_orchestrator import MCPOrchestrator, MCPToolCall
# RAG service is optional
try:
    from services.rag_service import get_rag_service, RAGResult
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    # Create mock classes for type hints
    class RAGResult:
        def __init__(self):
            self.content = ""
            self.sources = []
            self.score = 0.0
    
    def get_rag_service(*args, **kwargs):
        return None
from core.conversation_manager import ConversationManager, get_conversation_manager
from core.summarizer import get_summarizer
from core.task_state import get_task_state_manager


class ToolSelection(Enum):
    """Выбор инструмента для обработки запроса."""
    MCP = "mcp"
    RAG = "rag"
    KNOWLEDGE = "knowledge"
    HYBRID = "hybrid"


@dataclass
class AgentConfig:
    """Конфигурация агента."""
    
    # Модели LLM
    default_model: str = "openrouter/free"
    fallback_model: str = "openrouter/free"
    
    # Настройки промптов
    system_prompt_template: str = """
    Ты умный логист-ассистент. У тебя есть инструменты:
    
    ## MCP ИНСТРУМЕНТЫ (для расчётов):
    - mcp_geocode_batch - координаты городов
    - mcp_find_optimal_route - оптимальный маршрут
    - mcp_calculate_cost - стоимость доставки (через API ПЭК)
    
    ## RAG (для поиска в документах):
    - Ты можешь искать информацию в документах (тарифы, правила, постановления)
    
    ## ПРАВИЛА ВЫБОРА ИНСТРУМЕНТА:
    - Если вопрос про **расстояние, маршрут, стоимость доставки, координаты** → используй MCP
    - Если вопрос про **правила, обязанности, тарифы из документов, API ПЭК** → сначала поищи в RAG
    - Если вопрос про **общие понятия** → отвечай из своих знаний
    
    ## КОНТЕКСТ ДИАЛОГА:
    {conversation_context}
    
    ## ФОРМАТ ОТВЕТА:
    - Если использовал MCP: укажи 🔧 MCP
    - Если использовал RAG: укажи 📚 RAG + источник
    - Если использовал знания: укажи 💡
    
    ОГРАНИЧЕНИЕ: Могу обработать максимум 5 городов. 
    Если пользователь указывает больше 5 городов:
    1. Используй только первые 5
    2. Сообщи пользователю: "⚠️ Я могу обработать только первые 5 городов из указанных. Будет рассчитан маршрут: [список 5 городов]"
    
    Важно: вызывай инструменты строго по порядку.
    Важно: запрещено пользоваться поиском по интернету, до тех пор пока доступны инструменты mcp
    Важно: сообщи пользователю использовал ли ты mcp или дал ответ без него (например, из-за недоступности сервиса)
    
    Если пользователь не указал города или указал менее двух, попроси уточнить.
    Если город не найден, сообщи об этом пользователю.
    """
    
    # Настройки обработки
    max_cities: int = 5
    max_tokens: int = 2000
    temperature: float = 0.3
    
    # Настройки инструментов
    use_mcp: bool = True
    use_rag: bool = True
    auto_summarize: bool = True


@dataclass
class AgentState:
    """Состояние агента."""
    
    messages: List[Dict[str, Any]] = field(default_factory=list)
    mcp_calls: List[MCPToolCall] = field(default_factory=list)
    rag_calls: List[Dict[str, Any]] = field(default_factory=list)
    mcp_available: bool = False
    rag_available: bool = False
    current_conversation_id: Optional[int] = None
    model: str = "openrouter/free"
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует состояние в словарь."""
        return {
            "mcp_available": self.mcp_available,
            "rag_available": self.rag_available,
            "current_conversation_id": self.current_conversation_id,
            "model": self.model,
            "mcp_calls_count": len(self.mcp_calls),
            "rag_calls_count": len(self.rag_calls)
        }


class LogsitAgent:
    """Основной агент логистического ассистента."""
    
    def __init__(self, model: Optional[str] = None, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        
        # Устанавливаем модель
        self.model = model or self.config.default_model
        
        # Инициализируем состояние
        self.state = AgentState(model=self.model)
        
        # Инициализируем клиент LLM
        self.client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Инициализируем менеджеры
        self.conversation_manager = get_conversation_manager()
        self.summarizer = get_summarizer(self.conversation_manager)
        self.task_state_manager = get_task_state_manager(self.conversation_manager)
        
        # Инициализируем сервисы
        self.mcp_orchestrator = MCPOrchestrator()
        
        # Инициализируем RAG сервис если доступен
        if RAG_AVAILABLE:
            self.rag_service = get_rag_service()
        else:
            self.rag_service = None
        
        # Проверяем доступность сервисов
        self._check_services_availability()
    
    def _check_services_availability(self):
        """Проверяет доступность сервисов."""
        if self.rag_service is not None:
            self.state.rag_available = self.rag_service.is_available()
        else:
            self.state.rag_available = False
        
        if self.state.rag_available:
            print("✅ RAG сервис доступен")
        elif self.rag_service is None:
            print("⚠️ RAG сервис не инициализирован (зависимости не установлены)")
        else:
            print("⚠️ RAG сервис недоступен")
    
    async def connect_mcp(self) -> bool:
        """
        Подключает MCP серверы.
        
        Returns:
            True если подключение успешно, иначе False
        """
        try:
            success = await self.mcp_orchestrator.connect_all_servers()
            self.state.mcp_available = success
            
            if success:
                print("✅ MCP серверы подключены")
            else:
                print("⚠️ Не удалось подключить MCP серверы")
            
            return success
            
        except Exception as e:
            print(f"❌ Ошибка подключения MCP: {e}")
            self.state.mcp_available = False
            return False
    
    async def disconnect_mcp(self):
        """Отключает MCP серверы."""
        try:
            await self.mcp_orchestrator.disconnect_all_servers()
            self.state.mcp_available = False
            print("✅ MCP серверы отключены")
        except Exception as e:
            print(f"❌ Ошибка отключения MCP: {e}")
    
    def _select_tool(self, message: str) -> ToolSelection:
        """
        Выбирает инструмент для обработки запроса.
        
        Args:
            message: Сообщение пользователя
            
        Returns:
            Выбранный инструмент
        """
        message_lower = message.lower()
        
        # Ключевые слова для MCP
        mcp_keywords = [
            "маршрут", "расстояние", "координат", "город", "расчет", "стоимость",
            "доставк", "оптимальн", "путь", "проезд", "километр", "миля"
        ]
        
        # Ключевые слова для RAG
        rag_keywords = [
            "тариф", "правило", "обязанност", "документ", "постановление",
            "услов", "требован", "закон", "регламент", "инструкция", "api пэк"
        ]
        
        # Проверяем MCP ключевые слова
        mcp_score = sum(1 for keyword in mcp_keywords if keyword in message_lower)
        
        # Проверяем RAG ключевые слова
        rag_score = sum(1 for keyword in rag_keywords if keyword in message_lower)
        
        # Определяем выбор инструмента
        if mcp_score > 0 and rag_score > 0:
            return ToolSelection.HYBRID
        elif mcp_score > rag_score:
            return ToolSelection.MCP
        elif rag_score > 0:
            return ToolSelection.RAG
        else:
            return ToolSelection.KNOWLEDGE
    
    def _build_conversation_context(self, conversation_id: int) -> str:
        """
        Строит контекст диалога для промпта.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Контекст диалога
        """
        context_parts = []
        
        # Информация о диалоге
        conversation = self.conversation_manager.get_conversation(conversation_id)
        if conversation:
            context_parts.append(f"📝 Диалог: {conversation.title}")
            context_parts.append(f"📊 Сообщений: {conversation.message_count}")
        
        # Состояние задачи
        task_context = self.task_state_manager.format_task_state_for_prompt(conversation_id)
        if task_context:
            context_parts.append("")
            context_parts.append(task_context)
        
        # Последняя суммаризация
        summary = self.summarizer.get_summary_for_conversation(conversation_id)
        if summary:
            context_parts.append("")
            context_parts.append(f"📋 Последняя сводка: {summary}")
        
        return "\n".join(context_parts)
    
    def _build_system_prompt(self, conversation_id: int) -> str:
        """
        Строит системный промпт.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Системный промпт
        """
        conversation_context = self._build_conversation_context(conversation_id)
        
        prompt = self.config.system_prompt_template.format(
            conversation_context=conversation_context
        )
        
        # Добавляем информацию о доступных инструментах
        if self.state.mcp_available:
            available_tools = self.mcp_orchestrator.get_available_tools()
            tool_info = []
            
            for server, tools in available_tools.items():
                tool_info.append(f"{server}: {', '.join(tools)}")
            
            prompt += f"\n\nДоступные MCP инструменты:\n" + "\n".join(tool_info)
        
        return prompt
    
    async def _process_with_mcp(self, message: str) -> Tuple[bool, str]:
        """
        Обрабатывает запрос с помощью MCP инструментов.
        
        Args:
            message: Сообщение пользователя
            
        Returns:
            Кортеж (успех, результат)
        """
        if not self.state.mcp_available:
            return False, "MCP серверы недоступны"
        
        try:
            # TODO: Реализовать интеллектуальный вызов MCP инструментов
            # Пока используем простую логику
            
            # Проверяем наличие городов в запросе
            city_keywords = ["москв", "санкт-петербург", "казан", "нижний новгород", "екатеринбург"]
            has_cities = any(city in message.lower() for city in city_keywords)
            
            if has_cities:
                # Пробуем вызвать инструмент geocode_batch
                success, result = await self.mcp_orchestrator.call_tool(
                    server_name="route_planner",
                    tool_name="geocode_batch",
                    arguments={"cities": ["Москва", "Санкт-Петербург"]}  # Пример
                )
                
                if success:
                    return True, f"🔧 MCP: {result}"
                else:
                    return False, f"Ошибка MCP: {result}"
            else:
                return False, "В запросе не указаны города для обработки MCP"
                
        except Exception as e:
            return False, f"Ошибка обработки MCP: {e}"
    
    async def _process_with_rag(self, message: str) -> Tuple[bool, List[RAGResult]]:
        """
        Обрабатывает запрос с помощью RAG поиска.
        
        Args:
            message: Сообщение пользователя
            
        Returns:
            Кортеж (успех, результаты поиска)
        """
        if not self.state.rag_available or self.rag_service is None:
            return False, []
        
        try:
            results = self.rag_service.search(message, top_k=3)
            
            if results:
                # Сохраняем запись о вызове RAG
                rag_call = {
                    "query": message,
                    "timestamp": "now",
                    "results_count": len(results),
                    "top_score": results[0].similarity_score if results else 0.0
                }
                self.state.rag_calls.append(rag_call)
                
                return True, results
            else:
                return False, []
                
        except Exception as e:
            print(f"❌ Ошибка RAG поиска: {e}")
            return False, []
    
    async def _generate_response(self, message: str, conversation_id: int,
                               tool_selection: ToolSelection,
                               mcp_result: Optional[str] = None,
                               rag_results: Optional[List[RAGResult]] = None) -> str:
        """
        Генерирует ответ с помощью LLM.
        
        Args:
            message: Сообщение пользователя
            conversation_id: ID диалога
            tool_selection: Выбранный инструмент
            mcp_result: Результат MCP (опционально)
            rag_results: Результаты RAG (опционально)
            
        Returns:
            Сгенерированный ответ
        """
        try:
            # Строим системный промпт
            system_prompt = self._build_system_prompt(conversation_id)
            
            # Собираем контекст для пользовательского промпта
            user_context = []
            
            if tool_selection == ToolSelection.MCP and mcp_result:
                user_context.append(f"Результат MCP: {mcp_result}")
            
            if tool_selection in [ToolSelection.RAG, ToolSelection.HYBRID] and rag_results:
                rag_context = self.rag_service.format_results_for_display(rag_results)
                user_context.append(rag_context)
            
            # Формируем финальный промпт
            final_prompt = message
            if user_context:
                final_prompt = f"{message}\n\nКонтекст:\n" + "\n\n".join(user_context)
            
            # Генерируем ответ
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Добавляем префикс инструмента
            if tool_selection == ToolSelection.MCP:
                answer = f"🔧 {answer}"
            elif tool_selection == ToolSelection.RAG:
                answer = f"📚 {answer}"
            elif tool_selection == ToolSelection.HYBRID:
                answer = f"🔧📚 {answer}"
            else:
                answer = f"💡 {answer}"
            
            return answer
            
        except Exception as e:
            print(f"❌ Ошибка генерации ответа: {e}")
            
            # Возвращаем запасной ответ
            fallback_responses = {
                ToolSelection.MCP: "🔧 К сожалению, не удалось выполнить расчеты через MCP. Попробуйте позже.",
                ToolSelection.RAG: "📚 Не удалось найти информацию в документах.",
                ToolSelection.HYBRID: "🔧📚 Возникла ошибка при обработке запроса.",
                ToolSelection.KNOWLEDGE: "💡 Извините, произошла ошибка при генерации ответа."
            }
            
            return fallback_responses.get(tool_selection, "❌ Произошла ошибка.")
    
    async def process_message(self, message: str) -> str:
        """
        Обрабатывает сообщение пользователя.
        
        Args:
            message: Сообщение пользователя
            
        Returns:
            Ответ агента
        """
        try:
            # Проверяем текущий диалог
            if not self.state.current_conversation_id:
                # Создаем новый диалог
                conversation = self.conversation_manager.create_conversation()
                self.state.current_conversation_id = conversation.id
                self.conversation_manager.active_conversation_id = conversation.id
            
            conversation_id = self.state.current_conversation_id
            
            # Добавляем сообщение пользователя в историю
            self.conversation_manager.add_message(
                conversation_id=conversation_id,
                role="user",
                content=message
            )
            
            # Обновляем состояние задачи на основе сообщения
            self.task_state_manager.update_from_message(conversation_id, message)
            
            # Проверяем и выполняем суммаризацию при необходимости
            if self.config.auto_summarize:
                summarized, summary = await self.summarizer.check_and_summarize(conversation_id)
                if summarized:
                    print(f"📋 Выполнена автосуммаризация диалога {conversation_id}")
            
            # Выбираем инструмент для обработки
            tool_selection = self._select_tool(message)
            print(f"🛠️ Выбран инструмент: {tool_selection.value}")
            
            # Обрабатываем в зависимости от выбранного инструмента
            mcp_result = None
            rag_results = None
            
            if tool_selection in [ToolSelection.MCP, ToolSelection.HYBRID]:
                mcp_success, mcp_result = await self._process_with_mcp(message)
                if not mcp_success:
                    print(f"⚠️ MCP обработка не удалась: {mcp_result}")
            
            if tool_selection in [ToolSelection.RAG, ToolSelection.HYBRID]:
                rag_success, rag_results = await self._process_with_rag(message)
                if not rag_success:
                    print("⚠️ RAG поиск не дал результатов")
            
            # Генерируем ответ
            response = await self._generate_response(
                message=message,
                conversation_id=conversation_id,
                tool_selection=tool_selection,
                mcp_result=mcp_result,
                rag_results=rag_results
            )
            
            # Добавляем ответ ассистента в историю
            self.conversation_manager.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=response
            )
            
            # Обновляем состояние задачи на основе ответа
            self.task_state_manager.update_from_message(
                conversation_id,
                f"Ответ ассистента: {response}",
                role="assistant"
            )
            
            return response
            
        except Exception as e:
            print(f"❌ Критическая ошибка обработки сообщения: {e}")
            return "❌ Произошла критическая ошибка при обработке вашего запроса. Попробуйте еще раз."
    
    def switch_conversation(self, conversation_id: int) -> bool:
        """
        Переключает текущий диалог.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            True если переключение успешно, иначе False
        """
        conversation = self.conversation_manager.get_conversation(conversation_id)
        if not conversation:
            return False
        
        self.state.current_conversation_id = conversation_id
        self.conversation_manager.active_conversation_id = conversation_id
        
        print(f"🔄 Переключен на диалог: {conversation.title} (ID: {conversation_id})")
        return True
    
    def create_new_conversation(self, title: Optional[str] = None) -> bool:
        """
        Создает новый диалог и переключается на него.
        
        Args:
            title: Заголовок диалога (опционально)
            
        Returns:
            True если создание успешно, иначе False
        """
        conversation = self.conversation_manager.create_conversation(title)
        if not conversation:
            return False
        
        return self.switch_conversation(conversation.id)
    
    def get_current_conversation(self):
        """
        Получает текущий диалог.
        
        Returns:
            Текущий диалог или None
        """
        if not self.state.current_conversation_id:
            return None
        
        return self.conversation_manager.get_conversation(self.state.current_conversation_id)
    
    def get_all_conversations(self):
        """
        Получает все диалоги.
        
        Returns:
            Список всех диалогов
        """
        return self.conversation_manager.get_all_conversations()
    
    def get_mcp_calls(self) -> List[MCPToolCall]:
        """
        Получает историю вызовов MCP.
        
        Returns:
            Список вызовов MCP
        """
        return self.mcp_orchestrator.get_tool_calls()
    
    def clear_history(self):
        """Очищает историю текущего диалога."""
        if self.state.current_conversation_id:
            self.conversation_manager.clear_conversation(self.state.current_conversation_id)
            print(f"🗑️ Очищена история диалога {self.state.current_conversation_id}")
    
    def update_config(self, **kwargs):
        """
        Обновляет конфигурацию агента.
        
        Args:
            **kwargs: Параметры конфигурации
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Возвращает текущее состояние агента.
        
        Returns:
            Словарь с состоянием
        """
        return self.state.to_dict()