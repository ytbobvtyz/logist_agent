"""
Расширенный LLM агент с поддержкой:
- Множества диалогов с переключением
- Автоматической суммаризации каждые 10 сообщений
- Task State (памяти задачи)
- Локальной модели Ollama
"""

import json
import os
import sys
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Загружаем переменные окружения
load_dotenv()

# Импорт новых модулей
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conversation_manager import ConversationManager, Message, get_conversation_manager
from summarizer import Summarizer, get_summarizer
from task_state import TaskStateManager, get_task_state_manager
from llm_client import create_llm_client, LLMClient


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Базовый системный промпт для OpenRouter (с MCP)
BASE_SYSTEM_PROMPT_MCP = """Ты умный логист-ассистент. У тебя есть инструменты:

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
Если город не найден, сообщи об этом пользователю."""

# Базовый системный промпт для локальной модели (без MCP)
BASE_SYSTEM_PROMPT_LOCAL = """Ты умный логист-ассистент.

## ИНСТРУМЕНТЫ:
- RAG (поиск в документах) - можешь искать информацию в документах (тарифы, правила, постановления)
- Собственные знания - для ответов на общие вопросы

## ПРАВИЛА РАБОТЫ:
1. Если вопрос про **правила, обязанности, тарифы из документов, API ПЭК** → система сама найдет информацию в документах
2. Если вопрос про **общие понятия логистики** → отвечай из своих знаний
3. Если вопрос про **расстояния, маршруты, расчеты** → объясни, что это требует инструментов MCP, которые недоступны в локальном режиме

## ФОРМАТ ОТВЕТА:
- Если система нашла информацию в документах: укажи 📚 RAG + источник
- Если отвечаешь из знаний: укажи 💡

## ОГРАНИЧЕНИЯ:
- MCP инструменты недоступны в локальном режиме
- Могу ответить только на вопросы, не требующие точных расчетов

Будь полезным и честным. Если не знаешь ответа, скажи об этом."""


@dataclass
class MCPToolCall:
    """Запись о вызове MCP инструмента."""
    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AgentState:
    """Состояние агента."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    mcp_calls: List[MCPToolCall] = field(default_factory=list)
    mcp_available: bool = False
    current_conversation_id: Optional[int] = None


class MCPOrchestrator:
    """Оркестратор для управления несколькими MCP серверами."""
    
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}  # server_name -> {tool_name -> tool_info}
        self._server_contexts: Dict[str, Any] = {}  # server_name -> context manager
        self._server_processes: Dict[str, Any] = {}  # server_name -> (read_stream, write_stream)
    
    async def connect_server(self, server_name: str, server_script: str) -> bool:
        """Подключает MCP сервер."""
        try:
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[server_script]
            )
            
            server_cm = stdio_client(server_params)
            read_stream, write_stream = await server_cm.__aenter__()
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()
            
            self._server_contexts[server_name] = server_cm
            self._server_processes[server_name] = (read_stream, write_stream)
            self.sessions[server_name] = session
            
            tools_result = await session.list_tools()
            server_tools = {}
            for tool in tools_result.tools:
                parameters = tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}}
                prefixed_name = f"{server_name}__{tool.name}"
                server_tools[prefixed_name] = {
                    "type": "function",
                    "function": {
                        "name": prefixed_name,
                        "description": tool.description or "",
                        "parameters": parameters
                    },
                    "original_name": tool.name,
                    "server": server_name
                }
            
            self.tools[server_name] = server_tools
            print(f"MCP сервер '{server_name}' подключен. Инструменты: {list(server_tools.keys())}")
            return True
        
        except Exception as e:
            print(f"Ошибка подключения к MCP серверу '{server_name}': {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def disconnect_all(self):
        """Отключает все серверы."""
        for server_name, session in list(self.sessions.items()):
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
        
        self.sessions.clear()
        self.tools.clear()
        self._server_contexts.clear()
        self._server_processes.clear()
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Возвращает список всех доступных инструментов от всех серверов."""
        all_tools = []
        for server_tools in self.tools.values():
            for tool_info in server_tools.values():
                function_info = tool_info["function"]
                all_tools.append({
                    "type": "function",
                    "function": {
                        "name": function_info["name"],
                        "description": function_info["description"],
                        "parameters": function_info["parameters"]
                    }
                })
        return all_tools
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Вызывает инструмент на соответствующем сервере."""
        server_name = None
        original_tool_name = None
        
        for s_name, s_tools in self.tools.items():
            if tool_name in s_tools:
                server_name = s_name
                original_tool_name = s_tools[tool_name]["original_name"]
                break
        
        if server_name is None:
            raise Exception(f"Неизвестный инструмент: {tool_name}")
        
        session = self.sessions.get(server_name)
        if not session:
            raise Exception(f"Сессия для сервера '{server_name}' не найдена")
        
        result = await session.call_tool(str(original_tool_name), arguments)
        
        result_text = ""
        if hasattr(result, 'content'):
            for content in result.content:
                if hasattr(content, 'text'):
                    result_text += str(content.text)
                elif isinstance(content, str):
                    result_text += content
                else:
                    result_text += str(content)
        elif isinstance(result, str):
            result_text = result
        else:
            result_text = str(result)
        
        return result_text
    
    def is_available(self) -> bool:
        """Проверяет, подключен ли хотя бы один сервер."""
        return len(self.sessions) > 0


class EnhancedRoutePlannerAgent:
    """Расширенный агент с поддержкой диалогов, суммаризации и task state."""
    
    def __init__(self, model: str = "openrouter/auto", use_local: bool = False, db_path: str = "conversations.db"):
        """
        Инициализация расширенного агента.
        
        Args:
            model: Название модели
            use_local: Использовать локальную модель Ollama
            db_path: Путь к базе данных диалогов
        """
        self.model = model
        self.use_local = use_local
        
        # Создаем клиент LLM через абстрактный слой
        self.llm_client = create_llm_client(model, use_local)
        
        # Для обратной совместимости оставляем self.client
        if use_local:
            # Для локальной модели используем Ollama-совместимый клиент
            self.client = AsyncOpenAI(
                base_url=os.getenv("OLLAMA_URL", "http://localhost:11434") + "/v1",
                api_key="ollama",
                timeout=120.0
            )
        else:
            self.client = AsyncOpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                timeout=120.0,
                max_retries=2
            )
        
        self.orchestrator = MCPOrchestrator()
        self.state = AgentState()
        self.rag_retriever = None
        
        # Инициализация новых компонентов
        self.conversation_manager = get_conversation_manager(db_path)
        self.summarizer = get_summarizer(self.conversation_manager)
        self.task_state_manager = get_task_state_manager(self.conversation_manager)
        
        # Устанавливаем активный диалог (создаст новый если нет)
        self.current_conversation = self.conversation_manager.get_active_conversation()
        self.state.current_conversation_id = self.current_conversation.id
        
        # Инициализация RAG retriever
        self._init_rag_retriever()
        
        model_type = "локальная (Ollama)" if use_local else "OpenRouter"
        print(f"✅ Инициализирован расширенный агент. Модель: {model_type} ({self.model}). Текущий диалог: #{self.current_conversation.id}")
    
    def _get_system_prompt(self) -> str:
        """Возвращает системный промпт в зависимости от типа модели."""
        if self.use_local:
            return BASE_SYSTEM_PROMPT_LOCAL
        else:
            return BASE_SYSTEM_PROMPT_MCP

    def _init_rag_retriever(self):
        """Инициализирует RAG retriever."""
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, project_root)
            
            from rag_retriever import RAGRetriever
            
            db_path = os.path.join(project_root, "metadata.db")
            self.rag_retriever = RAGRetriever(db_path=db_path)
            print("✅ RAG Retriever загружен")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить RAG Retriever: {e}")
            self.rag_retriever = None
    
    async def connect_mcp(self) -> bool:
        """Подключается к MCP серверам."""
        current_dir = os.path.dirname(__file__)
        
        yandex_script = os.path.join(current_dir, "mcp_server.py")
        yandex_success = await self.orchestrator.connect_server("yandex", yandex_script)
        
        pecom_script = os.path.join(current_dir, "pecom_server.py")
        pecom_success = await self.orchestrator.connect_server("pecom", pecom_script)
        
        self.state.mcp_available = self.orchestrator.is_available()
        
        if self.state.mcp_available:
            all_tools = self.orchestrator.get_all_tools()
            print(f"Всего доступно инструментов: {len(all_tools)}")
            print(f"Инструменты: {[t['function']['name'] for t in all_tools]}")
        
        return self.state.mcp_available
    
    async def disconnect_mcp(self):
        """Отключается от всех MCP серверов."""
        self.state.mcp_available = False
        try:
            await self.orchestrator.disconnect_all()
        except Exception as e:
            print(f"⚠️ Ошибка при отключении MCP: {e}")
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Вызывает MCP инструмент через оркестратор."""
        call_record = MCPToolCall(
            tool_name=tool_name,
            arguments=arguments,
            success=False
        )
        
        try:
            result_text = await self.orchestrator.call_tool(tool_name, arguments)
            
            call_record.success = True
            call_record.result = result_text
            self.state.mcp_calls.append(call_record)
            return result_text
        
        except Exception as e:
            call_record.error = str(e)
            self.state.mcp_calls.append(call_record)
            raise
    
    def get_current_conversation(self):
        """Получает текущий активный диалог."""
        return self.current_conversation
    
    def switch_conversation(self, conversation_id: int) -> bool:
        """
        Переключается на другой диалог.
        
        Args:
            conversation_id: ID диалога для переключения
            
        Returns:
            True если успешно
        """
        success = self.conversation_manager.set_active_conversation(conversation_id)
        if success:
            self.current_conversation = self.conversation_manager.get_conversation(conversation_id)
            self.state.current_conversation_id = conversation_id
            
            # Очищаем кэшированное состояние агента для нового диалога
            self.state.messages.clear()
            self.state.mcp_calls.clear()
            
            print(f"✅ Переключен на диалог #{conversation_id}")
            return True
        
        return False
    
    def create_new_conversation(self, title: str = None) -> bool:
        """
        Создает новый диалог и переключается на него.
        
        Args:
            title: Заголовок диалога
            
        Returns:
            True если успешно
        """
        conversation = self.conversation_manager.create_conversation(title)
        if conversation:
            self.current_conversation = conversation
            self.state.current_conversation_id = conversation.id
            
            # Очищаем кэшированное состояние агента
            self.state.messages.clear()
            self.state.mcp_calls.clear()
            
            return True
        
        return False
    
    async def process_message(self, user_message: str, callback=None) -> str:
        """
        Обрабатывает сообщение пользователя с учетом контекста диалога.
        
        Args:
            user_message: Сообщение пользователя
            callback: Функция обратного вызова для стриминга
            
        Returns:
            Ответ агента
        """
        if not self.current_conversation:
            # Создаем новый диалог если нет активного
            self.create_new_conversation()
        
        conversation_id = self.current_conversation.id
        
        # Добавляем сообщение пользователя в БД
        user_msg_obj = self.conversation_manager.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_message
        )
        
        # Обновляем состояние задачи
        await self.task_state_manager.update_from_new_message(
            conversation_id,
            Message(role="user", content=user_message)
        )
        
        # Проверяем, нужно ли суммировать диалог
        summary = await self.summarizer.check_and_summarize(conversation_id)
        if summary and callback:
            await callback(f"📋 Обновлена краткая сводка диалога")
        
        # Определяем стратегию обработки
        use_rag = self._should_use_rag(user_message)
        
        if callback:
            await callback(f"📊 Анализ запроса... {'📚 RAG' if use_rag else '🔧 MCP'} режим")
        
        # Получаем контекст для промпта
        context = self._build_context_for_prompt(conversation_id)
        
        if use_rag and self.rag_retriever:
            # Режим RAG
            try:
                if callback:
                    await callback("🔍 Ищу информацию в документах...")
                
                chunks = self.search_with_rag(user_message, top_k=3)
                
                if chunks:
                    rag_prompt = self._build_rag_prompt(user_message, chunks, context)
                    
                    if callback:
                        await callback(f"📄 Найдено {len(chunks)} релевантных фрагментов")
                    
                    response_text = await self._call_llm_with_context(rag_prompt, context)
                    
                    answer = response_text or "Не удалось получить ответ"
                    sources = ", ".join(set(chunk['filename'] for chunk in chunks))
                    final_response = f"📚 RAG (источники: {sources})\n\n{answer}"
                else:
                    # Если ничего не найдено
                    prompt = self._build_basic_prompt(user_message, context)
                    response_text = await self._call_llm_with_context(prompt, context)
                    
                    answer = response_text or "Не удалось получить ответ"
                    final_response = f"💡 (RAG ничего не нашёл, использованы знания)\n\n{answer}"
                
                # Сохраняем ответ ассистента
                self.conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=final_response
                )
                
                return final_response
                
            except Exception as e:
                error_msg = f"❌ Ошибка RAG: {e}"
                self.conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=error_msg
                )
                return error_msg
        else:
            # Режим MCP
            return await self._process_with_mcp(user_message, context, conversation_id, callback)
    
    def _build_context_for_prompt(self, conversation_id: int) -> str:
        """Строит контекст для промпта на основе диалога."""
        context_parts = []
        
        # Контекст из task state
        task_context = self.task_state_manager.get_context_for_llm(conversation_id)
        if task_context:
            context_parts.append(task_context)
        
        # Контекст из суммаризации
        summary_context = self.summarizer.get_summary_context(conversation_id)
        if summary_context:
            context_parts.append(summary_context)
        
        # Получаем последние сообщения для контекста (если нет суммаризации)
        if not summary_context:
            last_messages = self.conversation_manager.get_last_messages(conversation_id, count=5)
            if last_messages and len(last_messages) > 1:
                # Берем все кроме последнего сообщения пользователя
                context_messages = last_messages[:-1]
                if context_messages:
                    dialogue_context = "Последние сообщения в диалоге:\n"
                    for msg in context_messages:
                        role = "Пользователь" if msg.role == "user" else "Ассистент"
                        dialogue_context += f"{role}: {msg.content}\n\n"
                    context_parts.append(dialogue_context.strip())
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def _build_basic_prompt(self, user_message: str, context: str = "") -> str:
        """Строит базовый промпт."""
        if context:
            return f"""{context}

{user_message}"""
        return user_message
    
    def _build_rag_prompt(self, query: str, chunks: List[Dict], context: str = "") -> str:
        """Строит промпт с RAG контекстом."""
        rag_context = "\n".join([
            f"📄 [{chunk['filename']}] {chunk['text']}"
            for chunk in chunks
        ])
        
        prompt_parts = []
        
        if context:
            prompt_parts.append(context)
        
        prompt_parts.append(f"""Пользователь задал вопрос: {query}

## Релевантная информация из документов:
{rag_context}

Ответь на вопрос пользователя, используя информацию из документов если она есть.
Если информации в документах недостаточно, дополни ответ своими знаниями.
Всегда указывай источники информации.""")
        
        return "\n\n".join(prompt_parts)
    
    async def _call_llm_with_context(self, user_prompt: str, context: str = "") -> str:
        """
        Вызывает LLM с учетом контекста.
        
        Args:
            user_prompt: Промпт пользователя
            context: Контекст диалога
            
        Returns:
            Ответ LLM (текст)
        """
        # Формируем полный системный промпт
        if context:
            system_prompt = f"""{self._get_system_prompt()}

## КОНТЕКСТ ТЕКУЩЕГО ДИАЛОГА:
{context}

Учитывай этот контекст при ответе."""
        else:
            system_prompt = self._get_system_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.llm_client.chat_completion(messages, max_tokens=1024)
        return response.get("content", "")

    async def _process_without_tools(self, user_message: str, context: str,
                                   conversation_id: int, callback=None) -> str:
        """Обрабатывает сообщение без использования MCP инструментов."""
        if self.use_local and callback:
            await callback("💡 Использую локальную модель без инструментов")
        
        return await self._call_llm_with_context(user_message, context)
    
    async def _process_with_mcp(self, user_message: str, context: str, 
                               conversation_id: int, callback=None) -> str:
        """Обрабатывает сообщение с использованием MCP инструментов."""
        # Проверяем, доступны ли MCP инструменты для локальной модели
        if self.use_local:
            if callback:
                await callback("⚠️ Локальная модель не поддерживает инструменты MCP. Использую режим RAG/знаний.")
            return await self._process_without_tools(user_message, context, conversation_id, callback)
        
        # Формируем полный промпт
        if context:
            system_prompt = f"""{self._get_system_prompt()}

## КОНТЕКСТ ТЕКУЩЕГО ДИАЛОГА:
{context}

Учитывай этот контекст при выборе инструментов и формировании ответа."""
        else:
            system_prompt = self._get_system_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        mcp_tools = self.orchestrator.get_all_tools() if self.state.mcp_available else []
        
        try:
            response = await self.llm_client.chat_completion(
                messages=messages,
                tools=mcp_tools if mcp_tools else None,
                max_tokens=1024
            )
            
            assistant_content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])
            
            max_iterations = 10
            iteration_count = 0
            
            while tool_calls and iteration_count < max_iterations:
                iteration_count += 1
                
                # Добавляем сообщение ассистента с вызовами инструментов
                messages.append({
                    "role": "assistant",
                    "content": assistant_content or "",
                    "tool_calls": tool_calls
                })
                
                for tool_call in tool_calls:
                    tool_name = None
                    arguments = {}
                    tool_call_id = tool_call.get("id", str(hash(str(tool_call))))
                    
                    try:
                        tool_func = tool_call.get("function", {})
                        tool_name = tool_func.get("name", "")
                        arguments_str = tool_func.get("arguments", "{}")
                        arguments = json.loads(arguments_str)
                    except Exception as e:
                        print(f"⚠️ Ошибка извлечения данных из tool_call: {e}")
                        tool_name = "unknown"
                        arguments = {}
                    
                    if not tool_name:
                        continue
                    
                    if callback:
                        await callback(f"🔧 Вызываю {tool_name}...")
                    
                    try:
                        result = await self.call_mcp_tool(tool_name, arguments)
                        tool_result = result
                    except Exception as e:
                        tool_result = json.dumps({"error": str(e)})
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result
                    })
                
                # Получаем новый ответ от LLM
                next_response = await self.llm_client.chat_completion(
                    messages=messages,
                    tools=mcp_tools if mcp_tools else None,
                    max_tokens=1024
                )
                assistant_content = next_response.get("content", "")
                tool_calls = next_response.get("tool_calls", [])
            
            if iteration_count >= max_iterations and tool_calls:
                final_response = "❌ Превышено максимальное количество итераций. Возможно, проблема с интерпретацией запроса."
                self.conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=final_response
                )
                return final_response
            
            final_response = assistant_content or "Не удалось получить ответ"
            
            # Добавляем информацию о статусе MCP
            prefix = "🔧 MCP"
            if not self.state.mcp_available:
                prefix = "⚠️ MCP недоступен"
                final_response += "\n\n⚠️ MCP инструменты недоступны. Ответ дан без использования инструментов."
            elif self.state.mcp_calls:
                success_count = sum(1 for c in self.state.mcp_calls if c.success)
                final_response += f"\n\n✅ Использовано MCP инструментов: {success_count}/{len(self.state.mcp_calls)}"
            
            final_response = f"{prefix}\n\n{final_response}"
            
            # Сохраняем ответ ассистента
            self.conversation_manager.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=final_response
            )
            
            return final_response
        
        except Exception as e:
            error_msg = f"❌ Ошибка MCP обработки: {e}"
            calls_info = f"\n📊 Вызовы инструментов: {len(self.state.mcp_calls)}"
            for i, call in enumerate(self.state.mcp_calls):
                status = "✅" if call.success else "❌"
                calls_info += f"\n  {i+1}. {tool_name}: {status}"
            
            error_msg += calls_info
            self.conversation_manager.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=error_msg
            )
            return error_msg
    
    def search_with_rag(self, query: str, top_k: int = 3) -> List[Dict]:
        """Ищет информацию в документах через RAG."""
        if not self.rag_retriever:
            return []
        
        try:
            return self.rag_retriever.search(query, top_k)
        except Exception as e:
            print(f"⚠️ Ошибка RAG поиска: {e}")
            return []
    
    def _should_use_rag(self, user_message: str) -> bool:
        """Определяет, нужно ли использовать RAG для данного запроса."""
        rag_keywords = [
            'тариф', 'правила', 'обязанности', 'постановление', 'закон',
            'стоимость', 'цена', 'расценки', 'условия', 'требования',
            'api', 'документ', 'инструкция', 'руководство', 'справочник',
            'фрахтователь', 'фрахтовщик', 'перевозчик', 'экспедитор'
        ]
        
        mcp_keywords = [
            'расстояние', 'маршрут', 'координаты', 'город', 'городa',
            'рассчитать', 'стоимость доставки', 'цена перевозки',
            'оптимальный путь', 'кратчайший маршрут', 'геокодирование'
        ]
        
        message_lower = user_message.lower()
        
        has_rag_keywords = any(keyword in message_lower for keyword in rag_keywords)
        has_mcp_keywords = any(keyword in message_lower for keyword in mcp_keywords)
        
        if has_rag_keywords and not has_mcp_keywords:
            return True
        
        if not has_mcp_keywords and len(user_message.split()) > 3:
            return True
            
        return False
    
    def get_mcp_calls(self) -> List[MCPToolCall]:
        """Возвращает список вызовов MCP инструментов."""
        return self.state.mcp_calls
    
    def clear_history(self):
        """Очищает историю сообщений в текущем диалоге."""
        self.state.messages.clear()
        self.state.mcp_calls.clear()
        
        if self.current_conversation:
            # Сбрасываем состояние задачи для текущего диалога
            self.task_state_manager.reset_task_state(self.current_conversation.id)
            print(f"🔄 Очищена история диалога #{self.current_conversation.id}")
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Получает список всех диалогов."""
        conversations = self.conversation_manager.get_all_conversations()
        return [
            {
                "id": conv.id,
                "title": conv.title,
                "message_count": conv.message_count,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "active": conv.active
            }
            for conv in conversations
        ]
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Получает статистику по диалогам."""
        return self.conversation_manager.get_statistics()


# Список моделей OpenRouter
OPENROUTER_MODELS = [
    # Бесплатные модели
    {"name": "Qwen 3.6 Plus Preview (бесплатно)", "id": "qwen/qwen3.6-plus-preview:free"},
    {"name": "Step-3.5 Flash (бесплатно)", "id": "stepfun/step-3.5-flash:free"},
    {"name": "OpenRouter Free (бесплатно)", "id": "openrouter/free"},
    # Платные модели
    {"name": "Xiaomi MiMo v2 Pro", "id": "xiaomi/mimo-v2-pro"},
    {"name": "MiniMax M2.5", "id": "minimax/minimax-m2.5"},
    {"name": "DeepSeek v3.2", "id": "deepseek/deepseek-v3.2"},
]