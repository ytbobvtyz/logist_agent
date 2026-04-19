"""
LLM агент для планирования маршрутов.
Использует OpenAI API через OpenRouter, MCP инструменты и RAG для поиска в документах.
"""

import json
import os
import sys
import asyncio
import anyio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Загружаем переменные окружения
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SYSTEM_PROMPT = """Ты умный логист-ассистент. У тебя есть инструменты:

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
Важно: при использовании RAG - всегда выводи в конце цитаты из источников.
Если пользователь не указал города или указал менее двух, попроси уточнить.
Если город не найден, сообщи об этом пользователю."""


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


class MCPOrchestrator:
    """Оркестратор для управления несколькими MCP серверами."""
    
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}  # server_name -> {tool_name -> tool_info}
        self._server_contexts: Dict[str, Any] = {}  # server_name -> context manager
        self._server_processes: Dict[str, Any] = {}  # server_name -> (read_stream, write_stream)
    
    async def connect_server(self, server_name: str, server_script: str) -> bool:
        """
        Подключает MCP сервер.
        
        Args:
            server_name: Уникальное имя сервера
            server_script: Путь к скрипту сервера
        
        Returns:
            True если подключение успешно
        """
        try:
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[server_script]
            )
            
            # Создаём контекстный менеджер
            server_cm = stdio_client(server_params)
            
            # Входим в контекст
            read_stream, write_stream = await server_cm.__aenter__()
            
            # Создаём сессию
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()
            
            # Сохраняем всё
            self._server_contexts[server_name] = server_cm
            self._server_processes[server_name] = (read_stream, write_stream)
            self.sessions[server_name] = session
            
            # Получаем список инструментов
            tools_result = await session.list_tools()
            server_tools = {}
            for tool in tools_result.tools:
                parameters = tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}}
                # Добавляем префикс имени сервера к имени инструмента
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
                # Возвращаем в формате OpenAI
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
        """
        Вызывает инструмент на соответствующем сервере.
        
        Args:
            tool_name: Имя инструмента (с префиксом сервера, например "pecom__calculate_cost")
            arguments: Аргументы инструмента
        
        Returns:
            Результат вызова
        """
        # Находим сервер по имени инструмента
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
        
        # Вызываем инструмент с оригинальным именем
        result = await session.call_tool(str(original_tool_name), arguments)
        
        # Извлекаем текст из результата
        result_text = ""
        if hasattr(result, 'content'):
            for content in result.content:
                if hasattr(content, 'text'):
                    result_text += str(content.text)
                elif isinstance(content, str):
                    result_text += content
                else:
                    # Пробуем преобразовать в строку
                    result_text += str(content)
        elif isinstance(result, str):
            result_text = result
        else:
            result_text = str(result)
        
        return result_text
    
    def is_available(self) -> bool:
        """Проверяет, подключен ли хотя бы один сервер."""
        return len(self.sessions) > 0


class RoutePlannerAgent:
    """Агент для планирования маршрутов с MCP инструментами и RAG поиском."""
    
    def __init__(self, model: str = "openrouter/auto"):
        """
        Инициализация агента.
        
        Args:
            model: Название модели OpenRouter
        """
        self.model = model
        self.client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0,
            max_retries=2
        )
        self.orchestrator = MCPOrchestrator()
        self.state = AgentState()
        self.rag_retriever = None
        
        # Инициализация RAG retriever
        self._init_rag_retriever()
    
    def _init_rag_retriever(self):
        """Инициализирует RAG retriever."""
        try:
            # Добавляем путь к корневой директории проекта для импорта
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, project_root)
            
            from rag_retriever import RAGRetriever
            
            # Указываем абсолютный путь к базе данных
            db_path = os.path.join(project_root, "metadata.db")
            self.rag_retriever = RAGRetriever(db_path=db_path)
            print("✅ RAG Retriever загружен")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить RAG Retriever: {e}")
            self.rag_retriever = None
    
    async def connect_mcp(self) -> bool:
        """
        Подключается к MCP серверам.
        
        Returns:
            True если подключение хотя бы одного сервера успешно
        """
        current_dir = os.path.dirname(__file__)
        
        # Подключаем основной сервер (yandex)
        yandex_script = os.path.join(current_dir, "mcp_server.py")
        yandex_success = await self.orchestrator.connect_server("yandex", yandex_script)
        
        # Подключаем сервер ПЭК
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
            # Не пробрасываем ошибку дальше, чтобы не падать
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Вызывает MCP инструмент через оркестратор.
        
        Args:
            tool_name: Название инструмента (с префиксом сервера)
            arguments: Аргументы инструмента
        
        Returns:
            Результат вызова в виде строки
        """
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
    
    async def process_message(self, user_message: str, callback=None) -> str:
        """
        Обрабатывает сообщение пользователя с интеллектуальной маршрутизацией между RAG и MCP.
        
        Args:
            user_message: Сообщение пользователя
            callback: Функция обратного вызова для стриминга
        
        Returns:
            Ответ агента
        """
        # Добавляем сообщение пользователя в историю
        self.state.messages.append({"role": "user", "content": user_message})
        
        # Определяем стратегию обработки
        use_rag = self._should_use_rag(user_message)
        
        if callback:
            await callback(f"📊 Анализ запроса... {'📚 RAG' if use_rag else '🔧 MCP'} режим")
        
        if use_rag and self.rag_retriever:
            # Режим RAG: поиск в документах
            try:
                if callback:
                    await callback("🔍 Ищу информацию в документах...")
                
                # Поиск релевантных чанков
                chunks = self.search_with_rag(user_message, top_k=3)
                
                if chunks:
                    # Строим промпт с RAG контекстом
                    rag_prompt = self._build_rag_prompt(user_message, chunks)
                    
                    if callback:
                        await callback(f"📄 Найдено {len(chunks)} релевантных фрагментов")
                    
                    # Получаем ответ от LLM с RAG контекстом
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": rag_prompt}
                    ]
                    
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=1024
                    )
                    
                    answer = response.choices[0].message.content or "Не удалось получить ответ"
                    
                    # Добавляем информацию об использованных источниках
                    sources = ", ".join(set(chunk['filename'] for chunk in chunks))
                    final_response = f"📚 RAG (источники: {sources})\n\n{answer}"
                    
                else:
                    # Если ничего не найдено, используем обычный режим
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ]
                    
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=1024
                    )
                    
                    answer = response.choices[0].message.content or "Не удалось получить ответ"
                    final_response = f"💡 (RAG ничего не нашёл, использованы знания)\n\n{answer}"
                
                self.state.messages.append({"role": "assistant", "content": final_response})
                return final_response
                
            except Exception as e:
                error_msg = f"❌ Ошибка RAG: {e}"
                self.state.messages.append({"role": "assistant", "content": error_msg})
                return error_msg
        
        else:
            # Режим MCP: использование инструментов для расчетов
            return await self._process_with_mcp(user_message, callback)
    
    async def _process_with_mcp(self, user_message: str, callback=None) -> str:
        """
        Обрабатывает сообщение с использованием MCP инструментов.
        
        Args:
            user_message: Сообщение пользователя
            callback: Функция обратного вызова
        
        Returns:
            Ответ агента
        """
        # Формируем контекст для LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # Получаем все инструменты от оркестратора
        mcp_tools = self.orchestrator.get_all_tools() if self.state.mcp_available else []
        
        try:
            # Первый запрос к LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=mcp_tools if mcp_tools else None,
                max_tokens=1024
            )
            
            assistant_message = response.choices[0].message
            
            # Обрабатываем вызовы инструментов с ограничением по количеству итераций
            max_iterations = 10
            iteration_count = 0
            
            while assistant_message.tool_calls and iteration_count < max_iterations:
                iteration_count += 1
                
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    # Создаем запись с tool_calls в старом формате
                    tool_calls_list = []
                    for tc in assistant_message.tool_calls:
                        try:
                            # Пробуем разные способы доступа к данным
                            if hasattr(tc, 'function'):
                                if hasattr(tc.function, 'name') and hasattr(tc.function, 'arguments'):
                                    tool_calls_list.append({
                                        "id": tc.id if hasattr(tc, 'id') else str(hash(tc)),
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments
                                        }
                                    })
                                else:
                                    # Альтернативный способ
                                    tool_data = getattr(tc, 'function', {})
                                    tool_calls_list.append({
                                        "id": tc.id if hasattr(tc, 'id') else str(hash(tc)),
                                        "type": "function",
                                        "function": {
                                            "name": tool_data.get('name', 'unknown'),
                                            "arguments": tool_data.get('arguments', '{}')
                                        }
                                    })
                        except Exception as e:
                            print(f"⚠️ Ошибка обработки tool_call: {e}")
                            continue
                    
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": tool_calls_list
                    })
                
                for tool_call in assistant_message.tool_calls:
                    tool_name = None
                    arguments = {}
                    
                    try:
                        if hasattr(tool_call, 'function'):
                            if hasattr(tool_call.function, 'name') and hasattr(tool_call.function, 'arguments'):
                                tool_name = tool_call.function.name
                                arguments = json.loads(tool_call.function.arguments)
                            else:
                                tool_data = getattr(tool_call, 'function', {})
                                tool_name = tool_data.get('name', '')
                                arguments = json.loads(tool_data.get('arguments', '{}'))
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
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                
                # Запрашиваем следующий ответ от LLM
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=mcp_tools if mcp_tools else None,
                    max_tokens=1024
                )
                assistant_message = response.choices[0].message
            
            if iteration_count >= max_iterations and assistant_message.tool_calls:
                final_response = "❌ Превышено максимальное количество итераций. Возможно, проблема с интерпретацией запроса."
                self.state.messages.append({"role": "assistant", "content": final_response})
                return final_response
            
            # Финальный ответ
            final_response = assistant_message.content or "Не удалось получить ответ"
            
            # Добавляем информацию о статусе MCP
            prefix = "🔧 MCP"
            if not self.state.mcp_available:
                prefix = "⚠️ MCP недоступен"
                final_response += "\n\n⚠️ MCP инструменты недоступны. Ответ дан без использования инструментов."
            elif self.state.mcp_calls:
                success_count = sum(1 for c in self.state.mcp_calls if c.success)
                final_response += f"\n\n✅ Использовано MCP инструментов: {success_count}/{len(self.state.mcp_calls)}"
            
            final_response = f"{prefix}\n\n{final_response}"
            self.state.messages.append({"role": "assistant", "content": final_response})
            return final_response
        
        except Exception as e:
            error_msg = f"❌ Ошибка MCP обработки: {e}"
            calls_info = f"\n📊 Вызовы инструментов: {len(self.state.mcp_calls)}"
            for i, call in enumerate(self.state.mcp_calls):
                status = "✅" if call.success else "❌"
                calls_info += f"\n  {i+1}. {call.tool_name}: {status}"
            
            error_msg += calls_info
            self.state.messages.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    def search_with_rag(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Ищет информацию в документах через RAG.
        
        Args:
            query: Текстовый запрос
            top_k: Количество возвращаемых результатов
            
        Returns:
            Список словарей с информацией о чанках или пустой список при ошибке
        """
        if not self.rag_retriever:
            return []
        
        try:
            return self.rag_retriever.search(query, top_k)
        except Exception as e:
            print(f"⚠️ Ошибка RAG поиска: {e}")
            return []
    
    def _build_rag_prompt(self, query: str, chunks: List[Dict]) -> str:
        """
        Строит промпт с RAG контекстом.
        
        Args:
            query: Исходный запрос
            chunks: Найденные чанки
            
        Returns:
            Промпт с контекстом
        """
        if not chunks:
            return query
        
        context = "\n".join([
            f"📄 [{chunk['filename']}] {chunk['text']}"
            for chunk in chunks
        ])
        
        return f"""Пользователь задал вопрос: {query}

## Релевантная информация из документов:
{context}

Ответь на вопрос пользователя, используя информацию из документов если она есть.
Если информации в документах недостаточно, дополни ответ своими знаниями.
Всегда указывай источники информации."""
    
    def _should_use_rag(self, user_message: str) -> bool:
        """
        Определяет, нужно ли использовать RAG для данного запроса.
        
        Args:
            user_message: Сообщение пользователя
            
        Returns:
            True если нужно использовать RAG
        """
        # Ключевые слова для RAG
        rag_keywords = [
            'тариф', 'правила', 'обязанности', 'постановление', 'закон',
            'стоимость', 'цена', 'расценки', 'условия', 'требования',
            'api', 'документ', 'инструкция', 'руководство', 'справочник',
            'фрахтователь', 'фрахтовщик', 'перевозчик', 'экспедитор'
        ]
        
        # Ключевые слова для MCP (расчетные запросы)
        mcp_keywords = [
            'расстояние', 'маршрут', 'координаты', 'город', 'городa',
            'рассчитать', 'стоимость доставки', 'цена перевозки',
            'оптимальный путь', 'кратчайший маршрут', 'геокодирование'
        ]
        
        message_lower = user_message.lower()
        
        # Если есть ключевые слова для RAG и нет явных маршрутных запросов
        has_rag_keywords = any(keyword in message_lower for keyword in rag_keywords)
        has_mcp_keywords = any(keyword in message_lower for keyword in mcp_keywords)
        
        # Приоритет RAG для информационных запросов
        if has_rag_keywords and not has_mcp_keywords:
            return True
        
        # Если это общий информационный запрос без явных маршрутных указаний
        if not has_mcp_keywords and len(user_message.split()) > 3:
            return True
            
        return False
    
    def get_mcp_calls(self) -> List[MCPToolCall]:
        """Возвращает список вызовов MCP инструментов."""
        return self.state.mcp_calls
    
    def clear_history(self):
        """Очищает историю сообщений."""
        self.state.messages.clear()
        self.state.mcp_calls.clear()


# Список моделей OpenRouter (согласно PRD)
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