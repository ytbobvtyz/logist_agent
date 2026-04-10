"""
LLM агент для планирования маршрутов.
Использует OpenAI API через OpenRouter и MCP инструменты.
"""

import json
import os
import sys
import asyncio
import anyio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Загружаем переменные окружения
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SYSTEM_PROMPT = """Ты агент-планировщик маршрутов.

Когда пользователь просит найти маршрут между городами:
1. Вызови geocode_batch(cities=[...])
2. Передай результат в find_optimal_route(coordinates_json=...)
3. Передай результат в format_route_summary(route_json=...)
4. Ответь пользователю итоговым текстом

Важно: вызывай инструменты строго по порядку.
Важно: запрещено пользоваться поиском по интернету, до тех пор пока доступны инструменты mcp
Важно: сообщи пользователю использовал ли ты mcp или дал ответ без него (например, из-за недоступности сервиса)

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
    messages: List[Dict[str, str]] = field(default_factory=list)
    mcp_calls: List[MCPToolCall] = field(default_factory=list)
    mcp_available: bool = False


class RoutePlannerAgent:
    """Агент для планирования маршрутов с MCP инструментами."""
    
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
        self.mcp_session: Optional[ClientSession] = None
        self.mcp_tools: List[Dict[str, Any]] = []
        self.state = AgentState()
        self._mcp_cm = None  # Контекстный менеджер MCP
    
    async def connect_mcp(self) -> bool:
        """
        Подключается к MCP серверу.
        
        Returns:
            True если подключение успешно
        """
        try:
            # Путь к MCP серверу
            server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
            server_params = StdioServerParameters(
                command=sys.executable,  # Используем текущий Python
                args=[server_script]
            )
            
            # Создаём контекстный менеджер
            self._mcp_cm = stdio_client(server_params)
            
            # Входим в контекст
            read_stream, write_stream = await self._mcp_cm.__aenter__()
            
            # Создаём сессию
            self.mcp_session = ClientSession(read_stream, write_stream)
            await self.mcp_session.__aenter__()
            await self.mcp_session.initialize()
            
            # Получаем список доступных инструментов
            tools_result = await self.mcp_session.list_tools()
            self.mcp_tools = []
            for tool in tools_result.tools:
                # Преобразуем схему параметров в формат OpenAI
                parameters = tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}}
                self.mcp_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": parameters
                    }
                })
            
            self.state.mcp_available = True
            print(f"MCP подключено. Инструменты: {[t['function']['name'] for t in self.mcp_tools]}")
            return True
        
        except Exception as e:
            print(f"Ошибка подключения к MCP: {e}")
            import traceback
            traceback.print_exc()
            self.state.mcp_available = False
            return False
    
    async def disconnect_mcp(self):
        """Отключается от MCP сервера."""
        self.state.mcp_available = False
        
        # Закрываем сессию
        if self.mcp_session:
            try:
                await self.mcp_session.__aexit__(None, None, None)
            except Exception:
                pass
            self.mcp_session = None
        
        # Не вызываем __aexit__ на контекстном менеджере, т.к. это вызывает
        # cancel scope error при вызове из другого потока/задачи.
        # Вместо этого просто обнуляем ссылку — процесс MCP сервера
        # завершится сам при закрытии stdin или при завершении приложения.
        self._mcp_cm = None
        self.mcp_tools = []
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Вызывает MCP инструмент.
        
        Args:
            tool_name: Название инструмента
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
            if not self.mcp_session:
                raise Exception("MCP сессия не инициализирована")
            
            result = await self.mcp_session.call_tool(tool_name, arguments)
            # Извлекаем текст из результата
            result_text = ""
            if hasattr(result, 'content'):
                for content in result.content:
                    if hasattr(content, 'text'):
                        result_text += content.text
            
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
        Обрабатывает сообщение пользователя.
        
        Args:
            user_message: Сообщение пользователя
            callback: Функция обратного вызова для стриминга
        
        Returns:
            Ответ агента
        """
        # Добавляем сообщение пользователя в историю
        self.state.messages.append({"role": "user", "content": user_message})
        
        # Формируем контекст для LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.state.messages
        ]
        
        try:
            # Первый запрос к LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.mcp_tools if self.state.mcp_available else None,
                max_tokens=1024
            )
            
            assistant_message = response.choices[0].message
            
            # Обрабатываем вызовы инструментов
            while assistant_message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
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
                    tools=self.mcp_tools if self.state.mcp_available else None,
                    max_tokens=1024
                )
                assistant_message = response.choices[0].message
            
            # Финальный ответ
            final_response = assistant_message.content or "Не удалось получить ответ"
            
            # Добавляем информацию о статусе MCP
            if not self.state.mcp_available:
                final_response += "\n\n⚠️ MCP инструменты недоступны. Ответ дан без использования инструментов."
            elif self.state.mcp_calls:
                success_count = sum(1 for c in self.state.mcp_calls if c.success)
                final_response += f"\n\n✅ Использовано MCP инструментов: {success_count}/{len(self.state.mcp_calls)}"
            
            self.state.messages.append({"role": "assistant", "content": final_response})
            return final_response
        
        except Exception as e:
            error_msg = f"❌ Ошибка агента: {e}"
            self.state.messages.append({"role": "assistant", "content": error_msg})
            return error_msg
    
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