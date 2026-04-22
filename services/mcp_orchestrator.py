"""
Оркестратор для управления несколькими MCP серверами.
Координация работы инструментов MCP для логистических расчетов.
"""

import asyncio
import sys
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from utils.config import settings


@dataclass
class MCPToolCall:
    """Запись о вызове MCP инструмента."""
    
    tool_name: str
    arguments: Dict[str, Any]
    success: bool = False
    result: Optional[str] = None
    error: Optional[str] = None
    server_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: "now")
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует запись в словарь."""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "success": self.success,
            "result": self.result[:500] + "..." if self.result and len(self.result) > 500 else self.result,
            "error": self.error,
            "server_name": self.server_name,
            "timestamp": self.timestamp
        }


class MCPOrchestrator:
    """Оркестратор для управления несколькими MCP серверами."""
    
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}  # server_name -> {tool_name -> tool_info}
        self._server_contexts: Dict[str, Any] = {}  # server_name -> context manager
        self._server_processes: Dict[str, Any] = {}  # server_name -> (read_stream, write_stream)
        self._exit_stack = AsyncExitStack()
        self._tool_calls: List[MCPToolCall] = []
        self._max_tool_calls = 100  # Максимальное количество хранимых вызовов
    
    async def connect_all_servers(self) -> bool:
        """
        Подключается ко всем MCP серверам из конфигурации.
        
        Returns:
            True если подключение успешно, иначе False
        """
        print("🔌 Подключение к MCP серверам...")
        
        success_count = 0
        total_servers = len(settings.mcp_servers)
        
        for server_name, server_script in settings.mcp_servers.items():
            try:
                success = await self.connect_server(server_name, server_script)
                if success:
                    success_count += 1
                    print(f"  ✅ {server_name}: подключен")
                else:
                    print(f"  ❌ {server_name}: не удалось подключиться")
            except Exception as e:
                print(f"  ❌ {server_name}: ошибка {e}")
        
        print(f"📡 Подключено {success_count}/{total_servers} MCP серверов")
        return success_count > 0
    
    async def connect_server(self, server_name: str, server_script: str) -> bool:
        """
        Подключает MCP сервер.
        
        Args:
            server_name: Имя сервера
            server_script: Путь к скрипту сервера
            
        Returns:
            True если подключение успешно, иначе False
        """
        try:
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[server_script]
            )
            
            # Создаем контекст для подключения
            stdio_ctx = stdio_client(server_params)
            
            # Используем AsyncExitStack для управления контекстами
            stdio_transport = await self._exit_stack.enter_async_context(stdio_ctx)
            read_stream, write_stream = stdio_transport
            
            # Создаем сессию
            session = ClientSession(read_stream, write_stream)
            await self._exit_stack.enter_async_context(session)
            
            # Инициализируем сессию
            await session.initialize()
            
            # Получаем список инструментов
            tools_response = await session.list_tools()
            server_tools = {}
            
            for tool in tools_response.tools:
                server_tools[tool.name] = {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
            
            # Сохраняем сессию и инструменты
            self.sessions[server_name] = session
            self.tools[server_name] = server_tools
            self._server_processes[server_name] = (read_stream, write_stream)
            
            print(f"  🔧 {server_name}: {len(server_tools)} инструментов")
            return True
            
        except Exception as e:
            print(f"  ❌ Ошибка подключения к серверу {server_name}: {e}")
            
            # Очищаем ресурсы при ошибке
            if server_name in self.sessions:
                del self.sessions[server_name]
            if server_name in self.tools:
                del self.tools[server_name]
            if server_name in self._server_processes:
                del self._server_processes[server_name]
            
            return False
    
    async def disconnect_all_servers(self):
        """Отключает все MCP серверы."""
        print("🔌 Отключение MCP серверов...")
        
        # Закрываем все контексты через AsyncExitStack
        await self._exit_stack.aclose()
        
        # Очищаем словари
        self.sessions.clear()
        self.tools.clear()
        self._server_processes.clear()
        self._server_contexts.clear()
        
        print("✅ MCP серверы отключены")
    
    async def call_tool(self, server_name: str, tool_name: str, 
                       arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Вызывает инструмент MCP.
        
        Args:
            server_name: Имя сервера
            tool_name: Имя инструмента
            arguments: Аргументы инструмента
            
        Returns:
            Кортеж (успех, результат или ошибка)
        """
        if server_name not in self.sessions:
            return False, f"Сервер {server_name} не подключен"
        
        if tool_name not in self.tools.get(server_name, {}):
            return False, f"Инструмент {tool_name} не найден на сервере {server_name}"
        
        try:
            session = self.sessions[server_name]
            
            # Вызываем инструмент
            result = await session.call_tool(tool_name, arguments)
            
            # Сохраняем запись о вызове
            tool_call = MCPToolCall(
                tool_name=tool_name,
                arguments=arguments,
                success=True,
                result=result.content[0].text if result.content else "Нет результата",
                server_name=server_name
            )
            self._add_tool_call(tool_call)
            
            return True, tool_call.result
            
        except Exception as e:
            error_msg = str(e)
            
            # Сохраняем запись об ошибке
            tool_call = MCPToolCall(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                error=error_msg,
                server_name=server_name
            )
            self._add_tool_call(tool_call)
            
            return False, error_msg
    
    def _add_tool_call(self, tool_call: MCPToolCall):
        """Добавляет запись о вызове инструмента."""
        self._tool_calls.append(tool_call)
        
        # Ограничиваем количество хранимых записей
        if len(self._tool_calls) > self._max_tool_calls:
            self._tool_calls = self._tool_calls[-self._max_tool_calls:]
    
    def get_tool_calls(self) -> List[MCPToolCall]:
        """
        Возвращает историю вызовов инструментов.
        
        Returns:
            Список записей о вызовах
        """
        return self._tool_calls.copy()
    
    def clear_tool_calls(self):
        """Очищает историю вызовов инструментов."""
        self._tool_calls.clear()
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """
        Возвращает список доступных инструментов по серверам.
        
        Returns:
            Словарь {server_name: [tool_names]}
        """
        return {
            server_name: list(tools.keys())
            for server_name, tools in self.tools.items()
        }
    
    def get_tool_info(self, server_name: str, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает информацию об инструменте.
        
        Args:
            server_name: Имя сервера
            tool_name: Имя инструмента
            
        Returns:
            Информация об инструменте или None
        """
        return self.tools.get(server_name, {}).get(tool_name)
    
    def format_tool_calls_for_display(self) -> str:
        """
        Форматирует историю вызовов для отображения.
        
        Returns:
            Отформатированная строка с историей вызовов
        """
        if not self._tool_calls:
            return "📭 Пока нет вызовов MCP"
        
        lines = []
        for i, call in enumerate(reversed(self._tool_calls[-20:]), 1):  # Последние 20 вызовов
            status = "✅" if call.success else "❌"
            server_info = f"[{call.server_name}] " if call.server_name else ""
            
            lines.append(f"[{i}] {status} {server_info}{call.tool_name}")
            lines.append(f"    📝 Аргументы: {json.dumps(call.arguments, ensure_ascii=False)}")
            
            if call.success and call.result:
                result_preview = call.result[:200] + "..." if len(call.result) > 200 else call.result
                lines.append(f"    📊 Результат: {result_preview}")
            
            if call.error:
                lines.append(f"    ❗ Ошибка: {call.error}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Проверяет доступность всех серверов.
        
        Returns:
            Словарь {server_name: доступен}
        """
        health_status = {}
        
        for server_name in list(self.sessions.keys()):
            try:
                # Пробуем получить список инструментов
                session = self.sessions[server_name]
                await session.list_tools()
                health_status[server_name] = True
            except Exception:
                health_status[server_name] = False
        
        return health_status
    
    @property
    def connected_servers(self) -> List[str]:
        """Возвращает список подключенных серверов."""
        return list(self.sessions.keys())
    
    @property
    def total_tools(self) -> int:
        """Возвращает общее количество доступных инструментов."""
        return sum(len(tools) for tools in self.tools.values())