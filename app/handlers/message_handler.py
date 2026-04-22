"""
Обработчик сообщений пользователя.
Координация работы агента, обновление UI и обработка ошибок.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import gradio as gr

from utils.async_helpers import run_in_background
from core.agent import LogsitAgent
from services.mcp_orchestrator import MCPOrchestrator
# RAG service is optional
try:
    from services.rag_service import get_rag_service
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    
    def get_rag_service(*args, **kwargs):
        return None
from app.components.sidebar import SidebarComponent
from app.components.chat import ChatComponent


class MessageHandler:
    """Обработчик сообщений."""
    
    def __init__(self, agent: LogsitAgent, sidebar: SidebarComponent, chat: ChatComponent):
        self.agent = agent
        self.sidebar = sidebar
        self.chat = chat
        self._processing = False
    
    async def handle_message(self, message: str, history: list) -> Tuple[list, str, str, str, dict, str]:
        """
        Обрабатывает сообщение пользователя.
        
        Args:
            message: Сообщение пользователя
            history: История чата
            
        Returns:
            Кортеж с обновленными компонентами:
            (история, отладка, ввод, статус, диалоги, информация)
        """
        if not message.strip():
            return self._get_empty_response()
        
        # Устанавливаем флаг обработки
        self._processing = True
        
        try:
            # Обрабатываем сообщение через агента
            response = await self.agent.process_message(message)
            
            # Обновляем историю чата
            new_history = history + [(message, response)]
            
            # Получаем отладочную информацию
            debug_output = self._format_debug_output()
            
            # Получаем статус MCP
            mcp_status = self._get_mcp_status()
            
            # Обновляем список диалогов
            conversations_update = self._update_conversations_list()
            
            # Получаем информацию о текущем диалоге
            conv_info = self._get_conversation_info()
            
            return (
                new_history,
                debug_output,
                "",
                mcp_status,
                conversations_update,
                conv_info
            )
            
        except Exception as e:
            print(f"❌ Ошибка обработки сообщения: {e}")
            error_response = f"❌ Произошла ошибка при обработке запроса: {str(e)}"
            
            new_history = history + [(message, error_response)]
            
            return (
                new_history,
                f"Ошибка: {e}",
                "",
                self._get_mcp_status(),
                self._update_conversations_list(),
                self._get_conversation_info()
            )
        
        finally:
            self._processing = False
    
    def _get_empty_response(self) -> Tuple[list, str, str, str, dict, str]:
        """Возвращает пустой ответ для пустого сообщения."""
        return (
            [],  # история
            self._format_debug_output(),  # отладка
            "",  # ввод
            self._get_mcp_status(),  # статус
            self._update_conversations_list(),  # диалоги
            self._get_conversation_info()  # информация
        )
    
    def _format_debug_output(self) -> str:
        """Форматирует отладочную информацию."""
        mcp_calls = self.agent.get_mcp_calls()
        
        if not mcp_calls:
            return "📭 Пока нет вызовов MCP"
        
        lines = []
        for i, call in enumerate(reversed(mcp_calls[-10:]), 1):  # Последние 10 вызовов
            status = "✅" if call.success else "❌"
            server_info = f"[{call.server_name}] " if call.server_name else ""
            
            lines.append(f"[{i}] {status} {server_info}{call.tool_name}")
            
            # Обрезаем длинные аргументы
            import json
            args_str = json.dumps(call.arguments, ensure_ascii=False)
            if len(args_str) > 100:
                args_str = args_str[:97] + "..."
            lines.append(f"    📝 Аргументы: {args_str}")
            
            if call.success and call.result:
                result_preview = call.result[:100] + "..." if len(call.result) > 100 else call.result
                lines.append(f"    📊 Результат: {result_preview}")
            
            if call.error:
                lines.append(f"    ❗ Ошибка: {call.error}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _get_mcp_status(self) -> str:
        """Получает статус MCP и RAG."""
        result = []
        
        # Статус MCP
        if self.agent.state.mcp_available:
            mcp_servers = self.agent.mcp_orchestrator.connected_servers
            if mcp_servers:
                result.append("📡 MCP серверы:")
                for server in mcp_servers:
                    result.append(f"  ✅ {server}")
            else:
                result.append("📡 MCP серверы: ❌ Нет подключенных серверов")
        else:
            result.append("📡 MCP серверы: ❌ Не подключены")
        
        # Статус RAG
        rag_service = get_rag_service()
        rag_available = rag_service.is_available()
        
        result.append("")
        result.append("🔍 RAG статус:")
        
        if rag_available:
            result.append("  ✅ RAG доступен")
            
            # Получаем статистику RAG
            stats = rag_service.get_index_stats()
            if stats.get("available"):
                result.append(f"  📊 Чанков: {stats.get('total_chunks', 0)}")
                result.append(f"  📁 Файлов: {stats.get('total_files', 0)}")
        else:
            result.append("  ❌ RAG недоступен")
        
        return "\n".join(result)
    
    def _update_conversations_list(self) -> dict:
        """Обновляет список диалогов."""
        conversations = self.agent.get_all_conversations()
        conv_dicts = [conv.to_dict() for conv in conversations]
        
        choices = self.sidebar.format_conversation_choices(conv_dicts)
        
        # Определяем текущее значение
        current_conv = self.agent.get_current_conversation()
        value = str(current_conv.id) if current_conv else ""
        
        return gr.update(choices=choices, value=value)
    
    def _get_conversation_info(self) -> str:
        """Получает информацию о текущем диалоге."""
        conversation = self.agent.get_current_conversation()
        if not conversation:
            return "Нет активного диалога"
        
        lines = []
        lines.append(f"📝 **{conversation.title}**")
        lines.append("")
        lines.append(f"📊 Сообщений: {conversation.message_count}")
        lines.append(f"👤 Сообщений пользователя: {conversation.user_message_count}")
        
        # Добавляем информацию о суммаризации
        from core.summarizer import get_summarizer
        summarizer = get_summarizer()
        
        if conversation.id:
            summary = summarizer.get_summary_for_conversation(conversation.id)
            if summary:
                lines.append("")
                lines.append("📋 **Последняя суммаризация:**")
                if len(summary) > 200:
                    lines.append(f"   {summary[:197]}...")
                else:
                    lines.append(f"   {summary}")
            
            # Проверяем необходимость суммаризации
            should_summarize, user_count = summarizer.should_summarize(conversation.id)
            if user_count >= 10:
                if should_summarize:
                    lines.append("")
                    lines.append("📋 **Суммаризация:** требуется обновление (прошло ≥10 сообщений)")
                else:
                    lines.append("")
                    lines.append("📋 **Суммаризация:** актуальна")
                lines.append(f"   👤 Сообщений пользователя: {user_count}")
        
        return "\n".join(lines)
    
    def is_processing(self) -> bool:
        """Проверяет, идет ли обработка сообщения."""
        return self._processing
    
    def update_loading_indicator(self) -> str:
        """Обновляет индикатор загрузки."""
        return self.sidebar.create_loading_indicator(self._processing)


class ConversationHandler:
    """Обработчик операций с диалогами."""
    
    def __init__(self, agent: LogsitAgent, sidebar: SidebarComponent, chat: ChatComponent):
        self.agent = agent
        self.sidebar = sidebar
        self.chat = chat
    
    def switch_conversation(self, conv_dropdown_value: str) -> Tuple[str, list, str, dict]:
        """
        Переключает текущий диалог.
        
        Args:
            conv_dropdown_value: Значение из выпадающего списка
            
        Returns:
            Кортеж (результат, история, информация, диалоги)
        """
        if not conv_dropdown_value:
            return "❌ Выберите диалог из списка", [], "", gr.update(choices=[], value="")
        
        try:
            conv_id = int(conv_dropdown_value)
            success = self.agent.switch_conversation(conv_id)
            
            if success:
                # Очищаем историю чата
                self.chat.clear_chat()
                
                # Получаем информацию о новом диалоге
                conv_info = self._get_conversation_info()
                
                # Обновляем список диалогов
                conversations_update = self._update_conversations_list()
                
                return (
                    "✅ Переключен на новый диалог",
                    [],
                    conv_info,
                    conversations_update
                )
            else:
                return "❌ Не удалось переключить диалог", [], "", self._update_conversations_list()
            
        except ValueError:
            return "❌ Неверный ID диалога", [], "", self._update_conversations_list()
        except Exception as e:
            return f"❌ Ошибка переключения: {e}", [], "", self._update_conversations_list()
    
    def create_new_conversation(self) -> Tuple[str, list, str, dict]:
        """
        Создает новый диалог.
        
        Returns:
            Кортеж (результат, история, информация, диалоги)
        """
        success = self.agent.create_new_conversation()
        
        if success:
            # Очищаем историю чата
            self.chat.clear_chat()
            
            # Получаем информацию о новом диалоге
            conv_info = self._get_conversation_info()
            
            # Обновляем список диалогов
            conversations_update = self._update_conversations_list()
            
            return (
                "✅ Создан новый диалог",
                [],
                conv_info,
                conversations_update
            )
        else:
            return "❌ Не удалось создать диалог", [], "", self._update_conversations_list()
    
    def delete_conversation(self, conv_dropdown_value: str) -> Tuple[str, dict, str]:
        """
        Удаляет диалог.
        
        Args:
            conv_dropdown_value: Значение из выпадающего списка
            
        Returns:
            Кортеж (результат, диалоги, информация)
        """
        if not conv_dropdown_value:
            return "❌ Выберите диалог для удаления", self._update_conversations_list(), ""
        
        try:
            conv_id = int(conv_dropdown_value)
            
            # Проверяем, не пытаемся ли удалить активный диалог
            current_conv = self.agent.get_current_conversation()
            if current_conv and current_conv.id == conv_id:
                return (
                    "❌ Нельзя удалить активный диалог. Переключитесь на другой диалог сначала.",
                    self._update_conversations_list(),
                    self._get_conversation_info()
                )
            
            # Удаляем диалог
            from core.conversation_manager import get_conversation_manager
            manager = get_conversation_manager()
            success = manager.delete_conversation(conv_id)
            
            if success:
                # Обновляем список диалогов
                conversations_update = self._update_conversations_list()
                
                return (
                    f"✅ Удален диалог ID: {conv_id}",
                    conversations_update,
                    self._get_conversation_info()
                )
            else:
                return (
                    "❌ Не удалось удалить диалог",
                    self._update_conversations_list(),
                    self._get_conversation_info()
                )
            
        except ValueError:
            return "❌ Неверный ID диалога", self._update_conversations_list(), ""
        except Exception as e:
            return f"❌ Ошибка удаления: {e}", self._update_conversations_list(), ""
    
    def _get_conversation_info(self) -> str:
        """Получает информацию о текущем диалоге."""
        conversation = self.agent.get_current_conversation()
        if not conversation:
            return "Нет активного диалога"
        
        lines = []
        lines.append(f"📝 **{conversation.title}**")
        lines.append("")
        lines.append(f"📊 Сообщений: {conversation.message_count}")
        lines.append(f"👤 Сообщений пользователя: {conversation.user_message_count}")
        
        return "\n".join(lines)
    
    def _update_conversations_list(self) -> dict:
        """Обновляет список диалогов."""
        conversations = self.agent.get_all_conversations()
        conv_dicts = [conv.to_dict() for conv in conversations]
        
        choices = self.sidebar.format_conversation_choices(conv_dicts)
        
        # Определяем текущее значение
        current_conv = self.agent.get_current_conversation()
        value = str(current_conv.id) if current_conv else ""
        
        return gr.update(choices=choices, value=value)