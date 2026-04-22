"""
Главное приложение Logsit Agent.
Объединенный UI с поддержкой всех функций.
"""

import gradio as gr
import asyncio
from typing import Dict, Any

from utils.config import settings
from utils.async_helpers import start_background_loop, stop_background_loop, get_background_loop
from core.agent import LogsitAgent
from app.components.sidebar import SidebarComponent
from app.components.chat import ChatComponent
from app.handlers.message_handler import MessageHandler, ConversationHandler


class LogsitApp:
    """Главное приложение Logsit Agent."""
    
    def __init__(self):
        self.agent = None
        self.sidebar = None
        self.chat = None
        self.message_handler = None
        self.conversation_handler = None
        self.demo = None
    
    def initialize(self):
        """Инициализирует приложение."""
        print("🚀 Инициализация Logsit Agent...")
        
        # Запускаем фоновый event loop
        start_background_loop()
        
        # Создаем агента
        self.agent = LogsitAgent(model=settings.default_model)
        
        # Создаем компоненты UI
        self.sidebar = SidebarComponent()
        self.chat = ChatComponent()
        
        # Создаем обработчики
        self.message_handler = MessageHandler(self.agent, self.sidebar, self.chat)
        self.conversation_handler = ConversationHandler(self.agent, self.sidebar, self.chat)
        
        print("✅ Приложение инициализировано")
    
    def create_ui(self):
        """Создает пользовательский интерфейс."""
        print("🎨 Создание UI...")
        
        with gr.Blocks(
            title="🗺️ Logsit Agent",
            theme=gr.themes.Soft()
        ) as self.demo:
            gr.Markdown("# 🗺️ Logsit Agent")
            gr.Markdown("Умный логистический ассистент с поддержкой множества диалогов")
            
            with gr.Row():
                # Боковая панель
                sidebar_components = self.sidebar.create()
                
                # Чат
                chat_components = self.chat.create()
            
            # Регистрируем обработчики событий
            self._register_event_handlers(sidebar_components, chat_components)
        
        print("✅ UI создан")
        return self.demo
    
    def _register_event_handlers(self, sidebar_components: Dict[str, Any], 
                                chat_components: Dict[str, Any]):
        """Регистрирует обработчики событий."""
        
        # Обработка сообщений через Enter
        chat_components["msg_input"].submit(
            fn=self.message_handler.handle_message,
            inputs=[chat_components["msg_input"], chat_components["chatbot"]],
            outputs=[
                chat_components["chatbot"],
                sidebar_components["debug_output"],
                chat_components["msg_input"],
                sidebar_components["mcp_status"],
                sidebar_components["conversations_dropdown"],
                sidebar_components["conversation_info"]
            ],
            show_progress="minimal"
        )
        
        # Обработка сообщений через кнопку
        chat_components["send_btn"].click(
            fn=self.message_handler.handle_message,
            inputs=[chat_components["msg_input"], chat_components["chatbot"]],
            outputs=[
                chat_components["chatbot"],
                sidebar_components["debug_output"],
                chat_components["msg_input"],
                sidebar_components["mcp_status"],
                sidebar_components["conversations_dropdown"],
                sidebar_components["conversation_info"]
            ],
            show_progress="minimal"
        )
        
        # Обновление модели
        sidebar_components["model_dropdown"].change(
            fn=self._update_model,
            inputs=[sidebar_components["model_dropdown"]],
            outputs=[
                sidebar_components["model_status"],
                sidebar_components["conversations_dropdown"],
                sidebar_components["conv_action_result"]
            ]
        )
        
        # Переключение диалога
        sidebar_components["switch_conv_btn"].click(
            fn=self.conversation_handler.switch_conversation,
            inputs=[sidebar_components["conversations_dropdown"]],
            outputs=[
                sidebar_components["conv_action_result"],
                chat_components["chatbot"],
                sidebar_components["conversation_info"],
                sidebar_components["conversations_dropdown"]
            ]
        )
        
        # Создание нового диалога
        sidebar_components["new_conv_btn"].click(
            fn=self.conversation_handler.create_new_conversation,
            outputs=[
                sidebar_components["conv_action_result"],
                chat_components["chatbot"],
                sidebar_components["conversation_info"],
                sidebar_components["conversations_dropdown"]
            ]
        )
        
        # Переподключение MCP
        sidebar_components["reconnect_btn"].click(
            fn=self._reconnect_mcp,
            outputs=[
                sidebar_components["mcp_status"],
                sidebar_components["conversations_dropdown"],
                sidebar_components["conv_action_result"]
            ]
        )
        
        # Очистка текущего диалога
        sidebar_components["clear_btn"].click(
            fn=self._clear_current_history,
            outputs=[
                chat_components["chatbot"],
                sidebar_components["debug_output"],
                sidebar_components["conversation_info"]
            ]
        )
        
        # Удаление диалога
        sidebar_components["delete_conv_btn"].click(
            fn=self.conversation_handler.delete_conversation,
            inputs=[sidebar_components["conversations_dropdown"]],
            outputs=[
                sidebar_components["conv_action_result"],
                sidebar_components["conversations_dropdown"],
                sidebar_components["conversation_info"]
            ]
        )
        
        # Инициализация при загрузке
        self.demo.load(
            fn=self._on_load,
            outputs=[
                sidebar_components["mcp_status"],
                sidebar_components["conversations_dropdown"],
                sidebar_components["conversation_info"]
            ]
        )
        
        # Обновление информации о диалоге при изменении выбора
        sidebar_components["conversations_dropdown"].change(
            fn=self._update_conversation_info,
            inputs=[sidebar_components["conversations_dropdown"]],
            outputs=[sidebar_components["conversation_info"]]
        )
    
    def _update_model(self, model_name: str):
        """Обновляет модель агента."""
        print(f"🔄 Обновление модели на: {model_name}")
        
        # TODO: Реализовать обновление модели агента
        # Пока просто обновляем статус
        
        return (
            f"Модель: {model_name}",
            gr.update(choices=[], value=""),
            "Модель обновлена. Функциональность будет доступна в следующей версии."
        )
    
    def _reconnect_mcp(self):
        """Переподключает MCP серверы."""
        print("🔄 Переподключение MCP...")
        
        # Запускаем в фоновом потоке
        def _reconnect():
            loop = get_background_loop().loop
            if loop:
                future = asyncio.run_coroutine_threadsafe(
                    self.agent.disconnect_mcp(),
                    loop
                )
                future.result(timeout=10)
                
                future = asyncio.run_coroutine_threadsafe(
                    self.agent.connect_mcp(),
                    loop
                )
                future.result(timeout=10)
        
        try:
            _reconnect()
            status = "✅ MCP переподключен"
        except Exception as e:
            print(f"❌ Ошибка переподключения MCP: {e}")
            status = f"❌ Ошибка переподключения MCP: {e}"
        
        # Обновляем список диалогов
        conversations_update = self._update_conversations_list()
        
        return status, conversations_update, ""
    
    def _clear_current_history(self):
        """Очищает историю текущего диалога."""
        print("🗑️ Очистка истории текущего диалога...")
        
        self.agent.clear_history()
        
        return [], "История очищена", self._get_conversation_info()
    
    def _on_load(self):
        """Инициализация при загрузке страницы."""
        print("📱 Загрузка страницы...")
        
        # Подключаем MCP при первой загрузке
        if not self.agent.state.mcp_available:
            def _connect_mcp():
                loop = get_background_loop().loop
                if loop:
                    future = asyncio.run_coroutine_threadsafe(
                        self.agent.connect_mcp(),
                        loop
                    )
                    future.result(timeout=10)
            
            try:
                _connect_mcp()
            except Exception as e:
                print(f"⚠️ Не удалось подключить MCP при загрузке: {e}")
        
        # Получаем статус MCP
        mcp_status = self.message_handler._get_mcp_status()
        
        # Обновляем список диалогов
        conversations_update = self._update_conversations_list()
        
        # Получаем информацию о диалоге
        conv_info = self._get_conversation_info()
        
        return mcp_status, conversations_update, conv_info
    
    def _update_conversation_info(self, conv_dropdown_value: str):
        """Обновляет информацию о диалоге при изменении выбора."""
        if not conv_dropdown_value:
            return "Выберите диалог"
        
        try:
            conv_id = int(conv_dropdown_value)
            conversation = self.agent.conversation_manager.get_conversation(conv_id)
            
            if conversation:
                lines = []
                lines.append(f"📝 **{conversation.title}**")
                lines.append("")
                lines.append(f"📊 Сообщений: {conversation.message_count}")
                lines.append(f"👤 Сообщений пользователя: {conversation.user_message_count}")
                return "\n".join(lines)
            else:
                return "Диалог не найден"
        
        except ValueError:
            return "Неверный ID диалога"
    
    def _update_conversations_list(self):
        """Обновляет список диалогов."""
        conversations = self.agent.get_all_conversations()
        conv_dicts = [conv.to_dict() for conv in conversations]
        
        choices = self.sidebar.format_conversation_choices(conv_dicts)
        
        # Определяем текущее значение
        current_conv = self.agent.get_current_conversation()
        value = str(current_conv.id) if current_conv else ""
        
        return gr.update(choices=choices, value=value)
    
    def _get_conversation_info(self):
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
    
    def run(self):
        """Запускает приложение."""
        print(f"🌐 Запуск сервера на http://{settings.app_host}:{settings.app_port}")
        
        try:
            self.demo.launch(
                server_name=settings.app_host,
                server_port=settings.app_port,
                share=False,
                show_error=True,
                debug=settings.debug
            )
        except KeyboardInterrupt:
            print("\n🛑 Остановка приложения...")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Останавливает приложение."""
        print("🔌 Остановка приложения...")
        
        # Отключаем MCP
        if self.agent and self.agent.state.mcp_available:
            try:
                loop = get_background_loop().loop
                if loop:
                    future = asyncio.run_coroutine_threadsafe(
                        self.agent.disconnect_mcp(),
                        loop
                    )
                    future.result(timeout=10)
            except Exception as e:
                print(f"⚠️ Ошибка при отключении MCP: {e}")
        
        # Останавливаем фоновый event loop
        stop_background_loop()
        
        print("✅ Приложение остановлено")


def main():
    """Точка входа приложения."""
    app = LogsitApp()
    app.initialize()
    app.create_ui()
    app.run()


if __name__ == "__main__":
    main()