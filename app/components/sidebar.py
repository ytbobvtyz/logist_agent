"""
Компонент боковой панели UI.
Содержит настройки, управление диалогами и статус инструментов.
"""

import gradio as gr
from typing import List, Tuple, Dict, Any, Optional, Callable

from utils.config import settings


class SidebarComponent:
    """Компонент боковой панели."""
    
    def __init__(self):
        self.components = {}
        self._callbacks = {}
    
    def create(self) -> Dict[str, Any]:
        """
        Создает компоненты боковой панели.
        
        Returns:
            Словарь с компонентами
        """
        with gr.Column(scale=1):
            # Настройки
            gr.Markdown("## ⚙️ Настройки")
            
            # Выбор модели
            model_names = ["openrouter/free", "qwen/qwen3.6-plus-preview:free", 
                          "stepfun/step-3.5-flash:free"]
            self.components["model_dropdown"] = gr.Dropdown(
                choices=model_names,
                value=model_names[0],
                label="Выберите модель",
                interactive=True
            )
            
            self.components["model_status"] = gr.Markdown(f"Модель: {model_names[0]}")
            
            # Управление диалогами
            gr.Markdown("## 💬 Управление диалогами")
            
            self.components["conversation_info"] = gr.Markdown("Загрузка информации о диалоге...")
            
            # Список диалогов
            self.components["conversations_dropdown"] = gr.Dropdown(
                choices=[],
                label="Все диалоги",
                interactive=True,
                allow_custom_value=False
            )
            
            with gr.Row():
                self.components["switch_conv_btn"] = gr.Button(
                    "↻ Переключиться", 
                    variant="secondary", 
                    scale=1
                )
                self.components["new_conv_btn"] = gr.Button(
                    "➕ Новый диалог", 
                    variant="primary", 
                    scale=1
                )
            
            self.components["conv_action_result"] = gr.Markdown("")
            
            # Статус инструментов
            gr.Markdown("## 📡 Статус инструментов")
            
            self.components["mcp_status"] = gr.Textbox(
                value="Загрузка статуса...",
                label="Статус MCP и RAG",
                interactive=False,
                lines=6
            )
            
            self.components["reconnect_btn"] = gr.Button(
                "🔄 Переподключить MCP", 
                variant="secondary"
            )
            
            # Режимы работы
            gr.Markdown("## 🔍 Режимы работы")
            gr.Markdown("""
            **Агент автоматически выбирает инструменты:**
            
            - 📚 **RAG** - для поиска информации в документах
            - 🔧 **MCP** - для расчетов маршрутов и стоимости
            - 💡 **Знания** - для общих вопросов
            
            **Новые возможности:**
            - 💬 **Множество диалогов** - переключайтесь между задачами
            - 📋 **Автосуммаризация** - каждые 10 сообщений
            - 🎯 **Память задачи** - отслеживание контекста
            
            Система сама определяет, какой инструмент использовать!
            """)
            
            # Отладка MCP
            gr.Markdown("## 🔍 Отладка MCP")
            self.components["debug_output"] = gr.Textbox(
                label="Лог вызовов",
                lines=10,
                interactive=False
            )
            
            # Управление историей
            gr.Markdown("## 📜 История")
            self.components["clear_btn"] = gr.Button(
                "🗑️ Очистить текущий диалог", 
                variant="secondary"
            )
            self.components["delete_conv_btn"] = gr.Button(
                "❌ Удалить диалог", 
                variant="stop"
            )
        
        return self.components
    
    def register_callbacks(self, callbacks: Dict[str, Callable]):
        """
        Регистрирует callback функции для компонентов.
        
        Args:
            callbacks: Словарь {имя_компонента: callback}
        """
        self._callbacks.update(callbacks)
    
    def get_component(self, name: str):
        """
        Возвращает компонент по имени.
        
        Args:
            name: Имя компонента
            
        Returns:
            Компонент Gradio
        """
        return self.components.get(name)
    
    def get_all_components(self) -> Dict[str, Any]:
        """
        Возвращает все компоненты.
        
        Returns:
            Словарь с компонентами
        """
        return self.components.copy()
    
    def update_conversation_info(self, info: str):
        """
        Обновляет информацию о диалоге.
        
        Args:
            info: Информация о диалоге
        """
        if "conversation_info" in self.components:
            return gr.update(value=info)
        return None
    
    def update_conversations_list(self, choices: List[Tuple[str, str]], value: str = ""):
        """
        Обновляет список диалогов.
        
        Args:
            choices: Список диалогов в формате [(отображение, значение), ...]
            value: Текущее значение
            
        Returns:
            Обновленный компонент или None
        """
        if "conversations_dropdown" in self.components:
            return gr.update(choices=choices, value=value)
        return None
    
    def update_mcp_status(self, status: str):
        """
        Обновляет статус MCP.
        
        Args:
            status: Текст статуса
            
        Returns:
            Обновленный компонент или None
        """
        if "mcp_status" in self.components:
            return gr.update(value=status)
        return None
    
    def update_model_status(self, model_name: str):
        """
        Обновляет статус модели.
        
        Args:
            model_name: Имя модели
            
        Returns:
            Обновленный компонент или None
        """
        if "model_status" in self.components:
            return gr.update(value=f"Модель: {model_name}")
        return None
    
    def update_debug_output(self, debug_text: str):
        """
        Обновляет вывод отладки.
        
        Args:
            debug_text: Текст отладки
            
        Returns:
            Обновленный компонент или None
        """
        if "debug_output" in self.components:
            return gr.update(value=debug_text)
        return None
    
    def clear_conversation_result(self):
        """
        Очищает результат операций с диалогами.
        
        Returns:
            Обновленный компонент
        """
        if "conv_action_result" in self.components:
            return gr.update(value="")
        return None
    
    def format_conversation_choices(self, conversations: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        Форматирует список диалогов для выпадающего списка.
        
        Args:
            conversations: Список диалогов
            
        Returns:
            Список в формате [(отображение, значение), ...]
        """
        formatted = []
        
        for conv in conversations:
            # Добавляем иконку активности
            prefix = "● " if conv.get('active') else "○ "
            
            # Обрезаем длинные заголовки
            title = conv.get('title', 'Новый диалог')
            if len(title) > 30:
                title = title[:27] + "..."
            
            # Добавляем информацию о сообщениях
            msg_count = conv.get('message_count', 0)
            if msg_count > 0:
                title = f"{title} ({msg_count} сообщ.)"
            
            conv_id_str = str(conv.get('id', ''))
            formatted.append((f"{prefix}{title}", conv_id_str))
        
        return formatted
    
    def create_loading_indicator(self, processing: bool) -> str:
        """
        Создает индикатор загрузки.
        
        Args:
            processing: Показывать ли индикатор
            
        Returns:
            HTML индикатора
        """
        if processing:
            return """
                <div style='text-align: center; padding: 10px;'>
                    <div style='display: inline-flex; align-items: center; background: #f0f0f0; padding: 8px 16px; border-radius: 20px;'>
                        <div style='width: 16px; height: 16px; border: 2px solid #007bff; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 8px;'></div>
                        <span style='color: #666;'>Обработка запроса...</span>
                    </div>
                </div>
                <style>
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
            """
        else:
            return "<div style='display: none;'></div>"