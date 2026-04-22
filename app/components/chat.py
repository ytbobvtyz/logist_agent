"""
Компонент чата для общения с ассистентом.
Основная область для ввода сообщений и просмотра истории.
"""

import gradio as gr
from typing import List, Dict, Any, Optional, Callable


class ChatComponent:
    """Компонент чата."""
    
    def __init__(self):
        self.components = {}
        self._callbacks = {}
    
    def create(self) -> Dict[str, Any]:
        """
        Создает компоненты чата.
        
        Returns:
            Словарь с компонентами
        """
        with gr.Column(scale=3):
            # Чат
            self.components["chatbot"] = gr.Chatbot(
                label="Чат с ассистентом",
                height=500,
                avatar_images=(None, None)  # Можно добавить аватарки
            )
            
            # Поле ввода сообщения
            with gr.Row():
                self.components["msg_input"] = gr.Textbox(
                    label="Сообщение",
                    placeholder="Введите сообщение...",
                    lines=2,
                    scale=4,
                    show_label=False,
                    container=False
                )
                self.components["send_btn"] = gr.Button(
                    "Отправить", 
                    variant="primary", 
                    scale=1
                )
            
            # Индикатор загрузки
            self.components["loading_indicator"] = gr.HTML(value="")
        
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
    
    def update_chat_history(self, history: List[Dict[str, str]]):
        """
        Обновляет историю чата.
        
        Args:
            history: История чата в формате [{"role": "...", "content": "..."}, ...]
            
        Returns:
            Обновленный компонент
        """
        if "chatbot" in self.components:
            # Конвертируем в формат Gradio Chatbot
            gradio_history = []
            for msg in history:
                if msg["role"] == "user":
                    gradio_history.append((msg["content"], None))
                elif msg["role"] == "assistant":
                    if gradio_history and gradio_history[-1][1] is None:
                        gradio_history[-1] = (gradio_history[-1][0], msg["content"])
                    else:
                        gradio_history.append((None, msg["content"]))
            
            return gr.update(value=gradio_history)
        return None
    
    def clear_chat(self):
        """
        Очищает историю чата.
        
        Returns:
            Обновленный компонент
        """
        if "chatbot" in self.components:
            return gr.update(value=[])
        return None
    
    def clear_input(self):
        """
        Очищает поле ввода сообщения.
        
        Returns:
            Обновленный компонент
        """
        if "msg_input" in self.components:
            return gr.update(value="")
        return None
    
    def disable_input(self):
        """
        Отключает поле ввода сообщения.
        
        Returns:
            Обновленный компонент
        """
        if "msg_input" in self.components:
            return gr.update(interactive=False)
        return None
    
    def enable_input(self):
        """
        Включает поле ввода сообщения.
        
        Returns:
            Обновленный компонент
        """
        if "msg_input" in self.components:
            return gr.update(interactive=True)
        return None
    
    def show_loading_indicator(self, show: bool = True):
        """
        Показывает или скрывает индикатор загрузки.
        
        Args:
            show: Показывать ли индикатор
            
        Returns:
            Обновленный компонент
        """
        if "loading_indicator" in self.components:
            if show:
                return gr.update(value=self._create_loading_html())
            else:
                return gr.update(value="")
        return None
    
    def _create_loading_html(self) -> str:
        """Создает HTML для индикатора загрузки."""
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
    
    def format_history_for_display(self, messages: List[Dict[str, str]]) -> List[tuple]:
        """
        Форматирует историю сообщений для отображения в Gradio Chatbot.
        
        Args:
            messages: Список сообщений в формате [{"role": "...", "content": "..."}, ...]
            
        Returns:
            История в формате Gradio Chatbot
        """
        gradio_history = []
        
        for msg in messages:
            if msg["role"] == "user":
                gradio_history.append((msg["content"], None))
            elif msg["role"] == "assistant":
                if gradio_history and gradio_history[-1][1] is None:
                    gradio_history[-1] = (gradio_history[-1][0], msg["content"])
                else:
                    gradio_history.append((None, msg["content"]))
        
        return gradio_history
    
    def add_message_to_history(self, history: List[tuple], 
                              role: str, content: str) -> List[tuple]:
        """
        Добавляет сообщение в историю чата.
        
        Args:
            history: Текущая история чата
            role: Роль отправителя ("user" или "assistant")
            content: Текст сообщения
            
        Returns:
            Обновленная история
        """
        if role == "user":
            history.append((content, None))
        elif role == "assistant":
            if history and history[-1][1] is None:
                history[-1] = (history[-1][0], content)
            else:
                history.append((None, content))
        
        return history