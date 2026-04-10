"""
Gradio UI для умного ассистента логиста.
Чат-интерфейс с боковой панелью для настройки и отладки.
"""

import gradio as gr
import asyncio
import sys
import os
from typing import List, Tuple, Dict

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(__file__))

from agent import RoutePlannerAgent, OPENROUTER_MODELS


class AppState:
    """Глобальное состояние приложения."""
    def __init__(self):
        self.agent: RoutePlannerAgent | None = None
        self.mcp_connected: bool = False
        self.selected_model: str = "qwen/qwen3.6-plus-preview:free"
        self.debug_log: List[str] = []


app_state = AppState()


async def init_agent_async(model: str) -> Tuple[RoutePlannerAgent | None, bool]:
    """
    Инициализация агента.
    
    Returns:
        Tuple (agent, success)
    """
    try:
        agent = RoutePlannerAgent(model=model)
        success = await agent.connect_mcp()
        return agent, success
    except Exception as e:
        print(f"Ошибка инициализации агента: {e}")
        return None, False


def init_agent(model: str) -> Tuple[RoutePlannerAgent | None, bool]:
    """Синхронная обёртка для инициализации агента."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(init_agent_async(model))
    finally:
        loop.close()


async def process_message_async(agent: RoutePlannerAgent | None, message: str) -> str:
    """Асинхронная обработка сообщения."""
    if not agent:
        return "❌ Агент не инициализирован"
    return await agent.process_message(message)


def process_message(agent: RoutePlannerAgent | None, message: str) -> str:
    """Синхронная обёртка для обработки сообщения."""
    if not agent:
        return "❌ Агент не инициализирован"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(process_message_async(agent, message))
    finally:
        loop.close()


def format_mcp_calls(agent: RoutePlannerAgent | None) -> str:
    """Форматирует вызовы MCP для отладки."""
    if not agent:
        return "Агент не инициализирован"
    
    calls = agent.get_mcp_calls()
    if not calls:
        return "Пока нет вызовов MCP"
    
    lines = []
    for i, call in enumerate(calls):
        status = "✅" if call.success else "❌"
        lines.append(f"[{i+1}] {status} {call.tool_name}")
        lines.append(f"    Аргументы: {call.arguments}")
        if call.success and call.result:
            result_preview = call.result[:200] + "..." if len(call.result) > 200 else call.result
            lines.append(f"    Результат: {result_preview}")
        if call.error:
            lines.append(f"    Ошибка: {call.error}")
        lines.append("")
    
    return "\n".join(lines)


def chat_response(message: str, history: List[List[Dict[str, str]]]) -> Tuple[List[List[Dict[str, str]]], str, str]:
    """
    Обрабатывает сообщение пользователя и возвращает ответ.
    
    Returns:
        Tuple (updated_history, debug_log, empty_string)
    """
    if not message.strip():
        return history, format_mcp_calls(app_state.agent), ""
    
    # Инициализация агента при первом сообщении
    if not app_state.agent:
        agent, success = init_agent(app_state.selected_model)
        app_state.agent = agent
        app_state.mcp_connected = success
    
    # Получаем ответ от агента
    response = process_message(app_state.agent, message)
    
    # Обновляем историю в формате для Chatbot
    # Gradio Chatbot ожидает, что история содержит пары сообщений
    # Каждый элемент истории - это список из двух сообщений: [user, assistant]
    # Мы добавляем новую пару сообщений к существующей истории
    new_history = history + [
        [{"role": "user", "content": message},
         {"role": "assistant", "content": response}]
    ]
    
    # Обновляем отладку
    debug = format_mcp_calls(app_state.agent)
    
    return new_history, debug, ""


def update_model(model_name: str):
    """Обновляет модель агента."""
    # Находим ID модели по имени
    model_id = None
    for m in OPENROUTER_MODELS:
        if m["name"] == model_name:
            model_id = m["id"]
            break
    
    if model_id and model_id != app_state.selected_model:
        app_state.selected_model = model_id
        # Отключаем старого агента
        if app_state.agent:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(app_state.agent.disconnect_mcp())
                loop.close()
            except Exception:
                pass
        app_state.agent = None
        app_state.mcp_connected = False
    
    return f"Модель: {model_name}"


def reconnect_mcp():
    """Переподключает MCP."""
    if app_state.agent:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(app_state.agent.disconnect_mcp())
            loop.close()
        except Exception:
            pass
    app_state.agent = None
    app_state.mcp_connected = False
    return "MCP переподключено. Статус обновится при следующем сообщении."


def clear_history():
    """Очищает историю."""
    if app_state.agent:
        app_state.agent.clear_history()
    return [], "История очищена"


def get_mcp_status() -> str:
    """Возвращает статус MCP."""
    if app_state.mcp_connected:
        return "✅ Подключено"
    else:
        return "❌ Не подключено"


# Создаём Gradio интерфейс
with gr.Blocks(
    title="🗺️ Твой логист-ассистент"
) as demo:
    gr.Markdown("# 🗺️ Твой логист-ассистент")
    gr.Markdown("Умный планировщик маршрутов между городами")
    
    with gr.Row():
        # Боковая панель
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Настройки")
            
            # Выбор модели
            model_names = [m["name"] for m in OPENROUTER_MODELS]
            model_dropdown = gr.Dropdown(
                choices=model_names,
                value=model_names[0],
                label="Выберите модель",
                interactive=True
            )
            
            model_status = gr.Markdown(f"Модель: {model_names[0]}")
            
            gr.Markdown("## 📡 Статус MCP")
            mcp_status = gr.Markdown(get_mcp_status())
            
            reconnect_btn = gr.Button("🔄 Переподключить MCP", variant="secondary")
            
            gr.Markdown("## 🔍 Отладка MCP")
            debug_output = gr.Textbox(
                label="Лог вызовов",
                lines=10,
                interactive=False
            )
            
            gr.Markdown("## 📜 История")
            clear_btn = gr.Button("🗑️ Очистить историю", variant="secondary")
        
        # Основная область - чат
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Чат с ассистентом",
                height=500
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Сообщение",
                    placeholder="Введите сообщение...",
                    lines=2,
                    scale=4,
                    show_label=False
                )
                send_btn = gr.Button("Отправить", variant="primary", scale=1)
    
    # Обработчики событий
    msg_input.submit(
        fn=chat_response,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, debug_output, msg_input],
        show_progress="hidden"
    )
    
    send_btn.click(
        fn=chat_response,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, debug_output, msg_input],
        show_progress="hidden"
    )
    
    model_dropdown.change(
        fn=update_model,
        inputs=[model_dropdown],
        outputs=[model_status]
    )
    
    reconnect_btn.click(
        fn=reconnect_mcp,
        outputs=[mcp_status]
    )
    
    clear_btn.click(
        fn=clear_history,
        outputs=[chatbot, debug_output]
    )
    
    # Авто-обновление статуса MCP при загрузке
    demo.load(
        fn=get_mcp_status,
        outputs=[mcp_status]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True,
        theme="soft",
        css="""
        .main-container { max-width: 1200px; margin: auto; }
        .chat-container { height: 500px; }
        .debug-container { height: 300px; overflow-y: auto; }
        """
    )