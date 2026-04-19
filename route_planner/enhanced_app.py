"""
Расширенный Gradio UI с поддержкой:
- Нескольких диалогов с переключением
- Автоматической суммаризации
- Task State (память задачи)
"""

import gradio as gr
import asyncio
import sys
import os
import threading
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from enhanced_agent import EnhancedRoutePlannerAgent, OPENROUTER_MODELS


# ── Фоновый event loop в отдельном потоке ──────────────────────────────

_bg_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
_bg_thread: threading.Thread


def _run_bg_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


_bg_thread = threading.Thread(target=_run_bg_loop, args=(_bg_loop,), daemon=True)
_bg_thread.start()


def run_async(coro):
    """Запускает корутину на фоновом loop'е и ждёт результат."""
    future = asyncio.run_coroutine_threadsafe(coro, _bg_loop)
    return future.result(timeout=120)


# ── Состояние приложения ────────────────────────────────────────────────

class AppState:
    """Глобальное состояние приложения."""
    def __init__(self):
        self.agent: Optional[EnhancedRoutePlannerAgent] = None
        self.mcp_connected: bool = False
        self.mcp_servers_status: Dict[str, bool] = {}
        self.selected_model: str = "openrouter/free"
        self.rag_available: bool = False
        self.conversations_list: List[Dict] = []
        self.current_conversation_id: Optional[int] = None


app_state = AppState()


# ── Функции-обёртки (синхронные, вызываются из Gradio) ──────────────────

def init_agent(model: str) -> Tuple[Optional[EnhancedRoutePlannerAgent], bool]:
    """Инициализация агента на фоновом loop'е."""
    async def _init():
        try:
            agent = EnhancedRoutePlannerAgent(model=model)
            success = await agent.connect_mcp()
            
            # Сохраняем статус серверов
            if agent.orchestrator.sessions:
                app_state.mcp_servers_status = {
                    name: True for name in agent.orchestrator.sessions.keys()
                }
            
            # Проверяем доступность RAG
            app_state.rag_available = agent.rag_retriever is not None
            if app_state.rag_available:
                print("✅ RAG доступен в приложении")
            
            # Получаем список диалогов
            app_state.conversations_list = agent.get_all_conversations()
            app_state.current_conversation_id = agent.get_current_conversation().id
            
            return agent, success
        except Exception as e:
            print(f"Ошибка инициализации агента: {e}")
            return None, False

    try:
        return run_async(_init())
    except Exception as e:
        print(f"Ошибка инициализации агента: {e}")
        return None, False


def process_message(agent: Optional[EnhancedRoutePlannerAgent], message: str) -> str:
    """Синхронная обёртка для обработки сообщения."""
    if not agent:
        return "❌ Агент не инициализирован"

    async def _process():
        return await agent.process_message(message)

    try:
        result = run_async(_process())
        if "API ПЭК вернул ошибку HTTP" in result:
            return "Cервис ПЭК в данный момент недоступен. Пожалуйста, попробуйте позже."
        return result
    except Exception as e:
        return f"❌ Ошибка: {e}"


def format_mcp_calls(agent: Optional[EnhancedRoutePlannerAgent]) -> str:
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


def get_conversations_choices(agent: Optional[EnhancedRoutePlannerAgent]) -> List[Tuple[str, str]]:
    """
    Получает список диалогов для выпадающего списка.
    
    Returns:
        Список кортежей (отображаемый_текст, id_диалога)
    """
    if not agent:
        return []
    
    conversations = agent.get_all_conversations()
    app_state.conversations_list = conversations
    
    # Форматируем для отображения
    formatted = []
    for conv in conversations:
        # Добавляем иконку активности
        prefix = "● " if conv['active'] else "○ "
        
        # Обрезаем длинные заголовки
        title = conv['title']
        if len(title) > 30:
            title = title[:27] + "..."
        
        # Добавляем информацию о сообщениях
        msg_count = conv['message_count']
        if msg_count > 0:
            title = f"{title} ({msg_count} сообщ.)"
        
        conv_id_str = str(conv['id'])
        formatted.append((f"{prefix}{title}", conv_id_str))
    
    return formatted


def get_current_conversation_value(agent: Optional[EnhancedRoutePlannerAgent]) -> str:
    """
    Получает текущее значение для выпадающего списка диалогов.
    
    Returns:
        ID текущего диалога в виде строки
    """
    if not agent or not app_state.current_conversation_id:
        return ""
    
    return str(app_state.current_conversation_id)


def switch_conversation(agent: Optional[EnhancedRoutePlannerAgent], conv_id_str: str) -> str:
    """
    Переключает текущий диалог.
    
    Args:
        agent: Экземпляр агента
        conv_id_str: ID диалога в виде строки
        
    Returns:
        Сообщение о результате
    """
    if not agent:
        return "❌ Агент не инициализирован"
    
    try:
        conv_id = int(conv_id_str)
        success = agent.switch_conversation(conv_id)
        
        if success:
            app_state.current_conversation_id = conv_id
            
            # Обновляем информацию о текущем диалоге
            conversation = agent.get_current_conversation()
            
            # Получаем статистику диалога
            stats = []
            stats.append(f"📝 Диалог: {conversation.title}")
            stats.append(f"📊 Сообщений: {conversation.message_count}")
            stats.append(f"👤 Сообщений пользователя: {conversation.user_message_count}")
            stats.append(f"🕐 Создан: {conversation.created_at[:19]}")
            
            # Обновляем список диалогов
            app_state.conversations_list = agent.get_all_conversations()
            
            return "✅ Переключен на новый диалог:\n\n" + "\n".join(stats)
        else:
            return "❌ Не удалось переключить диалог"
    
    except ValueError:
        return "❌ Неверный ID диалога"
    except Exception as e:
        return f"❌ Ошибка переключения: {e}"


def create_new_conversation(agent: Optional[EnhancedRoutePlannerAgent], title: str = None) -> str:
    """
    Создает новый диалог.
    
    Args:
        agent: Экземпляр агента
        title: Заголовок диалога (опционально)
        
    Returns:
        Сообщение о результате
    """
    if not agent:
        return "❌ Агент не инициализирован"
    
    try:
        success = agent.create_new_conversation(title)
        
        if success:
            # Обновляем список диалогов
            app_state.conversations_list = agent.get_all_conversations()
            app_state.current_conversation_id = agent.get_current_conversation().id
            
            conversation = agent.get_current_conversation()
            return f"✅ Создан новый диалог: {conversation.title}"
        else:
            return "❌ Не удалось создать диалог"
    
    except Exception as e:
        return f"❌ Ошибка создания диалога: {e}"


def get_current_conversation_info(agent: Optional[EnhancedRoutePlannerAgent]) -> str:
    """
    Получает информацию о текущем диалоге.
    
    Returns:
        Информация о диалоге в виде строки
    """
    if not agent or not app_state.current_conversation_id:
        return "Нет активного диалога"
    
    try:
        conversation = agent.get_current_conversation()
        if not conversation:
            return "❌ Диалог не найден"
        
        lines = []
        lines.append(f"📝 **{conversation.title}**")
        lines.append("")
        lines.append(f"📊 Сообщений: {conversation.message_count}")
        lines.append(f"👤 Сообщений пользователя: {conversation.user_message_count}")
        lines.append(f"🕐 Создан: {conversation.created_at[:19]}")
        lines.append(f"✏️ Обновлен: {conversation.updated_at[:19]}")
        lines.append("")
        
        # Получаем статистику суммаризации
        from conversation_manager import get_conversation_manager
        manager = get_conversation_manager()
        should_summarize, user_count = manager.should_summarize(conversation.id)
        
        if user_count >= 10:
            if should_summarize:
                lines.append("📋 **Суммаризация:** требуется обновление (прошло ≥10 сообщений)")
            else:
                lines.append("📋 **Суммаризация:** актуальна")
            lines.append(f"   👤 Сообщений пользователя: {user_count}")
        
        # Получаем информацию о состоянии задачи
        from task_state import get_task_state_manager
        task_manager = get_task_state_manager()
        task_state = task_manager.get_task_state(conversation.id)
        
        if task_state and task_state.goal:
            lines.append("")
            lines.append("🎯 **Цель диалога:**")
            lines.append(f"   {task_state.goal}")
        
        if task_state and task_state.clarified_details:
            lines.append("")
            lines.append("📋 **Уточненные детали:**")
            for detail in task_state.clarified_details[:5]:
                lines.append(f"   • {detail}")
            if len(task_state.clarified_details) > 5:
                lines.append(f"   ... и еще {len(task_state.clarified_details) - 5}")
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"❌ Ошибка получения информации: {e}"


def chat_response(message: str, history: list) -> Tuple[list, str, bool, str, list, str]:
    """
    Обрабатывает сообщение пользователя и возвращает ответ.
    
    Returns:
        Tuple (updated_history, debug_log, processing_complete, mcp_status, conversations_list, conv_info)
    """
    if not message.strip():
        return (
            history, 
            format_mcp_calls(app_state.agent), 
            True,
            get_mcp_status(),
            get_conversations_list(app_state.agent),
            get_current_conversation_info(app_state.agent)
        )
    
    # Инициализация агента при первом сообщении
    if not app_state.agent:
        agent, success = init_agent(app_state.selected_model)
        app_state.agent = agent
        app_state.mcp_connected = success
    
    # Получаем ответ от агента
    response = process_message(app_state.agent, message)
    
    # Синхронизируем статус MCP с реальным состоянием агента
    app_state.mcp_connected = app_state.agent.state.mcp_available if app_state.agent else False
    
    # Форматируем ответ для Gradio Chatbot (messages format)
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ]
    
    # Обновляем отладку
    debug = format_mcp_calls(app_state.agent)
    
    # Обновляем список диалогов и информацию о текущем диалоге
    conversations = get_conversations_list(app_state.agent)
    conv_info = get_current_conversation_info(app_state.agent)
    
    return (
        new_history, 
        debug, 
        True,
        get_mcp_status(),
        conversations,
        conv_info
    )


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
                run_async(app_state.agent.disconnect_mcp())
            except Exception:
                pass
        app_state.agent = None
        app_state.mcp_connected = False
        app_state.conversations_list = []
        app_state.current_conversation_id = None
    
    return f"Модель: {model_name}", [], ""


def reconnect_mcp():
    """Переподключает MCP."""
    if app_state.agent:
        try:
            run_async(app_state.agent.disconnect_mcp())
        except Exception:
            pass
    app_state.agent = None
    app_state.mcp_connected = False
    app_state.rag_available = False
    app_state.conversations_list = []
    app_state.current_conversation_id = None
    return "Инструменты переподключены. Статус обновится при следующем сообщении."


def clear_current_history():
    """Очищает историю текущего диалога."""
    if app_state.agent:
        app_state.agent.clear_history()
        # Сбрасываем состояние задачи
        from task_state import get_task_state_manager
        task_manager = get_task_state_manager()
        if app_state.current_conversation_id:
            task_manager.reset_task_state(app_state.current_conversation_id)
    return [], "История текущего диалога очищена", get_current_conversation_info(app_state.agent)


def delete_conversation(conv_id_str: str) -> str:
    """Удаляет диалог."""
    if not app_state.agent:
        return "❌ Агент не инициализирован"
    
    try:
        conv_id = int(conv_id_str)
        
        # Получаем информацию о диалоге перед удалением
        from conversation_manager import get_conversation_manager
        manager = get_conversation_manager()
        conversation = manager.get_conversation(conv_id)
        
        if not conversation:
            return "❌ Диалог не найден"
        
        # Нельзя удалить активный диалог
        if conv_id == app_state.current_conversation_id:
            return "❌ Нельзя удалить активный диалог. Переключитесь на другой диалог сначала."
        
        # Удаляем диалог
        success = manager.delete_conversation(conv_id)
        
        if success:
            # Обновляем список диалогов
            app_state.conversations_list = app_state.agent.get_all_conversations()
            return f"✅ Удален диалог: {conversation.title}"
        else:
            return "❌ Не удалось удалить диалог"
    
    except ValueError:
        return "❌ Неверный ID диалога"
    except Exception as e:
        return f"❌ Ошибка удаления: {e}"


def get_mcp_status() -> str:
    """Возвращает статус MCP и RAG."""
    if not app_state.agent:
        return "❌ Агент не инициализирован"
    
    # Получаем статус всех серверов
    servers = []
    if app_state.agent.orchestrator.sessions:
        for server_name in app_state.agent.orchestrator.sessions.keys():
            servers.append(f"✅ {server_name}")
    
    # Добавляем статус RAG
    rag_status = "✅ RAG доступен" if app_state.rag_available else "❌ RAG недоступен"
    
    result = []
    if servers:
        result.append("📡 MCP серверы:")
        result.extend(servers)
    else:
        result.append("📡 MCP серверы: ❌ Нет подключенных серверов")
    
    result.append("")
    result.append("🔍 RAG статус:")
    result.append(f"  {rag_status}")
    
    if app_state.rag_available and hasattr(app_state.agent.rag_retriever, 'get_index_stats'):
        try:
            stats = app_state.agent.rag_retriever.get_index_stats()
            if isinstance(stats, dict) and 'total_chunks' in stats:
                result.append(f"  Чанков: {stats['total_chunks']}")
                if 'total_files' in stats:
                    result.append(f"  Файлов: {stats['total_files']}")
        except Exception:
            pass
    
    return "\n".join(result)


def update_loading_indicator(processing_state: bool) -> str:
    """Обновляет индикатор загрузки."""
    if processing_state:
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


# ── Создаём Gradio интерфейс ────────────────────────────────────────────

with gr.Blocks(
    title="🗺️ Твой логист-ассистент (Enhanced)"
) as demo:
    gr.Markdown("# 🗺️ Твой логист-ассистент (Enhanced)")
    gr.Markdown("Умный планировщик маршрутов с поддержкой множества диалогов")
    
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
            
            gr.Markdown("## 💬 Управление диалогами")
            
            # Информация о текущем диалоге
            conversation_info = gr.Markdown("Загрузка информации о диалоге...")
            
            # Список диалогов
            conversations_dropdown = gr.Dropdown(
                choices=[],
                label="Все диалоги",
                interactive=True,
                allow_custom_value=False
            )
            
            with gr.Row():
                switch_conv_btn = gr.Button("↻ Переключиться", variant="secondary", scale=1)
                new_conv_btn = gr.Button("➕ Новый диалог", variant="primary", scale=1)
            
            conv_action_result = gr.Markdown("")
            
            gr.Markdown("## 📡 Статус инструментов")
            mcp_status = gr.Textbox(
                value="Загрузка статуса...",
                label="Статус MCP и RAG",
                interactive=False,
                lines=6
            )
            
            reconnect_btn = gr.Button("🔄 Переподключить MCP", variant="secondary")
            
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
            
            gr.Markdown("## 🔍 Отладка MCP")
            debug_output = gr.Textbox(
                label="Лог вызовов",
                lines=10,
                interactive=False
            )
            
            gr.Markdown("## 📜 История")
            clear_btn = gr.Button("🗑️ Очистить текущий диалог", variant="secondary")
            delete_conv_btn = gr.Button("❌ Удалить диалог", variant="stop")
        
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
            
            # Индикатор загрузки
            loading_indicator = gr.HTML(value="")
    
    
    # ── Обработчики событий ─────────────────────────────────────────────
    
    def handle_message(message: str, history: list, conv_dropdown_value: str):
        """Обрабатывает сообщение пользователя."""
        if not message.strip():
            return (
                history, 
                format_mcp_calls(app_state.agent), 
                "", 
                get_mcp_status(),
                get_conversations_list(app_state.agent),
                get_current_conversation_info(app_state.agent),
                conv_dropdown_value
            )
        
        # Инициализация агента при первом сообщении
        if not app_state.agent:
            agent, success = init_agent(app_state.selected_model)
            app_state.agent = agent
            app_state.mcp_connected = success
            
            # Обновляем статус RAG
            if app_state.agent:
                app_state.rag_available = app_state.agent.rag_retriever is not None
        
        # Получаем ответ от агента
        response = process_message(app_state.agent, message)
        
        # Синхронизируем статус MCP с реальным состоянием агента
        app_state.mcp_connected = app_state.agent.state.mcp_available if app_state.agent else False
        
        # Форматируем ответ для Gradio Chatbot
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
        
        # Обновляем отладку и статус
        debug = format_mcp_calls(app_state.agent)
        mcp_status_text = get_mcp_status()
        conversations = get_conversations_list(app_state.agent)
        conv_info = get_current_conversation_info(app_state.agent)
        
        # Обновляем значение dropdown текущего диалога
        current_conv_id = str(app_state.current_conversation_id) if app_state.current_conversation_id else ""
        conv_dropdown_value = current_conv_id
        
        return (
            new_history, 
            debug, 
            "", 
            mcp_status_text,
            conversations,
            conv_info,
            conv_dropdown_value
        )
    
    # Обработка сообщений
    msg_input.submit(
        fn=handle_message,
        inputs=[msg_input, chatbot, conversations_dropdown],
        outputs=[chatbot, debug_output, msg_input, mcp_status, conversations_dropdown, conversation_info, conversations_dropdown],
        show_progress="minimal"
    )
    
    send_btn.click(
        fn=handle_message,
        inputs=[msg_input, chatbot, conversations_dropdown],
        outputs=[chatbot, debug_output, msg_input, mcp_status, conversations_dropdown, conversation_info, conversations_dropdown],
        show_progress="minimal"
    )
    
    # Обновление модели
    model_dropdown.change(
        fn=update_model,
        inputs=[model_dropdown],
        outputs=[model_status, conversations_dropdown, conv_action_result]
    )
    
    # Переключение диалога
    def on_switch_conversation(conv_dropdown_value: str):
        if not conv_dropdown_value:
            return "❌ Выберите диалог из списка", gr.update()
        
        result = switch_conversation(app_state.agent, conv_dropdown_value)
        conversations = get_conversations_list(app_state.agent)
        conv_info = get_current_conversation_info(app_state.agent)
        
        # Очищаем историю чата при переключении диалога
        return result, [], conv_info, gr.update(value=conversations)
    
    switch_conv_btn.click(
        fn=on_switch_conversation,
        inputs=[conversations_dropdown],
        outputs=[conv_action_result, chatbot, conversation_info, conversations_dropdown]
    )
    
    # Создание нового диалога
    def on_new_conversation():
        result = create_new_conversation(app_state.agent)
        conversations = get_conversations_list(app_state.agent)
        conv_info = get_current_conversation_info(app_state.agent)
        
        # Очищаем историю чата для нового диалога
        return result, [], conv_info, gr.update(value=conversations)
    
    new_conv_btn.click(
        fn=on_new_conversation,
        outputs=[conv_action_result, chatbot, conversation_info, conversations_dropdown]
    )
    
    # Переподключение MCP
    reconnect_btn.click(
        fn=lambda: (reconnect_mcp(), get_mcp_status(), [], ""),
        outputs=[mcp_status, conversations_dropdown, conv_action_result]
    )
    
    # Очистка текущего диалога
    clear_btn.click(
        fn=clear_current_history,
        outputs=[chatbot, debug_output, conversation_info]
    )
    
    # Удаление диалога
    def on_delete_conversation(conv_dropdown_value: str):
        result = delete_conversation(conv_dropdown_value)
        conversations = get_conversations_list(app_state.agent)
        conv_info = get_current_conversation_info(app_state.agent)
        return result, gr.update(value=conversations), conv_info
    
    delete_conv_btn.click(
        fn=on_delete_conversation,
        inputs=[conversations_dropdown],
        outputs=[conv_action_result, conversations_dropdown, conversation_info]
    )
    
    # Инициализация при загрузке
    def on_load():
        """Инициализация при загрузке страницы."""
        # Инициализируем агента если еще не инициализирован
        if not app_state.agent:
            agent, success = init_agent(app_state.selected_model)
            app_state.agent = agent
            app_state.mcp_connected = success
            
            # Обновляем статус RAG
            if app_state.agent:
                app_state.rag_available = app_state.agent.rag_retriever is not None
        
        return (
            get_mcp_status(),
            get_conversations_list(app_state.agent),
            get_current_conversation_info(app_state.agent)
        )
    
    demo.load(
        fn=on_load,
        outputs=[mcp_status, conversations_dropdown, conversation_info]
    )
    
    # Обновление информации о диалоге при изменении выбора
    conversations_dropdown.change(
        fn=lambda conv_id: get_current_conversation_info(app_state.agent),
        inputs=[conversations_dropdown],
        outputs=[conversation_info]
    )


if __name__ == "__main__":
    print("Starting enhanced application...")
    print("Server will be available at http://localhost:7872")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7872,
        share=False,
        show_error=True,
        theme="soft"
    )