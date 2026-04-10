"""
Streamlit UI для умного ассистента логиста.
Чат-интерфейс с боковой панелью для настройки и отладки.
"""

import streamlit as st
import asyncio
import sys
import os

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(__file__))

from agent import RoutePlannerAgent, OPENROUTER_MODELS, create_agent


def init_session_state():
    """Инициализация состояния сессии."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "mcp_connected" not in st.session_state:
        st.session_state.mcp_connected = False
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "openrouter/auto"
    if "mcp_debug" not in st.session_state:
        st.session_state.mcp_debug = []


async def init_agent(model: str) -> tuple:
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
        st.error(f"Ошибка инициализации агента: {e}")
        return None, False


async def process_user_message(agent: RoutePlannerAgent, message: str) -> str:
    """
    Обработка сообщения пользователя.
    
    Returns:
        Ответ агента
    """
    return await agent.process_message(message)


def main():
    """Главная функция Streamlit приложения."""
    
    # Настройка страницы
    st.set_page_config(
        page_title="Логист-ассистент",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Инициализация состояния
    init_session_state()
    
    # Заголовок
    st.title("🗺️ Твой логист-ассистент")
    st.caption("Умный планировщик маршрутов между городами")
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Выбор модели
        model_options = {m["name"]: m["id"] for m in OPENROUTER_MODELS}
        selected_name = st.selectbox(
            "Выберите модель",
            options=list(model_options.keys()),
            index=0
        )
        selected_model = model_options[selected_name]
        
        # Проверка изменения модели
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.session_state.agent = None
            st.session_state.mcp_connected = False
            st.rerun()
        
        st.divider()
        
        # Статус подключения MCP
        st.header("📡 Статус MCP")
        if st.session_state.mcp_connected:
            st.success("✅ Подключено")
        else:
            st.error("❌ Не подключено")
        
        # Кнопка переподключения
        if st.button("🔄 Переподключить MCP"):
            st.session_state.agent = None
            st.session_state.mcp_connected = False
            st.rerun()
        
        st.divider()
        
        # Окошко отладки MCP
        st.header("🔍 Отладка MCP")
        if st.session_state.agent:
            mcp_calls = st.session_state.agent.get_mcp_calls()
            if mcp_calls:
                for i, call in enumerate(mcp_calls):
                    with st.expander(f"Call {i+1}: {call.tool_name}"):
                        st.json({
                            "tool": call.tool_name,
                            "arguments": call.arguments,
                            "success": call.success,
                            "result": call.result[:500] + "..." if call.result and len(call.result) > 500 else call.result,
                            "error": call.error
                        })
            else:
                st.info("Пока нет вызовов MCP")
        else:
            st.info("Агент не инициализирован")
        
        st.divider()
        
        # История сообщений
        st.header("📜 История")
        if st.session_state.messages:
            for msg in st.session_state.messages[-10:]:  # Последние 10
                role_icon = "👤" if msg["role"] == "user" else "🤖"
                st.text(f"{role_icon} {msg['content'][:50]}...")
        else:
            st.info("Нет сообщений")
        
        # Кнопка очистки истории
        if st.button("🗑️ Очистить историю"):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.clear_history()
            st.rerun()
    
    # Основная область - чат
    st.header("💬 Чат")
    
    # Отображение сообщений
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Поле ввода
    if prompt := st.chat_input("Введите сообщение..."):
        # Добавляем сообщение пользователя
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Проверяем инициализацию агента
        if not st.session_state.agent:
            with st.spinner("Инициализация агента..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                agent, success = loop.run_until_complete(
                    init_agent(st.session_state.selected_model)
                )
                st.session_state.agent = agent
                st.session_state.mcp_connected = success
        
        # Получаем ответ от агента
        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    process_user_message(st.session_state.agent, prompt)
                )
                st.markdown(response)
        
        # Добавляем ответ в историю
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Авто-скролл (rerun для обновления UI)
        st.rerun()


if __name__ == "__main__":
    main()
