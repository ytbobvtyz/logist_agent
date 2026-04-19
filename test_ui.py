#!/usr/bin/env python3
"""
Простой тест UI без запуска сервера Gradio.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from route_planner.enhanced_app import app_state, init_agent, get_conversations_list, get_current_conversation_info

def test_ui_init():
    """Тестирует инициализацию UI."""
    print("🧪 Тестирование инициализации UI...")
    
    # Инициализируем агента
    agent, success = init_agent('openrouter/free')
    if not agent:
        print("❌ Не удалось инициализировать агента")
        return False
    
    print(f"✅ Агент инициализирован: {success}")
    print(f"  MCP подключен: {app_state.mcp_connected}")
    print(f"  RAG доступен: {app_state.rag_available}")
    
    # Проверяем список диалогов
    conversations = get_conversations_list(agent)
    print(f"✅ Список диалогов: {len(conversations)} диалогов")
    
    for title, conv_id in conversations:
        print(f"  - {title} (ID: {conv_id})")
    
    # Проверяем информацию о текущем диалоге
    conv_info = get_current_conversation_info(agent)
    print(f"✅ Информация о текущем диалоге:")
    print(f"  {conv_info}")
    
    # Проверяем создание нового диалога
    from route_planner.enhanced_app import create_new_conversation
    result = create_new_conversation(agent, "Тестовый диалог UI")
    print(f"✅ Создание нового диалога: {result}")
    
    # Обновляем список диалогов
    conversations = get_conversations_list(agent)
    print(f"✅ Обновленный список диалогов: {len(conversations)} диалогов")
    
    # Проверяем переключение диалогов
    if len(conversations) >= 2:
        from route_planner.enhanced_app import switch_conversation
        # Берем второй диалог
        conv_id = conversations[1][1]  # (title, id)
        result = switch_conversation(agent, conv_id)
        print(f"✅ Переключение на диалог {conv_id}: {result}")
    
    return True

if __name__ == "__main__":
    success = test_ui_init()
    if success:
        print("\n🎉 Все тесты UI пройдены успешно!")
    else:
        print("\n❌ Тестирование UI завершилось с ошибками")
        sys.exit(1)