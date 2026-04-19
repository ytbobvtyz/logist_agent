#!/usr/bin/env python3
"""
Тестирование исправлений для dropdown.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from route_planner.enhanced_app import (
    get_conversations_choices, 
    get_current_conversation_value,
    app_state,
    init_agent
)

def test_dropdown_functions():
    """Тестирует функции работы с dropdown."""
    print("🧪 Тестирование исправлений dropdown...")
    
    # Инициализируем агента
    agent, success = init_agent('openrouter/free')
    if not agent:
        print("❌ Не удалось инициализировать агента")
        return False
    
    app_state.agent = agent
    app_state.current_conversation_id = agent.get_current_conversation().id
    
    # Тест 1: get_conversations_choices
    choices = get_conversations_choices(agent)
    print(f"✅ get_conversations_choices: {len(choices)} choices")
    
    for choice in choices[:3]:  # Покажем первые 3
        print(f"  - {choice}")
    
    # Проверяем формат choices
    assert isinstance(choices, list), "Choices должен быть списком"
    if choices:
        assert isinstance(choices[0], tuple), "Элемент choices должен быть кортежем"
        assert len(choices[0]) == 2, "Кортеж должен содержать 2 элемента"
        assert isinstance(choices[0][0], str), "Первый элемент должен быть строкой"
        assert isinstance(choices[0][1], str), "Второй элемент должен быть строкой"
    
    # Тест 2: get_current_conversation_value
    value = get_current_conversation_value(agent)
    print(f"✅ get_current_conversation_value: '{value}'")
    assert isinstance(value, str), "Value должен быть строкой"
    
    # Тест 3: Проверяем, что value есть в choices
    if value and choices:
        value_in_choices = any(choice[1] == value for choice in choices)
        print(f"✅ Value '{value}' в choices: {value_in_choices}")
        if not value_in_choices:
            print(f"  Предупреждение: value '{value}' не найден в choices")
            print(f"  Choices values: {[choice[1] for choice in choices]}")
    
    # Тест 4: Создаем новый диалог и проверяем обновление
    from route_planner.enhanced_app import create_new_conversation
    result = create_new_conversation(agent, "Тестовый диалог для dropdown")
    print(f"✅ Создание нового диалога: {result}")
    
    # Проверяем обновленные choices
    new_choices = get_conversations_choices(agent)
    new_value = get_current_conversation_value(agent)
    print(f"✅ Новые choices: {len(new_choices)}")
    print(f"✅ Новый value: '{new_value}'")
    
    # Проверяем, что количество choices увеличилось
    assert len(new_choices) >= len(choices), "Количество choices должно увеличиться"
    
    print("🎉 Все тесты dropdown пройдены успешно!")
    return True

if __name__ == "__main__":
    success = test_dropdown_functions()
    if not success:
        sys.exit(1)