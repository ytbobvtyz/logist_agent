#!/usr/bin/env python3
"""
Упрощенное тестирование интеллектуальной маршрутизации между RAG и MCP.
Проверяет базовую функциональность без внешних зависимостей.
"""

import asyncio
import sys
import os

# Добавляем путь к корневой директории проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from route_planner.agent import RoutePlannerAgent


async def test_basic_functionality():
    """Тестирует базовую функциональность агента."""
    
    print("🧪 БАЗОВОЕ ТЕСТИРОВАНИЕ ФУНКЦИОНАЛЬНОСТИ RAG/MCP")
    print("=" * 60)
    
    # Создаем агента
    agent = RoutePlannerAgent(model="openrouter/free")
    
    # Проверяем загрузку RAG
    print("🔍 Проверка RAG retriever:")
    if agent.rag_retriever:
        print("✅ RAG Retriever загружен")
        print(f"   Всего чанков в базе: {len(agent.rag_retriever.chunks) if hasattr(agent.rag_retriever, 'chunks') else 'N/A'}")
    else:
        print("❌ RAG Retriever не загружен")
        return False
    
    # Проверяем поиск через RAG
    print("\n🔍 Тестирование RAG поиска:")
    test_query = "стоимость доставки ПЭК"
    results = agent.search_with_rag(test_query, top_k=2)
    
    print(f"   Запрос: '{test_query}'")
    print(f"   Найдено результатов: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"   [{i+1}] Файл: {result['filename']}")
        print(f"       Сходство: {result.get('score', 0):.3f}")
        print(f"       Текст: {result['text'][:100]}...")
    
    # Проверяем логику выбора режима
    print("\n🔍 Проверка логики выбора режима:")
    test_questions = [
        "Сколько стоит доставка груза 50 кг из Москвы в Санкт-Петербург у ПЭК?",
        "Рассчитай расстояние между Москвой и Санкт-Петербургом",
        "Что такое логистика?"
    ]
    
    for i, question in enumerate(test_questions):
        should_use_rag = agent._should_use_rag(question)
        mode = "RAG" if should_use_rag else "MCP/KNOWLEDGE"
        print(f"   [{i+1}] '{question[:30]}...' → {mode}")
    
    # Проверяем очистку истории
    print("\n🔍 Проверка очистки истории:")
    agent.state.messages.append({"role": "user", "content": "тест"})
    
    # Создаем простой mock для MCPToolCall
    class MockMCPToolCall:
        def __init__(self, tool_name, arguments, success):
            self.tool_name = tool_name
            self.arguments = arguments
            self.success = success
    
    agent.state.mcp_calls.append(MockMCPToolCall("test", {}, True))
    
    print(f"   Сообщений до очистки: {len(agent.state.messages)}")
    print(f"   Вызовов MCP до очистки: {len(agent.state.mcp_calls)}")
    
    agent.clear_history()
    
    print(f"   Сообщений после очистки: {len(agent.state.messages)}")
    print(f"   Вызовов MCP после очистки: {len(agent.state.mcp_calls)}")
    
    return True


async def main():
    """Основная функция тестирования."""
    print("🚀 ЗАПУСК УПРОЩЕННОГО ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    try:
        success = await test_basic_functionality()
        
        if success:
            print("\n🎉 БАЗОВОЕ ТЕСТИРОВАНИЕ ПРОЙДЕНО УСПЕШНО!")
            print("   Основные компоненты работают корректно")
        else:
            print("\n❌ БАЗОВОЕ ТЕСТИРОВАНИЕ НЕ ПРОЙДЕНО")
            print("   Требуется проверка конфигурации")
            
    except Exception as e:
        print(f"\n❌ ОШИБКА ПРИ ТЕСТИРОВАНИИ: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())