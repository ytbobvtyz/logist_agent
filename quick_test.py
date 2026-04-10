#!/usr/bin/env python3
"""
Быстрый тест работы приложения без веб-интерфейса.
Проверяет основные функции агента напрямую.
"""

import asyncio
import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(__file__))

from route_planner.agent import RoutePlannerAgent


async def quick_test():
    """Быстрый тест работы агента."""
    print("🚀 Быстрый тест умного ассистента логиста")
    print("=" * 50)
    
    # Создаём агента
    agent = RoutePlannerAgent()
    
    # Подключаемся к MCP
    print("🔗 Подключаемся к MCP серверу...")
    success = await agent.connect_mcp()
    
    if not success:
        print("❌ Не удалось подключиться к MCP")
        return False
    
    print("✅ MCP подключено успешно")
    print(f"📋 Доступные инструменты: {[t['function']['name'] for t in agent.mcp_tools]}")
    
    # Тест 1: Простой маршрут
    print("\n🧪 Тест 1: Простой маршрут между Москвой и Санкт-Петербургом")
    try:
        response = await agent.process_message("Найди маршрут между Москвой и Санкт-Петербургом")
        print(f"✅ Ответ агента:")
        print(response[:500] + "..." if len(response) > 500 else response)
        
        # Проверяем использование инструментов
        calls = agent.get_mcp_calls()
        print(f"📊 Использовано MCP инструментов: {len(calls)}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False
    
    # Тест 2: Несколько городов
    print("\n🧪 Тест 2: Маршрут между тремя городами")
    try:
        response = await agent.process_message("Построй маршрут между Москвой, Казанью и Нижним Новгородом")
        print(f"✅ Ответ агента:")
        print(response[:500] + "..." if len(response) > 500 else response)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False
    
    # Отключаемся от MCP
    await agent.disconnect_mcp()
    print("\n✅ Все тесты пройдены успешно!")
    print("🌐 Веб-интерфейс доступен по адресу: http://localhost:7862")
    
    return True


if __name__ == "__main__":
    result = asyncio.run(quick_test())
    exit_code = 0 if result else 1
    sys.exit(exit_code)