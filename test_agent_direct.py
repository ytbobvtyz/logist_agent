#!/usr/bin/env python3
"""
Тестирование агента напрямую без веб-интерфейса
"""

import asyncio
import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(__file__))

from route_planner.agent import RoutePlannerAgent

async def test_agent():
    """Тестирует агента напрямую"""
    
    try:
        print("🔄 Инициализация агента...")
        agent = RoutePlannerAgent()
        
        print("🔄 Подключение к MCP...")
        success = await agent.connect_mcp()
        
        if not success:
            print("❌ Не удалось подключиться к MCP")
            return
        
        print("✅ Агент инициализирован и подключен к MCP")
        
        # Тестовый запрос
        message = "Найди маршрут между Москвой и Санкт-Петербургом"
        print(f"📤 Отправка запроса: {message}")
        
        response = await agent.process_message(message)
        print(f"📥 Получен ответ:")
        print(f"{response}")
        
        # Проверяем вызовы MCP
        calls = agent.get_mcp_calls()
        print(f"\n🔧 Вызовы MCP: {len(calls)}")
        for i, call in enumerate(calls):
            print(f"[{i+1}] {call.tool_name}: {call.success}")
            if call.error:
                print(f"   Ошибка: {call.error}")
        
        await agent.disconnect_mcp()
        print("✅ Тест завершен")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent())