#!/usr/bin/env python3
"""
Быстрый тест работы приложения без веб-интерфейса.
Проверяет основные функции агента напрямую.
Поддерживает интерактивный режим для тестирования любых запросов.
"""

import asyncio
import sys
import os
import argparse

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(__file__))

from route_planner.agent import RoutePlannerAgent


async def test_agent_with_message(agent: RoutePlannerAgent, message: str) -> bool:
    """Тестирует агента с конкретным сообщением."""
    try:
        response = await agent.process_message(message)
        print(f"\n🧪 Запрос: {message}")
        print(f"✅ Ответ агента:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        # Проверяем использование инструментов
        calls = agent.get_mcp_calls()
        print(f"📊 Использовано MCP инструментов: {len(calls)}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


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
    
    # Проверяем аргументы командной строки
    parser = argparse.ArgumentParser(description='Тестирование логист-ассистента')
    parser.add_argument('--message', '-m', type=str, 
                       help='Сообщение для тестирования (например: "Найди маршрут между Москвой и Санкт-Петербургом")')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Интерактивный режим')
    
    args = parser.parse_args()
    
    if args.message:
        # Тестируем конкретное сообщение
        success = await test_agent_with_message(agent, args.message)
        
    elif args.interactive:
        # Интерактивный режим
        print("\n💬 Интерактивный режим. Введите сообщения для тестирования.")
        print("Для выхода введите 'quit' или 'exit'")
        print("=" * 50)
        
        while True:
            try:
                message = input("\n💭 Ваш запрос: ").strip()
                
                if message.lower() in ['quit', 'exit', 'выход']:
                    break
                
                if not message:
                    continue
                
                await test_agent_with_message(agent, message)
                
            except KeyboardInterrupt:
                print("\n👋 Завершение работы...")
                break
            except EOFError:
                print("\n👋 Завершение работы...")
                break
    else:
        # Стандартные тесты
        test_messages = [
            "Найди маршрут между Москвой и Санкт-Петербургом",
            "Построй маршрут между Москвой, Казанью и Нижним Новгородом",
            "Найди лучший путь между Парижем и Берлином"
        ]
        
        for message in test_messages:
            success = await test_agent_with_message(agent, message)
            if not success:
                break
    
    # Отключаемся от MCP
    await agent.disconnect_mcp()
    print("\n✅ Тестирование завершено!")
    print("🌐 Веб-интерфейс доступен по адресу: http://localhost:7862")
    
    return True


if __name__ == "__main__":
    result = asyncio.run(quick_test())
    exit_code = 0 if result else 1
    sys.exit(exit_code)