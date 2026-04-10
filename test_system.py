#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы умного ассистента логиста.
Проверяет все компоненты системы: MCP сервер, агент и API.
"""

import asyncio
import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(__file__))

from route_planner.agent import RoutePlannerAgent


async def test_geocoding():
    """Тестирование геокодирования городов."""
    print("🧪 Тестирование геокодирования...")
    
    agent = RoutePlannerAgent()
    await agent.connect_mcp()
    
    # Тестовые города
    test_cities = ["Москва", "Санкт-Петербург", "Казань"]
    
    try:
        # Вызываем инструмент геокодирования напрямую
        result = await agent.call_mcp_tool("geocode_batch", {"cities": test_cities})
        print(f"✅ Геокодирование успешно: {result[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Ошибка геокодирования: {e}")
        return False
    finally:
        await agent.disconnect_mcp()


async def test_route_calculation():
    """Тестирование расчёта маршрута."""
    print("\n🧪 Тестирование расчёта маршрута...")
    
    agent = RoutePlannerAgent()
    await agent.connect_mcp()
    
    # Тестовые координаты
    test_coordinates = {
        "cities": [
            {"name": "Москва", "lat": 55.7558, "lon": 37.6173},
            {"name": "Санкт-Петербург", "lat": 59.9311, "lon": 30.3609},
            {"name": "Казань", "lat": 55.7961, "lon": 49.1064}
        ]
    }
    
    try:
        # Вызываем инструмент расчёта маршрута
        result = await agent.call_mcp_tool("find_optimal_route", {"coordinates_json": str(test_coordinates)})
        print(f"✅ Расчёт маршрута успешен: {result[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Ошибка расчёта маршрута: {e}")
        return False
    finally:
        await agent.disconnect_mcp()


async def test_full_pipeline():
    """Тестирование полного пайплайна."""
    print("\n🧪 Тестирование полного пайплайна...")
    
    agent = RoutePlannerAgent()
    await agent.connect_mcp()
    
    try:
        # Полный запрос через агента
        response = await agent.process_message("Найди маршрут между Москвой, Санкт-Петербургом и Казанью")
        print(f"✅ Полный пайплайн успешен:")
        print(f"   Ответ: {response[:300]}...")
        
        # Проверяем использование MCP инструментов
        calls = agent.get_mcp_calls()
        print(f"   Использовано инструментов: {len(calls)}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка полного пайплайна: {e}")
        return False
    finally:
        await agent.disconnect_mcp()


async def test_error_handling():
    """Тестирование обработки ошибок."""
    print("\n🧪 Тестирование обработки ошибок...")
    
    agent = RoutePlannerAgent()
    await agent.connect_mcp()
    
    # Тест 1: Недостаточно городов
    try:
        response = await agent.process_message("Маршрут")
        print(f"✅ Обработка недостатка городов: {response[:100]}...")
    except Exception as e:
        print(f"❌ Ошибка обработки недостатка городов: {e}")
    
    # Тест 2: Несуществующий город
    try:
        response = await agent.process_message("Найди маршрут между Москвой и НесуществующийГород")
        print(f"✅ Обработка несуществующего города: {response[:100]}...")
    except Exception as e:
        print(f"❌ Ошибка обработки несуществующего города: {e}")
    
    await agent.disconnect_mcp()
    return True


async def main():
    """Главная функция тестирования."""
    print("🚀 Запуск тестов умного ассистента логиста")
    print("=" * 50)
    
    tests = [
        test_geocoding,
        test_route_calculation,
        test_full_pipeline,
        test_error_handling
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 Результаты тестирования:")
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"✅ Успешных тестов: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 Все тесты пройдены успешно!")
        return 0
    else:
        print("⚠️ Некоторые тесты не пройдены")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)