#!/usr/bin/env python3
"""
Тестирование интеллектуальной маршрутизации между RAG и MCP.
Проверяет, что агент правильно выбирает инструменты для разных типов запросов.
"""

import asyncio
import sys
import os

# Добавляем путь к корневой директории проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from route_planner.agent import RoutePlannerAgent


async def test_rag_mcp_routing():
    """Тестирует интеллектуальную маршрутизацию между RAG и MCP."""
    
    print("🧪 ТЕСТИРОВАНИЕ ИНТЕЛЛЕКТУАЛЬНОЙ МАРШРУТИЗАЦИИ RAG/MCP")
    print("=" * 60)
    
    # Создаем агента
    agent = RoutePlannerAgent(model="openrouter/free")
    
    # Тестовые запросы для разных сценариев
    test_cases = [
        {
            "id": 1,
            "question": "Сколько стоит доставка груза 50 кг из Москвы в Санкт-Петербург у ПЭК?",
            "expected_mode": "RAG",  # Информационный запрос о тарифах
            "description": "Информационный запрос о тарифах ПЭК"
        },
        {
            "id": 2,
            "question": "Рассчитай расстояние между Москвой и Санкт-Петербургом",
            "expected_mode": "MCP",  # Расчетный запрос
            "description": "Расчетный запрос на расстояние"
        },
        {
            "id": 3,
            "question": "Какие обязанности у фрахтователя?",
            "expected_mode": "RAG",  # Информационный запрос из документов
            "description": "Запрос информации из юридических документов"
        },
        {
            "id": 4,
            "question": "Найди оптимальный маршрут Москва-Казань-Санкт-Петербург",
            "expected_mode": "MCP",  # Расчет маршрута
            "description": "Запрос на расчет оптимального маршрута"
        },
        {
            "id": 5,
            "question": "Что такое логистика?",
            "expected_mode": "KNOWLEDGE",  # Общий информационный запрос
            "description": "Общий информационный запрос"
        },
        {
            "id": 6,
            "question": "Какой максимальный вес принимает СДЭК для посылки?",
            "expected_mode": "RAG",  # Информация из документов
            "description": "Запрос информации о лимитах перевозчика"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📌 Тест {test_case['id']}: {test_case['description']}")
        print(f"   Вопрос: {test_case['question']}")
        print(f"   Ожидаемый режим: {test_case['expected_mode']}")
        print("-" * 40)
        
        # Очищаем историю перед каждым тестом
        agent.clear_history()
        
        # Обрабатываем запрос
        try:
            response = await agent.process_message(test_case['question'])
            
            # Анализируем ответ для определения использованного режима
            actual_mode = "UNKNOWN"
            if "📚 RAG" in response:
                actual_mode = "RAG"
            elif "🔧 MCP" in response:
                actual_mode = "MCP"
            elif "💡" in response or "(RAG ничего не нашёл" in response:
                actual_mode = "KNOWLEDGE"
            
            # Проверяем соответствие ожиданиям
            match = actual_mode == test_case['expected_mode']
            status = "✅ ПРОЙДЕН" if match else "❌ НЕ ПРОЙДЕН"
            
            print(f"   Реальный режим: {actual_mode}")
            print(f"   Статус: {status}")
            print(f"   Ответ (первые 200 символов): {response[:200]}...")
            
            results.append({
                "id": test_case['id'],
                "question": test_case['question'],
                "expected_mode": test_case['expected_mode'],
                "actual_mode": actual_mode,
                "match": match,
                "response_preview": response[:200]
            })
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            results.append({
                "id": test_case['id'],
                "question": test_case['question'],
                "expected_mode": test_case['expected_mode'],
                "actual_mode": "ERROR",
                "match": False,
                "response_preview": str(e)
            })
    
    # Выводим итоговую статистику
    print("\n" + "=" * 60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['match'])
    failed_tests = total_tests - passed_tests
    
    print(f"Всего тестов: {total_tests}")
    print(f"Пройдено: {passed_tests}")
    print(f"Не пройдено: {failed_tests}")
    print(f"Успешность: {passed_tests/total_tests*100:.1f}%")
    
    # Детальная таблица результатов
    print("\n📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print("-" * 80)
    print(f"{'ID':<4} {'Ожидаемый':<12} {'Реальный':<12} {'Статус':<10} {'Вопрос (первые 30 символов)'}")
    print("-" * 80)
    
    for result in results:
        question_short = result['question'][:30] + "..."
        status = "✅" if result['match'] else "❌"
        print(f"{result['id']:<4} {result['expected_mode']:<12} {result['actual_mode']:<12} {status:<10} {question_short}")
    
    # Проверка логики выбора инструментов
    print("\n🔍 АНАЛИЗ ЛОГИКИ ВЫБОРА ИНСТРУМЕНТОВ:")
    print("-" * 40)
    
    for result in results:
        if result['match']:
            print(f"✅ Тест {result['id']}: агент правильно выбрал {result['actual_mode']}")
        else:
            print(f"❌ Тест {result['id']}: ожидался {result['expected_mode']}, но выбран {result['actual_mode']}")
    
    return results


def test_rag_functionality():
    """Тестирует функциональность RAG retriever."""
    print("\n🧪 ТЕСТИРОВАНИЕ RAG ФУНКЦИОНАЛЬНОСТИ")
    print("=" * 40)
    
    agent = RoutePlannerAgent(model="openrouter/free")
    
    if agent.rag_retriever:
        print("✅ RAG Retriever загружен")
        
        # Тестовый поиск
        test_query = "стоимость доставки ПЭК"
        results = agent.search_with_rag(test_query, top_k=2)
        
        print(f"🔍 Поиск по запросу: '{test_query}'")
        print(f"   Найдено результатов: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"   [{i+1}] Файл: {result['filename']}")
            print(f"       Сходство: {result['score']:.3f}")
            print(f"       Текст: {result['text'][:100]}...")
    else:
        print("❌ RAG Retriever не загружен")
    
    return agent.rag_retriever is not None


async def main():
    """Основная функция тестирования."""
    print("🚀 ЗАПУСК ТЕСТИРОВАНИЯ ИНТЕГРАЦИИ RAG/MCP")
    print("=" * 60)
    
    # Тестируем RAG функциональность
    rag_ok = test_rag_functionality()
    
    if rag_ok:
        # Тестируем интеллектуальную маршрутизацию
        routing_results = await test_rag_mcp_routing()
        
        # Сохраняем результаты
        import json
        with open("rag_mcp_integration_results.json", "w", encoding="utf-8") as f:
            json.dump(routing_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Результаты сохранены в rag_mcp_integration_results.json")
        
        # Проверяем успешность тестирования
        success_rate = sum(1 for r in routing_results if r['match']) / len(routing_results)
        
        if success_rate >= 0.7:
            print("🎉 ТЕСТИРОВАНИЕ ПРОЙДЕНО УСПЕШНО!")
            print(f"   Успешность: {success_rate*100:.1f}%")
        else:
            print("⚠️ ТЕСТИРОВАНИЕ ТРЕБУЕТ ДОРАБОТКИ")
            print(f"   Успешность: {success_rate*100:.1f}%")
    else:
        print("❌ ТЕСТИРОВАНИЕ ПРЕРВАНО: RAG не работает")


if __name__ == "__main__":
    asyncio.run(main())