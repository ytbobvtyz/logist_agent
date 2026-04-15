#!/usr/bin/env python3
"""
Тестирование логики интеллектуальной маршрутизации между RAG и MCP.
Проверяет только логику выбора инструментов без внешних зависимостей.
"""

import sys
import os

# Добавляем путь к корневой директории проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from route_planner.agent import RoutePlannerAgent


def test_routing_logic():
    """Тестирует логику выбора между RAG и MCP."""
    
    print("🧪 ТЕСТИРОВАНИЕ ЛОГИКИ МАРШРУТИЗАЦИИ RAG/MCP")
    print("=" * 60)
    
    # Создаем агента без инициализации внешних зависимостей
    agent = RoutePlannerAgent.__new__(RoutePlannerAgent)
    agent.model = "test"
    
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
        
        # Тестируем логику выбора режима
        try:
            should_use_rag = agent._should_use_rag(test_case['question'])
            
            # Определяем ожидаемый результат
            expected_rag = test_case['expected_mode'] == "RAG"
            
            # Сравниваем с фактическим результатом
            match = should_use_rag == expected_rag
            status = "✅ ПРОЙДЕН" if match else "❌ НЕ ПРОЙДЕН"
            
            actual_mode = "RAG" if should_use_rag else "MCP/KNOWLEDGE"
            
            print(f"   Логика выбрала: {'RAG' if should_use_rag else 'MCP/KNOWLEDGE'}")
            print(f"   Статус: {status}")
            
            results.append({
                "id": test_case['id'],
                "question": test_case['question'],
                "expected_mode": test_case['expected_mode'],
                "actual_mode": actual_mode,
                "match": match,
                "should_use_rag": should_use_rag
            })
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            results.append({
                "id": test_case['id'],
                "question": test_case['question'],
                "expected_mode": test_case['expected_mode'],
                "actual_mode": "ERROR",
                "match": False,
                "should_use_rag": False
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
    print(f"{'ID':<4} {'Ожидаемый':<12} {'Реальный':<15} {'Статус':<10} {'Вопрос (первые 30 символов)'}")
    print("-" * 80)
    
    for result in results:
        question_short = result['question'][:30] + "..."
        status = "✅" if result['match'] else "❌"
        print(f"{result['id']:<4} {result['expected_mode']:<12} {result['actual_mode']:<15} {status:<10} {question_short}")
    
    # Проверка логики выбора инструментов
    print("\n🔍 АНАЛИЗ ЛОГИКИ ВЫБОРА ИНСТРУМЕНТОВ:")
    print("-" * 40)
    
    for result in results:
        if result['match']:
            print(f"✅ Тест {result['id']}: логика правильно выбрала {result['actual_mode']}")
        else:
            print(f"❌ Тест {result['id']}: ожидался {result['expected_mode']}, но логика выбрала {result['actual_mode']}")
    
    return results


def main():
    """Основная функция тестирования."""
    print("🚀 ЗАПУСК ТЕСТИРОВАНИЯ ЛОГИКИ МАРШРУТИЗАЦИИ")
    print("=" * 60)
    
    # Тестируем логику маршрутизации
    routing_results = test_routing_logic()
    
    # Сохраняем результаты
    import json
    with open("routing_logic_results.json", "w", encoding="utf-8") as f:
        json.dump(routing_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Результаты сохранены в routing_logic_results.json")
    
    # Проверяем успешность тестирования
    success_rate = sum(1 for r in routing_results if r['match']) / len(routing_results)
    
    if success_rate >= 0.7:
        print("🎉 ТЕСТИРОВАНИЕ ЛОГИКИ ПРОЙДЕНО УСПЕШНО!")
        print(f"   Успешность: {success_rate*100:.1f}%")
    else:
        print("⚠️ ТЕСТИРОВАНИЕ ЛОГИКИ ТРЕБУЕТ ДОРАБОТКИ")
        print(f"   Успешность: {success_rate*100:.1f}%")


if __name__ == "__main__":
    main()