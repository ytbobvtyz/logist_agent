#!/usr/bin/env python3
"""
Тестирование RAG на 10 контрольных вопросах
Сравнение ответов с RAG и без RAG

Запуск: python test_rag.py
"""

import os
import sys
import json
from typing import List, Dict
from datetime import datetime

# Добавляем текущую папку в путь
sys.path.insert(0, os.path.dirname(__file__))

# Импортируем компоненты
from rag_retriever import RAGRetriever


# ============================================================
# 1. ФИКТИВНЫЙ АГЕНТ ДЛЯ ТЕСТИРОВАНИЯ (без реального API)
# ============================================================

class MockLLM:
    """Заглушка для тестирования без реального API"""
    
    def ask_without_rag(self, question: str) -> str:
        """Имитация ответа без RAG (из "знаний" модели)"""
        # Простые ответы для тестов
        responses = {
            "стоимость доставки груза 50 кг из москвы в санкт-петербург у пэк": 
                "Стоимость доставки ПЭК из Москвы в Санкт-Петербург для груза 50 кг составляет примерно 2000-3000 рублей.",
            "максимальный вес принимает сдэк": 
                "СДЭК принимает посылки до 30 кг, согласно тарифам компании.",
            "стоит отправить посылку весом 1 кг почтой россии": 
                "Почта России: отправка посылки весом 1 кг стоит около 500 рублей.",
            "логистика": 
                "Логистика — это управление материальными и информационными потоками."
        }
        
        q_lower = question.lower()
        for key, answer in responses.items():
            if key in q_lower:
                return f"💡 {answer}"
        
        return "💡 Извините, у меня нет точной информации по этому вопросу."
    
    def ask_with_rag(self, question: str, chunks: List[Dict]) -> str:
        """Имитация ответа с RAG (на основе найденных чанков)"""
        if not chunks:
            return self.ask_without_rag(question)
        
        # Берём информацию из первого найденного чанка
        best_chunk = chunks[0]
        
        # Эмуляция ответа с указанием источника
        return f"""📄 **Ответ найден в документах:**

{best_chunk['text'][:300]}

📁 *Источник: {best_chunk['filename']}*
🔍 *Релевантность: {best_chunk.get('score', 0):.2f}*"""


# ============================================================
# 2. 10 КОНТРОЛЬНЫХ ВОПРОСОВ
# ============================================================

TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "Сколько стоит доставка груза 50 кг из Москвы в Санкт-Петербург у ПЭК?",
        "expected_keywords": ["ПЭК", "50", "кг", "Москва", "Санкт-Петербург"],
        "expected_source": "pecom.txt"
    },
    {
        "id": 2,
        "question": "Какой максимальный вес принимает СДЭК для посылки?",
        "expected_keywords": ["СДЭК", "30", "кг", "максимальный"],
        "expected_source": "cdek.txt"
    },
    {
        "id": 3,
        "question": "Сколько стоит отправить посылку весом 1 кг Почтой России?",
        "expected_keywords": ["Почта", "России", "посылку", "руб"],
        "expected_source": "post_russia.txt"
    },
    {
        "id": 4,
        "question": "Какая стоимость доставки ПЭК из Москвы в Казань для груза 100 кг?",
        "expected_keywords": ["ПЭК", "Москва", "Казань", "стоимость"],
        "expected_source": "pecom.txt"
    },
    {
        "id": 5,
        "question": "Какой URL у публичного API ПЭК для расчёта стоимости?",
        "expected_keywords": ["calc.pecom.ru", "ajax.php", "API"],
        "expected_source": "pecom_api_doc.txt"
    },
    {
        "id": 6,
        "question": "Как передать вес груза в API ПЭК?",
        "expected_keywords": ["вес", "параметр", "places"],
        "expected_source": "pecom_api_doc.txt"
    },
    {
        "id": 7,
        "question": "Какой формат ответа возвращает API ПЭК?",
        "expected_keywords": ["JSON", "метод", "Авто"],
        "expected_source": "pecom_api_doc.txt"
    },
    {
        "id": 8,
        "question": "Какие обязанности у фрахтователя?",
        "expected_keywords": ["фрахтователь", "обязан", "оплатить"],
        "expected_source": "postanovlenie.txt"
    },
    {
        "id": 9,
        "question": "Что такое фрахтовщик по закону?",
        "expected_keywords": ["фрахтовщик", "перевозчик", "экспедитор"],
        "expected_source": "postanovlenie.txt"
    },
    {
        "id": 10,
        "question": "Что такое логистика?",
        "expected_keywords": ["управление", "потоками", "перевозка"],
        "expected_source": None  # Вопрос без RAG
    }
]


# ============================================================
# 3. ФУНКЦИИ ОЦЕНКИ
# ============================================================

def evaluate_answer(answer: str, expected_keywords: List[str]) -> Dict:
    """Оценивает, содержит ли ответ ожидаемые ключевые слова"""
    answer_lower = answer.lower()
    
    found = []
    missing = []
    
    for kw in expected_keywords:
        if kw.lower() in answer_lower:
            found.append(kw)
        else:
            missing.append(kw)
    
    score = len(found) / len(expected_keywords) if expected_keywords else 1.0
    
    return {
        "found_keywords": found,
        "missing_keywords": missing,
        "score": score,
        "score_percent": round(score * 100)
    }


def check_source(answer: str, expected_source: str) -> bool:
    """Проверяет, указан ли источник в ответе (для RAG)"""
    if not expected_source:
        return True
    return expected_source in answer.lower()


# ============================================================
# 4. ОСНОВНАЯ ФУНКЦИЯ ТЕСТИРОВАНИЯ
# ============================================================

def run_tests(use_real_llm: bool = False):
    """
    Запускает тестирование 10 вопросов
    
    Args:
        use_real_llm: Если True, использует реальный LLM (требуется API)
                      Если False, использует заглушку для демонстрации
    """
    
    print("="*70)
    print("🔍 ТЕСТИРОВАНИЕ RAG: 10 КОНТРОЛЬНЫХ ВОПРОСОВ")
    print("="*70)
    print(f"📅 Дата теста: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Режим LLM: {'Реальный API' if use_real_llm else 'Демо-режим (заглушка)'}")
    print("="*70)
    
    # Инициализация компонентов
    retriever = None
    try:
        retriever = RAGRetriever()
    except Exception as e:
        print(f"⚠️ Ошибка загрузки RAG Retriever: {e}")
    
    # Инициализация LLM
    if use_real_llm:
        try:
            from agent import LogistAgent
            llm = LogistAgent()
            print("✅ Реальный LLM загружен")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки LLM: {e}")
            print("   Переключаюсь в демо-режим")
            llm = MockLLM()
            use_real_llm = False
    else:
        llm = MockLLM()
    
    # Результаты тестов
    results = []
    
    for test in TEST_QUESTIONS:
        print(f"\n{'─'*70}")
        print(f"📌 ВОПРОС {test['id']}: {test['question']}")
        print(f"🎯 Ожидаемые ключевые слова: {', '.join(test['expected_keywords'])}")
        if test['expected_source']:
            print(f"📁 Ожидаемый источник: {test['expected_source']}")
        print("-"*70)
        
        # ===== БЕЗ RAG =====
        print("\n❌ РЕЖИМ БЕЗ RAG:")
        answer_without = llm.ask_without_rag(test['question'])
        eval_without = evaluate_answer(answer_without, test['expected_keywords'])
        
        print(f"   Ответ: {answer_without[:300]}...")
        print(f"   📊 Оценка: {eval_without['score_percent']}%")
        print(f"   ✅ Найдено: {eval_without['found_keywords']}")
        if eval_without['missing_keywords']:
            print(f"   ❌ Не найдено: {eval_without['missing_keywords']}")
        
        # ===== С RAG =====
        print("\n✅ РЕЖИМ С RAG:")
        
        # Поиск чанков
        chunks = []
        if retriever:
            chunks = retriever.search(test['question'], top_k=2)
            print(f"   🔍 Найдено чанков: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"      [{i+1}] {chunk['filename']} (score: {chunk.get('score', 0):.3f})")
        
        # Ответ с RAG
        if type(llm).__name__ == 'LogistAgent':
            # Реальный LLM - он сам обрабатывает RAG
            answer_with = llm.ask_with_rag(test['question'])
        else:
            # Mock LLM - нужно передать чанки
            answer_with = llm.ask_with_rag(test['question'], chunks)
        eval_with = evaluate_answer(answer_with, test['expected_keywords'])
        source_correct = check_source(answer_with, test['expected_source']) if test['expected_source'] else True
        
        print(f"   Ответ: {answer_with[:300]}...")
        print(f"   📊 Оценка: {eval_with['score_percent']}%")
        print(f"   ✅ Найдено: {eval_with['found_keywords']}")
        if eval_with['missing_keywords']:
            print(f"   ❌ Не найдено: {eval_with['missing_keywords']}")
        if test['expected_source']:
            print(f"   📁 Источник указан: {'✅' if source_correct else '❌'}")
        
        # Сохраняем результат
        results.append({
            "id": test['id'],
            "question": test['question'],
            "expected_keywords": test['expected_keywords'],
            "expected_source": test['expected_source'],
            "answer_without_rag": answer_without,
            "answer_with_rag": answer_with,
            "score_without": eval_without['score'],
            "score_with": eval_with['score'],
            "source_correct": source_correct,
            "chunks_used": len(chunks)
        })
        
        # Пауза между запросами (для реального API)
        if use_real_llm:
            import time
            time.sleep(1)
    
    # ===== ИТОГОВАЯ ТАБЛИЦА =====
    print("\n" + "="*70)
    print("📊 ИТОГОВОЕ СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*70)
    
    print(f"\n{'ID':<4} {'Вопрос (первые 35 символов)':<38} {'Без RAG':<8} {'С RAG':<8} {'Улучшение':<10}")
    print("-"*70)
    
    total_without = 0
    total_with = 0
    
    for r in results:
        question_short = r['question'][:35] + "..." if len(r['question']) > 35 else r['question']
        improvement = r['score_with'] - r['score_without']
        improvement_str = f"+{improvement*100:.0f}%" if improvement > 0 else f"{improvement*100:.0f}%"
        
        print(f"{r['id']:<4} {question_short:<38} {r['score_without']*100:>5.0f}%    {r['score_with']*100:>5.0f}%    {improvement_str:>8}")
        
        total_without += r['score_without']
        total_with += r['score_with']
    
    print("-"*70)
    avg_without = total_without / len(results)
    avg_with = total_with / len(results)
    print(f"{'СРЕДНИЙ':<4} {'':<38} {avg_without*100:>5.0f}%    {avg_with*100:>5.0f}%    {avg_with - avg_without:.0%}")
    
    # ===== ВЫВОД ПО КАЖДОМУ ВОПРОСУ =====
    print("\n" + "="*70)
    print("📝 ДЕТАЛЬНЫЙ АНАЛИЗ ПО ВОПРОСАМ")
    print("="*70)
    
    for r in results:
        print(f"\n🔹 Вопрос {r['id']}: {r['question']}")
        print(f"   Без RAG: {r['score_without']*100:.0f}%")
        print(f"   С RAG:   {r['score_with']*100:.0f}%")
        
        if r['score_with'] > r['score_without']:
            print(f"   ✅ RAG улучшил ответ на {(r['score_with'] - r['score_without'])*100:.0f}%")
        elif r['score_with'] < r['score_without']:
            print(f"   ⚠️ RAG ухудшил ответ на {(r['score_without'] - r['score_with'])*100:.0f}%")
        else:
            print(f"   ➖ RAG не повлиял на качество")
    
    # ===== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ =====
    output = {
        "test_date": datetime.now().isoformat(),
        "mode": "real_llm" if use_real_llm else "mock",
        "summary": {
            "avg_score_without_rag": round(avg_without * 100),
            "avg_score_with_rag": round(avg_with * 100),
            "improvement": round((avg_with - avg_without) * 100)
        },
        "results": results
    }
    
    with open("rag_test_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*70)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*70)
    print(f"📁 Результаты сохранены в: rag_test_results.json")
    print(f"📊 Средняя точность без RAG: {avg_without*100:.0f}%")
    print(f"📊 Средняя точность с RAG:   {avg_with*100:.0f}%")
    print(f"🚀 Улучшение от RAG: {(avg_with - avg_without)*100:.0f}%")
    
    return results


# ============================================================
# 5. ТОЧКА ВХОДА
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Тестирование RAG на 10 вопросах")
    parser.add_argument("--real", action="store_true", 
                       help="Использовать реальный LLM (требуется API)")
    
    args = parser.parse_args()
    
    # Запуск тестов
    run_tests(use_real_llm=args.real)