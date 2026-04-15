#!/usr/bin/env python3
"""
Тестирование RAG с реальным LLM через OpenRouter
Использует модель DeepSeek v3.2 для точного сравнения эффективности

Запуск: python test_llm_rag.py
Требуется: OPENROUTER_API_KEY в .env файле
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# Добавляем текущую папку в путь
sys.path.insert(0, os.path.dirname(__file__))

# Загружаем переменные окружения
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("❌ Ошибка: OPENROUTER_API_KEY не найден в .env файле")
    print("   Добавьте OPENROUTER_API_KEY=ваш_ключ в файл .env")
    sys.exit(1)

# Импортируем RAG компоненты
from rag_retriever import RAGRetriever


# ============================================================
# 1. КОНФИГУРАЦИЯ ТЕСТИРОВАНИЯ
# ============================================================

DEEPSEEK_MODEL = "deepseek/deepseek-v3.2"
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "Сколько стоит доставка груза 50 кг из Москвы в Санкт-Петербург у ПЭК?",
        "expected_keywords": ["ПЭК", "50", "кг", "Москва", "Санкт-Петербург", "стоимость"],
        "expected_source": "pecom.txt",
        "category": "стоимость"
    },
    {
        "id": 2,
        "question": "Какой максимальный вес принимает СДЭК для посылки?",
        "expected_keywords": ["СДЭК", "30", "кг", "максимальный", "вес"],
        "expected_source": "cdek.txt",
        "category": "ограничения"
    },
    {
        "id": 3,
        "question": "Сколько стоит отправить посылку весом 1 кг Почтой России?",
        "expected_keywords": ["Почта", "России", "посылку", "руб", "стоимость"],
        "expected_source": "post_russia.txt",
        "category": "стоимость"
    },
    {
        "id": 4,
        "question": "Какая стоимость доставки ПЭК из Москвы в Казань для груза 100 кг?",
        "expected_keywords": ["ПЭК", "Москва", "Казань", "стоимость", "100", "кг"],
        "expected_source": "pecom.txt",
        "category": "стоимость"
    },
    {
        "id": 5,
        "question": "Какой URL у публичного API ПЭК для расчёта стоимости?",
        "expected_keywords": ["calc.pecom.ru", "ajax.php", "API", "URL"],
        "expected_source": "pecom_api_doc.txt",
        "category": "техническое"
    },
    {
        "id": 6,
        "question": "Как передать вес груза в API ПЭК?",
        "expected_keywords": ["вес", "параметр", "places", "weight"],
        "expected_source": "pecom_api_doc.txt",
        "category": "техническое"
    },
    {
        "id": 7,
        "question": "Какой формат ответа возвращает API ПЭК?",
        "expected_keywords": ["JSON", "метод", "Авто", "формат"],
        "expected_source": "pecom_api_doc.txt",
        "category": "техническое"
    },
    {
        "id": 8,
        "question": "Какие обязанности у фрахтователя?",
        "expected_keywords": ["фрахтователь", "обязан", "оплатить", "принять"],
        "expected_source": "postanovlenie.txt",
        "category": "юридическое"
    },
    {
        "id": 9,
        "question": "Что такое фрахтовщик по закону?",
        "expected_keywords": ["фрахтовщик", "перевозчик", "экспедитор", "закон"],
        "expected_source": "postanovlenie.txt",
        "category": "юридическое"
    },
    {
        "id": 10,
        "question": "Какие документы нужны для перевозки груза?",
        "expected_keywords": ["документы", "транспортная", "накладная", "перевозка"],
        "expected_source": "postanovlenie.txt",
        "category": "юридическое"
    }
]


# ============================================================
# 2. КЛАСС ДЛЯ РАБОТЫ С DEEPSEEK V3.2
# ============================================================

class DeepSeekAgent:
    """Агент для работы с DeepSeek v3.2 через OpenRouter."""
    
    def __init__(self, model: str = DEEPSEEK_MODEL):
        """Инициализация агента с указанной моделью."""
        self.model = model
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0
        )
        self.retriever = None
        
        # Пытаемся загрузить RAG retriever
        try:
            self.retriever = RAGRetriever()
            print(f"✅ RAG Retriever загружен для модели {model}")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить RAG Retriever: {e}")
    
    def ask_without_rag(self, question: str) -> Tuple[str, float]:
        """Запрос к LLM без использования RAG."""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты помощник-логист с глубокими знаниями в области перевозок и логистики. Отвечай точно и информативно на вопросы о перевозчиках, тарифах, документах и законодательстве."
                    },
                    {"role": "user", "content": question}
                ],
                max_tokens=1024,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content or "Не удалось получить ответ"
            response_time = time.time() - start_time
            
            return answer, response_time
            
        except Exception as e:
            return f"❌ Ошибка LLM: {e}", time.time() - start_time
    
    def ask_with_rag(self, question: str) -> Tuple[str, float, List[Dict]]:
        """Запрос к LLM с использованием RAG."""
        start_time = time.time()
        
        # Поиск релевантных чанков
        chunks = []
        if self.retriever:
            chunks = self.retriever.search(question, top_k=3)
        
        # Построение промпта с RAG
        prompt = self._build_rag_prompt(question, chunks)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content or "Не удалось получить ответ"
            response_time = time.time() - start_time
            
            return answer, response_time, chunks
            
        except Exception as e:
            return f"❌ Ошибка LLM с RAG: {e}", time.time() - start_time, chunks
    
    def _build_rag_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Построение промпта с RAG контекстом."""
        if not chunks:
            return f"""Ты помощник-логист. Ответь на вопрос:

{query}

💡 Используй свои знания, так как релевантная информация в документах не найдена."""
        
        context = "\n".join([
            f"📄 [{chunk['filename']}] Релевантность: {chunk['score']:.2f}\n{chunk['text']}"
            for chunk in chunks
        ])
        
        return f"""Ты помощник-логист.

## Инструкция:
- Если информация есть в документах ниже — используй её и укажи источник
- Если информации нет — используй свои знания
- Всегда указывай источник информации
- Будь максимально точным и информативным

## Релевантные документы:
{context}

## Вопрос пользователя:
{query}

## Твой ответ:
"""


# ============================================================
# 3. ФУНКЦИИ ОЦЕНКИ И АНАЛИЗА
# ============================================================

def extract_root(word: str) -> str:
    """Извлекает корень слова для русского языка."""
    # Простая эвристика для извлечения корней русских слов
    # Убираем распространенные окончания
    endings = ['тель', 'щик', 'ник', 'ец', 'ок', 'ек', 'ик', 'ка', 'ко', 'ая', 'ий', 'ый', 'ой']
    
    # Для длинных слов пробуем найти корень
    if len(word) > 4:
        for ending in endings:
            if word.endswith(ending):
                return word[:-len(ending)]
    
    # Если слово короткое или не найдено окончание, возвращаем как есть
    return word

def evaluate_answer(answer: str, expected_keywords: List[str]) -> Dict:
    """Оценивает ответ по наличию ключевых слов с учетом корней."""
    answer_lower = answer.lower()
    
    found = []
    missing = []
    
    for kw in expected_keywords:
        kw_lower = kw.lower()
        kw_root = extract_root(kw_lower)
        
        # Проверяем полное слово и корень
        if kw_lower in answer_lower or kw_root in answer_lower:
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


def check_source_relevance(answer: str, expected_source: str) -> Dict:
    """Проверяет релевантность источника в ответе."""
    if not expected_source:
        return {"source_found": True, "source_correct": True}
    
    answer_lower = answer.lower()
    source_found = expected_source.lower() in answer_lower
    
    # Проверяем наличие любых указаний на источник
    source_indicators = ["источник", "документ", "файл", "📄", "📁"]
    has_source_indication = any(indicator in answer_lower for indicator in source_indicators)
    
    return {
        "source_found": source_found,
        "source_correct": source_found,
        "has_source_indication": has_source_indication
    }


def analyze_response_quality(answer: str) -> Dict:
    """Анализирует качество ответа."""
    quality_metrics = {
        "length": len(answer),
        "has_details": len(answer) > 100,  # Подробный ответ
        "has_structure": any(marker in answer for marker in ["•", "-", "1.", "2."]),  # Структурированный
        "has_sources": any(marker in answer.lower() for marker in ["источник", "документ", "📄"])
    }
    
    return quality_metrics


# ============================================================
# 4. ОСНОВНАЯ ФУНКЦИЯ ТЕСТИРОВАНИЯ
# ============================================================

def run_deepseek_rag_test():
    """Запускает тестирование RAG с DeepSeek v3.2."""
    
    print("="*80)
    print("🔍 ТЕСТИРОВАНИЕ RAG С DEEPSEEK V3.2")
    print("="*80)
    print(f"📅 Дата теста: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Модель: {DEEPSEEK_MODEL}")
    print(f"📊 Вопросов для тестирования: {len(TEST_QUESTIONS)}")
    print("="*80)
    
    # Инициализация агента
    agent = DeepSeekAgent()
    
    # Результаты тестирования
    results = []
    total_time_without_rag = 0
    total_time_with_rag = 0
    
    for i, test in enumerate(TEST_QUESTIONS):
        print(f"\n{'─'*80}")
        print(f"📌 ВОПРОС {test['id']}/{len(TEST_QUESTIONS)}: {test['question']}")
        print(f"🎯 Категория: {test['category']}")
        print(f"🎯 Ожидаемые ключевые слова: {', '.join(test['expected_keywords'])}")
        if test['expected_source']:
            print(f"📁 Ожидаемый источник: {test['expected_source']}")
        print("-"*80)
        
        # === БЕЗ RAG ===
        print("\n❌ РЕЖИМ БЕЗ RAG:")
        answer_without, time_without = agent.ask_without_rag(test['question'])
        total_time_without_rag += time_without
        
        eval_without = evaluate_answer(answer_without, test['expected_keywords'])
        quality_without = analyze_response_quality(answer_without)
        
        print(f"   ⏱️  Время ответа: {time_without:.2f} сек")
        print(f"   📊 Оценка: {eval_without['score_percent']}%")
        print(f"   ✅ Найдено: {eval_without['found_keywords']}")
        if eval_without['missing_keywords']:
            print(f"   ❌ Не найдено: {eval_without['missing_keywords']}")
        print(f"   📝 Длина ответа: {quality_without['length']} символов")
        print(f"   Ответ: {answer_without[:200]}...")
        
        # Пауза между запросами
        time.sleep(2)
        
        # === С RAG ===
        print("\n✅ РЕЖИМ С RAG:")
        answer_with, time_with, chunks = agent.ask_with_rag(test['question'])
        total_time_with_rag += time_with
        
        eval_with = evaluate_answer(answer_with, test['expected_keywords'])
        quality_with = analyze_response_quality(answer_with)
        source_check = check_source_relevance(answer_with, test['expected_source'])
        
        print(f"   ⏱️  Время ответа: {time_with:.2f} сек")
        print(f"   🔍 Найдено чанков: {len(chunks)}")
        for j, chunk in enumerate(chunks):
            print(f"      [{j+1}] {chunk['filename']} (score: {chunk.get('score', 0):.3f})")
        
        print(f"   📊 Оценка: {eval_with['score_percent']}%")
        print(f"   ✅ Найдено: {eval_with['found_keywords']}")
        if eval_with['missing_keywords']:
            print(f"   ❌ Не найдено: {eval_with['missing_keywords']}")
        
        if test['expected_source']:
            print(f"   📁 Источник указан: {'✅' if source_check['source_correct'] else '❌'}")
            print(f"   📄 Указание на источник: {'✅' if source_check['has_source_indication'] else '❌'}")
        
        print(f"   📝 Длина ответа: {quality_with['length']} символов")
        print(f"   Ответ: {answer_with[:200]}...")
        
        # Сохраняем результат
        results.append({
            "id": test['id'],
            "question": test['question'],
            "category": test['category'],
            "expected_keywords": test['expected_keywords'],
            "expected_source": test['expected_source'],
            "answer_without_rag": answer_without,
            "answer_with_rag": answer_with,
            "time_without_rag": time_without,
            "time_with_rag": time_with,
            "score_without": eval_without['score'],
            "score_with": eval_with['score'],
            "quality_without": quality_without,
            "quality_with": quality_with,
            "source_check": source_check,
            "chunks_used": len(chunks),
            "chunks_details": chunks
        })
        
        # Пауза между вопросами
        time.sleep(3)
    
    # ===== ИТОГОВАЯ СТАТИСТИКА =====
    print("\n" + "="*80)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("="*80)
    
    # Основные метрики
    avg_score_without = sum(r['score_without'] for r in results) / len(results)
    avg_score_with = sum(r['score_with'] for r in results) / len(results)
    improvement = avg_score_with - avg_score_without
    
    avg_time_without = total_time_without_rag / len(results)
    avg_time_with = total_time_with_rag / len(results)
    time_increase = avg_time_with - avg_time_without
    
    print(f"\n📈 ОСНОВНЫЕ МЕТРИКИ:")
    print(f"   Средняя точность без RAG: {avg_score_without*100:.1f}%")
    print(f"   Средняя точность с RAG:   {avg_score_with*100:.1f}%")
    print(f"   Улучшение точности:       {improvement*100:+.1f}%")
    print(f"\n⏱️  ПРОИЗВОДИТЕЛЬНОСТЬ:")
    print(f"   Среднее время без RAG: {avg_time_without:.2f} сек")
    print(f"   Среднее время с RAG:   {avg_time_with:.2f} сек")
    print(f"   Изменение времени:     {time_increase:+.2f} сек")
    
    # Расчет процентного изменения времени
    if avg_time_without > 0:
        time_change_pct = ((avg_time_without - avg_time_with) / avg_time_without) * 100
        print(f"   Изменение времени:     {time_change_pct:+.1f}%")
    
    # Анализ эффективности RAG
    print(f"\n🔍 АНАЛИЗ ЭФФЕКТИВНОСТИ RAG:")
    
    # Определяем тип эффективности
    if improvement > 0 and time_increase < 0:
        # RAG быстрее и точнее
        print("   ✅ RAG БЫСТРЕЕ И ТОЧНЕЕ")
        print("      RAG демонстрирует оптимальную эффективность - улучшает точность ответов")
        print("      и сокращает время ответа. Рекомендуется к использованию.")
    elif improvement > 0 and time_increase >= 0:
        # RAG точнее, но медленнее
        print("   ⚖️  RAG ТОЧНЕЕ, НО МЕДЛЕННЕЕ")
        print("      RAG улучшает качество ответов, но требует больше времени.")
        print("      Решение о использовании зависит от приоритетов: точность vs скорость.")
    elif improvement <= 0 and time_increase < 0:
        # RAG быстрее, но менее точен
        print("   ⚡ RAG БЫСТРЕЕ, НО МЕНЕЕ ТОЧЕН")
        print("      RAG ускоряет ответы, но может снижать точность.")
        print("      Подходит для сценариев, где скорость важнее абсолютной точности.")
    else:
        # RAG медленнее и менее точен
        print("   ❌ RAG МЕДЛЕННЕЕ И МЕНЕЕ ТОЧЕН")
        print("      RAG ухудшает обе метрики. Требуется оптимизация системы.")
    
    # Дополнительная аналитика
    print(f"\n📊 ДОПОЛНИТЕЛЬНАЯ АНАЛИТИКА:")
    improved_questions = sum(1 for r in results if r['score_with'] > r['score_without'])
    faster_questions = sum(1 for r in results if r['time_with_rag'] < r['time_without_rag'])
    
    print(f"   Вопросов с улучшением точности: {improved_questions}/{len(results)} ({improved_questions/len(results)*100:.1f}%)")
    print(f"   Вопросов с ускорением ответа:   {faster_questions}/{len(results)} ({faster_questions/len(results)*100:.1f}%)")
    
    # Анализ по порогу улучшения
    significant_improvement = sum(1 for r in results if r['score_with'] - r['score_without'] > 0.2)
    print(f"   Значительное улучшение (>20%):  {significant_improvement}/{len(results)} вопросов")
    
    # Анализ по категориям
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    print(f"\n📂 АНАЛИЗ ПО КАТЕГОРИЯМ:")
    for cat, cat_results in categories.items():
        cat_score_without = sum(r['score_without'] for r in cat_results) / len(cat_results)
        cat_score_with = sum(r['score_with'] for r in cat_results) / len(cat_results)
        print(f"   {cat}: {cat_score_without*100:.1f}% → {cat_score_with*100:.1f}% ({cat_score_with - cat_score_without:+.1f}%)")
    
    # Детальная таблица с временными метриками
    print(f"\n{'ID':<4} {'Категория':<12} {'Без RAG':<8} {'С RAG':<8} {'Улучш.':<8} {'Время RAG':<10} {'Ускорение':<10}")
    print("-"*80)
    
    for r in results:
        improvement_pct = (r['score_with'] - r['score_without']) * 100
        improvement_str = f"{improvement_pct:+.1f}%"
        
        # Расчет ускорения/замедления в процентах
        if r['time_without_rag'] > 0:
            time_change_pct = ((r['time_without_rag'] - r['time_with_rag']) / r['time_without_rag']) * 100
            time_change_str = f"{time_change_pct:+.1f}%"
        else:
            time_change_str = "N/A"
        
        print(f"{r['id']:<4} {r['category']:<12} {r['score_without']*100:>5.1f}%   {r['score_with']*100:>5.1f}%   {improvement_str:>7}   {r['time_with_rag']:>7.1f}с   {time_change_str:>9}")
    
    # ===== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ =====
    output = {
        "test_date": datetime.now().isoformat(),
        "model": DEEPSEEK_MODEL,
        "summary": {
            "questions_count": len(results),
            "avg_score_without_rag": round(avg_score_without * 100, 1),
            "avg_score_with_rag": round(avg_score_with * 100, 1),
            "improvement": round(improvement * 100, 1),
            "avg_time_without_rag": round(avg_time_without, 2),
            "avg_time_with_rag": round(avg_time_with, 2),
            "time_increase": round(time_increase, 2)
        },
        "category_analysis": {
            cat: {
                "count": len(cat_results),
                "avg_score_without": sum(r['score_without'] for r in cat_results) / len(cat_results),
                "avg_score_with": sum(r['score_with'] for r in cat_results) / len(cat_results)
            }
            for cat, cat_results in categories.items()
        },
        "results": results
    }
    
    output_filename = f"deepseek_rag_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*80)
    print(f"📁 Результаты сохранены в: {output_filename}")
    print(f"📊 Средняя точность без RAG: {avg_score_without*100:.1f}%")
    print(f"📊 Средняя точность с RAG:   {avg_score_with*100:.1f}%")
    print(f"🚀 Улучшение от RAG: {improvement*100:+.1f}%")
    print(f"⏱️  Время тестирования: {total_time_without_rag + total_time_with_rag:.1f} сек")
    
    return output


# ============================================================
# 5. ТОЧКА ВХОДА
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Тестирование RAG с DeepSeek v3.2")
    parser.add_argument("--quick", action="store_true", 
                       help="Быстрый тест (только 3 вопроса)")
    
    args = parser.parse_args()
    
    # Настройка теста
    if args.quick:
        TEST_QUESTIONS = TEST_QUESTIONS[:3]
        print("⚡ Быстрый тест: 3 вопроса")
    
    # Запуск тестирования
    try:
        results = run_deepseek_rag_test()
    except KeyboardInterrupt:
        print("\n\n❌ Тестирование прервано пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()