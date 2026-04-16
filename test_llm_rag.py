#!/usr/bin/env python3
"""
Тестирование RAG с реранкингом и фильтрацией (Day 23)
Сравнивает качество: без фильтра → с фильтром → с реранкингом

Запуск: python test_llm_rag.py --rerank
Требуется: OPENROUTER_API_KEY в .env файле
"""

import os
import sys
import json
import time
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("❌ Ошибка: OPENROUTER_API_KEY не найден")
    sys.exit(1)

from rag_retriever import RAGRetriever


# ============================================================
# 1. КОНФИГУРАЦИЯ
# ============================================================

MODEL = "deepseek/deepseek-v3.2"

# Параметры реранкинга
RERANK_CONFIG = {
    "similarity_threshold": 0.3,      # Порог отсечения
    "top_k_before": 10,                # До фильтрации
    "top_k_after": 3,                  # После фильтрации
    "keyword_boost": 0.3,              # Вес ключевых слов
    "enable_rewrite": True,            # Включить query rewrite
}

TEST_QUESTIONS = [
    {"id": 1, "question": "Сколько стоит доставка груза 50 кг из Москвы в Санкт-Петербург у ПЭК?",
     "expected_keywords": ["ПЭК", "50", "кг", "Москва", "Санкт-Петербург", "стоимость"],
     "expected_source": "pecom.txt", "category": "стоимость"},
    {"id": 2, "question": "Какие есть логистические аспекты функционирования транспорта?",
     "expected_keywords": ["логист", "транспорт", "погруз", "услуг", "функц"],
     "expected_source": "transportnaya_logistika-titov_ba.pdf", "category": "образование"},
    {"id": 3, "question": "Расскажи про информационное обеспечение логистики?",
     "expected_keywords": ["информ", "поток", "система", "ЛИС", "транспорт"],
     "expected_source": "transportnaya_logistika-titov_ba.pdf", "category": "стоимость"},
    {"id": 4, "question": "Какая стоимость доставки ПЭК из Москвы в Казань для груза 100 кг?",
     "expected_keywords": ["ПЭК", "Москва", "Казань", "стоимость", "100", "кг"],
     "expected_source": "pecom.txt", "category": "стоимость"},
    {"id": 5, "question": "Какой URL у публичного API ПЭК для расчёта стоимости?",
     "expected_keywords": ["calc.pecom.ru", "ajax.php", "API", "URL"],
     "expected_source": "pecom_api_doc.txt", "category": "техническое"},
    {"id": 6, "question": "Как передать вес груза в API ПЭК?",
     "expected_keywords": ["вес", "параметр", "places", "weight"],
     "expected_source": "pecom_api_doc.txt", "category": "техническое"},
    {"id": 7, "question": "Какой формат ответа возвращает API ПЭК?",
     "expected_keywords": ["JSON", "метод", "Авто", "формат"],
     "expected_source": "pecom_api_doc.txt", "category": "техническое"},
    {"id": 8, "question": "Какие обязанности у фрахтователя?",
     "expected_keywords": ["фрахтователь", "обязан", "оплатить", "принять"],
     "expected_source": "postanovlenie.txt", "category": "юридическое"}

]


# ============================================================
# 2. РЕРАНКИНГ И ФИЛЬТРАЦИЯ
# ============================================================

class Reranker:
    """Реранкинг и фильтрация результатов поиска"""
    
    def __init__(self, config: Dict):
        self.threshold = config.get("similarity_threshold", 0.3)
        self.keyword_boost = config.get("keyword_boost", 0.3)
        self.enable_rewrite = config.get("enable_rewrite", True)
        
        # Стоп-слова для улучшения реранкинга
        self.stop_words = {'и', 'в', 'на', 'с', 'по', 'к', 'у', 'о', 'об', 'от', 'до', 
                          'за', 'под', 'над', 'без', 'для', 'не', 'ни', 'что', 'как'}
    
    def filter_by_threshold(self, chunks: List[Dict]) -> List[Dict]:
        """Фильтрация по порогу релевантности"""
        before = len(chunks)
        filtered = [c for c in chunks if c.get('score', 0) >= self.threshold]
        print(f"      📊 Фильтрация: {before} → {len(filtered)} чанков (порог={self.threshold})")
        return filtered
    
    def rerank_by_keywords(self, chunks: List[Dict], query: str) -> List[Dict]:
        """Реранкинг на основе ключевых слов из запроса"""
        if not chunks:
            return chunks
        
        keywords = self._extract_keywords(query)
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '').lower()
            
            # Считаем пересечение ключевых слов
            matched = sum(1 for kw in keywords if kw in chunk_text)
            keyword_score = matched / len(keywords) if keywords else 0
            
            # Итоговый скор = FAISS скор + бонус за ключевые слова
            chunk['keyword_score'] = keyword_score
            chunk['final_score'] = chunk.get('score', 0) + keyword_score * self.keyword_boost
            chunk['score'] = chunk['final_score']
        
        return sorted(chunks, key=lambda x: x['score'], reverse=True)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Извлекает значимые ключевые слова из запроса"""
        words = query.lower().split()
        # Фильтруем стоп-слова и короткие слова
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        # Добавляем расширения для ключевых слов
        expansions = {
            'стоимость': ['стоимость', 'цена', 'тариф', 'руб'],
            'доставка': ['доставка', 'перевозка', 'отправка'],
            'пэк': ['пэк', 'печ', 'первая экспедиционная'],
            'сдэк': ['сдэк', 'cdek'],
        }
        
        expanded = []
        for kw in keywords:
            expanded.append(kw)
            if kw in expansions:
                expanded.extend(expansions[kw])
        
        return list(set(expanded))
    
    def rewrite_query(self, query: str, original_chunks: List[Dict]) -> str:
        """Расширяет запрос, если мало результатов"""
        if not self.enable_rewrite:
            return query
        
        if len(original_chunks) >= 2:
            return query
        
        # Словарь синонимов
        synonyms = {
            'стоимость': ['цена', 'тариф', 'руб', 'сколько'],
            'доставка': ['перевозка', 'отправка', 'транспортировка'],
            'груз': ['посылка', 'отправление', 'товар'],
            'вес': ['масса', 'килограмм', 'кг'],
            'правила': ['условия', 'требования', 'нормы'],
        }
        
        words = query.lower().split()
        new_words = words.copy()
        
        for word in words:
            if word in synonyms:
                new_words.extend(synonyms[word])
        
        rewritten = ' '.join(list(dict.fromkeys(new_words))[:15])
        
        if rewritten != query:
            print(f"      🔄 Query Rewrite: '{query}' → '{rewritten}'")
        
        return rewritten
    
    def process(self, chunks: List[Dict], query: str) -> Tuple[List[Dict], str]:
        """Полный пайплайн реранкинга"""
        # Шаг 1: Фильтрация
        filtered = self.filter_by_threshold(chunks)
        
        # Шаг 2: Реранкинг по ключевым словам
        reranked = self.rerank_by_keywords(filtered, query)
        
        # Шаг 3: Query Rewrite при необходимости
        final_query = query
        if len(reranked) < 2:
            final_query = self.rewrite_query(query, chunks)
        
        return reranked[:3], final_query


# ============================================================
# 3. АГЕНТ С РЕРАНКИНГОМ
# ============================================================

class RAGAgentWithRerank:
    def __init__(self, model: str = MODEL):
        self.model = model
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0
        )
        self.retriever = None
        self.reranker = Reranker(RERANK_CONFIG)
        
        try:
            self.retriever = RAGRetriever()
            print(f"✅ RAG Retriever загружен")
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
    
    def ask_without_filter(self, question: str) -> Tuple[str, float, List[Dict]]:
        """Без фильтрации (простой поиск)"""
        start_time = time.time()
        chunks = self.retriever.search(question, top_k=5) if self.retriever else []
        prompt = self._build_prompt(question, chunks, mode="without_filter")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1
        )
        return response.choices[0].message.content, time.time() - start_time, chunks
    
    def ask_with_filter(self, question: str) -> Tuple[str, float, List[Dict], str]:
        """С фильтрацией и реранкингом"""
        start_time = time.time()
        
        # Первичный поиск (больше чанков)
        raw_chunks = self.retriever.search(question, top_k=RERANK_CONFIG["top_k_before"]) if self.retriever else []
        
        # Реранкинг
        filtered_chunks, final_query = self.reranker.process(raw_chunks, question)
        
        # Финальный промпт
        prompt = self._build_prompt(final_query, filtered_chunks, mode="with_filter")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1
        )
        return response.choices[0].message.content, time.time() - start_time, filtered_chunks, final_query
    
    def _build_prompt(self, query: str, chunks: List[Dict], mode: str) -> str:
        if not chunks:
            return f"Ты помощник-логист. Ответь на вопрос, используя свои знания.\n\nВопрос: {query}"
        
        context = "\n".join([
            f"📄 [{c['filename']}] (релевантность: {c.get('score', 0):.3f})\n{c['text']}"
            for c in chunks
        ])
        
        mode_desc = "Используй ТОЛЬКО информацию из документов" if mode == "with_filter" else "Используй документы как дополнительный контекст"
        
        return f"""Ты помощник-логист.

## Инструкция:
- {mode_desc}
- Если информации нет в документах — честно скажи об этом
- Всегда указывай источник (имя файла)
- Будь точным и информативным

## Документы:
{context}

## Вопрос:
{query}

## Ответ:"""


# ============================================================
# 4. ОЦЕНКА КАЧЕСТВА
# ============================================================

def evaluate_answer(answer: str, expected_keywords: List[str]) -> Dict:
    answer_lower = answer.lower()
    found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    score = len(found) / len(expected_keywords) if expected_keywords else 1.0
    return {"found": found, "missing": missing, "score": score, "percent": round(score * 100)}


# ============================================================
# 5. ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def run_comparison():
    print("="*80)
    print("🔍 ДЕНЬ 23: СРАВНЕНИЕ RAG С ФИЛЬТРАЦИЕЙ И БЕЗ")
    print("="*80)
    print(f"📅 Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Модель: {MODEL}")
    print(f"⚙️  Параметры: порог={RERANK_CONFIG['similarity_threshold']}, "
          f"top_k={RERANK_CONFIG['top_k_before']}→{RERANK_CONFIG['top_k_after']}")
    print("="*80)
    
    agent = RAGAgentWithRerank()
    
    results = []
    total_score_without = 0
    total_score_with = 0
    
    for test in TEST_QUESTIONS:
        print(f"\n{'─'*80}")
        print(f"📌 {test['question']}")
        
        # Без фильтрации
        answer_without, time_without, chunks_without = agent.ask_without_filter(test['question'])
        score_without = evaluate_answer(answer_without, test['expected_keywords'])
        total_score_without += score_without['score']
        
        print(f"\n  ❌ БЕЗ ФИЛЬТРАЦИИ ({time_without:.2f}с, чанков: {len(chunks_without)})")
        print(f"     Точность: {score_without['percent']}% | Найдено: {score_without['found']}")
        print(f"     Ответ: {answer_without[:150]}...")
        
        # С фильтрацией
        answer_with, time_with, chunks_with, final_query = agent.ask_with_filter(test['question'])
        score_with = evaluate_answer(answer_with, test['expected_keywords'])
        total_score_with += score_with['score']
        
        print(f"\n  ✅ С ФИЛЬТРАЦИЕЙ ({time_with:.2f}с, чанков: {len(chunks_with)})")
        if final_query != test['question']:
            print(f"     Query Rewrite: {final_query}")
        print(f"     Точность: {score_with['percent']}% | Найдено: {score_with['found']}")
        print(f"     Ответ: {answer_with[:150]}...")
        
        results.append({
            "id": test['id'],
            "question": test['question'],
            "score_without": score_without['score'],
            "score_with": score_with['score'],
            "time_without": time_without,
            "time_with": time_with,
            "chunks_without": len(chunks_without),
            "chunks_with": len(chunks_with)
        })
        
        time.sleep(2)
    
    # Итоговая таблица
    print("\n" + "="*80)
    print("📊 ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*80)
    
    avg_without = total_score_without / len(results)
    avg_with = total_score_with / len(results)
    
    print(f"\n{'ID':<4} {'Вопрос (первые 30 символов)':<35} {'Без фильтра':<12} {'С фильтром':<12} {'Δ':<8}")
    print("-"*80)
    
    for r in results:
        q_short = r['question'][:30] + "..."
        delta = (r['score_with'] - r['score_without']) * 100
        print(f"{r['id']:<4} {q_short:<35} {r['score_without']*100:>5.1f}%     {r['score_with']*100:>5.1f}%     {delta:>+5.1f}%")
    
    print("-"*80)
    print(f"{'СРЕДНИЙ':<4} {'':<35} {avg_without*100:>5.1f}%     {avg_with*100:>5.1f}%     {(avg_with - avg_without)*100:>+5.1f}%")
    
    print("\n" + "="*80)
    print("✅ ВЫВОД:")
    if avg_with > avg_without:
        print(f"   🎉 РЕРАНКИНГ УЛУЧШИЛ КАЧЕСТВО НА {(avg_with - avg_without)*100:.1f}%")
    else:
        print(f"   ⚠️ РЕРАНКИНГ НЕ ПОКАЗАЛ УЛУЧШЕНИЯ")
    
    # Сохранение результатов
    output = {
        "test_date": datetime.now().isoformat(),
        "config": RERANK_CONFIG,
        "avg_score_without": avg_without,
        "avg_score_with": avg_with,
        "improvement": avg_with - avg_without,
        "results": results
    }
    
    filename = f"rerank_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Результаты сохранены: {filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank", action="store_true", default=True)
    args = parser.parse_args()
    
    run_comparison()