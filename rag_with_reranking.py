#!/usr/bin/env python3
"""
RAG с реранкингом и фильтрацией для логист-агента
День 23: Улучшенный поиск с порогом релевантности
"""

import json
import sqlite3
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
import pickle
import os

# Импортируем существующие модули
try:
    import faiss
    from TF_IDF_vectorizer import SimpleTfidfVectorizer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS не доступен, работаем в упрощённом режиме")


class RAGReranker:
    """RAG с реранкингом и фильтрацией"""
    
    def __init__(self, 
                 db_path: str = "metadata.db",
                 index_path: str = "faiss_index",
                 vectorizer_path: str = "tfidf_vectorizer.pkl",
                 threshold: float = 0.3,
                 top_k_before: int = 10,
                 top_k_after: int = 3):
        
        self.db_path = db_path
        self.index_path = index_path
        self.vectorizer_path = vectorizer_path
        self.threshold = threshold
        self.top_k_before = top_k_before
        self.top_k_after = top_k_after
        
        self.index = None
        self.vectorizer = None
        self._load()
    
    def _load(self):
        """Загружает индекс и векторизатор"""
        print(f"🔧 Загрузка RAG с реранкингом (порог={self.threshold})...")
        
        # Загрузка векторизатора
        if os.path.exists(self.vectorizer_path):
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f"  ✓ Векторизатор загружен")
        
        # Загрузка FAISS индекса
        if FAISS_AVAILABLE and os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"  ✓ FAISS индекс загружен: {self.index.ntotal} векторов")
    
    def search(self, query: str) -> List[Dict]:
        """Поиск с реранкингом и фильтрацией"""
        
        # Этап 1: Первичный поиск (топ-K до фильтрации)
        raw_chunks = self._initial_search(query, self.top_k_before)
        
        # Этап 2: Фильтрация по порогу
        filtered_chunks = self._filter_by_threshold(raw_chunks)
        
        # Этап 3: Реранкинг (пересортировка)
        reranked_chunks = self._rerank(filtered_chunks, query)
        
        # Этап 4: Обрезаем до топ-K после фильтрации
        final_chunks = reranked_chunks[:self.top_k_after]
        
        # Этап 5: Query Rewrite если мало результатов
        if len(final_chunks) < 2:
            rewritten_query = self._rewrite_query(query, raw_chunks)
            if rewritten_query != query:
                print(f"  🔄 Query Rewrite: '{query}' → '{rewritten_query}'")
                return self.search(rewritten_query)
        
        return final_chunks
    
    def _initial_search(self, query: str, top_k: int) -> List[Dict]:
        """Первичный поиск в индексе"""
        
        if not self.vectorizer or self.index is None:
            return self._sqlite_search(query, top_k)
        
        try:
            # Получаем вектор запроса
            query_vector = self.vectorizer.transform([query])[0]
            query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            
            # Поиск в FAISS
            faiss.normalize_L2(query_vector)
            distances, indices = self.index.search(query_vector, top_k)
            
            # Получаем метаданные
            results = []
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0:
                    cursor.execute(
                        "SELECT id, text, filename FROM chunks WHERE id = ?", 
                        (idx + 1,)
                    )
                    row = cursor.fetchone()
                    if row:
                        results.append({
                            'id': row[0],
                            'text': row[1],
                            'filename': row[2],
                            'score': float(dist),
                            'raw_score': float(dist)
                        })
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"⚠️ Ошибка поиска: {e}")
            return self._sqlite_search(query, top_k)
    
    def _sqlite_search(self, query: str, top_k: int) -> List[Dict]:
        """Упрощённый поиск через SQLite"""
        results = []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        keywords = query.lower().split()
        for kw in keywords[:5]:
            cursor.execute("""
                SELECT id, text, filename FROM chunks 
                WHERE LOWER(text) LIKE ? 
                LIMIT ?
            """, (f'%{kw}%', top_k))
            
            for row in cursor.fetchall():
                if not any(r.get('id') == row[0] for r in results):
                    results.append({
                        'id': row[0],
                        'text': row[1],
                        'filename': row[2],
                        'score': 0.5,
                        'raw_score': 0.5
                    })
        
        conn.close()
        return results[:top_k]
    
    def _filter_by_threshold(self, chunks: List[Dict]) -> List[Dict]:
        """Фильтрация по порогу релевантности"""
        before = len(chunks)
        filtered = [c for c in chunks if c.get('score', 0) >= self.threshold]
        
        if before != len(filtered):
            print(f"  🔍 Фильтрация: {before} → {len(filtered)} чанков (порог={self.threshold})")
        
        return filtered
    
    def _rerank(self, chunks: List[Dict], query: str) -> List[Dict]:
        """Реранкинг на основе ключевых слов из запроса"""
        if not chunks:
            return chunks
        
        keywords = set(query.lower().split())
        
        for chunk in chunks:
            chunk_text = chunk['text'].lower()
            
            # Считаем пересечение ключевых слов
            matched = sum(1 for kw in keywords if kw in chunk_text)
            keyword_score = matched / len(keywords) if keywords else 0
            
            # Финальный скор: FAISS скор + бонус за ключевые слова
            chunk['keyword_score'] = keyword_score
            chunk['final_score'] = chunk.get('score', 0) + keyword_score * 0.3
            chunk['score'] = chunk['final_score']
        
        return sorted(chunks, key=lambda x: x['score'], reverse=True)
    
    def _rewrite_query(self, query: str, original_chunks: List[Dict]) -> str:
        """Расширяет запрос, если результатов мало"""
        if len(original_chunks) >= 2:
            return query
        
        # Словарь синонимов и связанных терминов
        expansions = {
            'стоимость': ['цена', 'тариф', 'руб'],
            'доставка': ['перевозка', 'отправка', 'транспортировка'],
            'груз': ['посылка', 'отправление', 'товар'],
            'пэк': ['ПЭК', 'Первая Экспедиционная Компания'],
            'сдэк': ['СДЭК', 'CDEK'],
        }
        
        words = query.lower().split()
        new_words = words.copy()
        
        for word in words:
            if word in expansions:
                new_words.extend(expansions[word])
        
        return ' '.join(new_words[:15])  # ограничиваем длину


# ============================================================
# ТЕСТИРОВАНИЕ
# ============================================================

def run_comparison():
    """Сравнение качества без фильтра и с фильтром"""
    
    print("="*70)
    print("🔍 ДЕНЬ 23: СРАВНЕНИЕ RAG С ФИЛЬТРАЦИЕЙ И БЕЗ")
    print("="*70)
    print(f"📅 Дата теста: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Инициализация
    rag_no_filter = RAGReranker(threshold=0.0, top_k_before=10, top_k_after=5)
    rag_with_filter = RAGReranker(threshold=0.3, top_k_before=10, top_k_after=3)
    
    # Тестовые запросы
    test_queries = [
        "стоимость доставки ПЭК Москва Казань",
        "максимальный вес посылки СДЭК",
        "обязанности фрахтователя постановление",
        "а скажите пожалуйста сколько будет стоить доставка груза из Москвы в Казань у ПЭК",
        "логистика",
        "ПЭК",
    ]
    
    results = []
    
    for query in test_queries:
        print(f"\n{'─'*70}")
        print(f"📌 Запрос: {query}")
        print("-"*70)
        
        # Без фильтра
        chunks_no_filter = rag_no_filter._initial_search(query, 5)
        print(f"\n❌ БЕЗ ФИЛЬТРАЦИИ (топ-5):")
        for i, c in enumerate(chunks_no_filter):
            print(f"   [{i+1}] {c['filename']} (score: {c['score']:.3f})")
            print(f"       {c['text'][:100]}...")
        
        # С фильтром
        chunks_with_filter = rag_with_filter.search(query)
        print(f"\n✅ С ФИЛЬТРАЦИЕЙ И РЕРАНКИНГОМ:")
        for i, c in enumerate(chunks_with_filter):
            print(f"   [{i+1}] {c['filename']} (score: {c['score']:.3f})")
            print(f"       {c['text'][:100]}...")
        
        # Оценка
        results.append({
            "query": query,
            "chunks_no_filter_count": len(chunks_no_filter),
            "chunks_with_filter_count": len(chunks_with_filter),
            "improvement": len(chunks_with_filter) > 0 and len(chunks_with_filter) < len(chunks_no_filter)
        })
    
    # Итоговая таблица
    print("\n" + "="*70)
    print("📊 ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*70)
    print(f"{'Запрос':<45} {'Без фильтра':<12} {'С фильтром':<12} {'Результат':<10}")
    print("-"*70)
    
    for r in results:
        query_short = r['query'][:42] + "..." if len(r['query']) > 42 else r['query']
        improvement = "✅ чище" if r['improvement'] else "➖"
        print(f"{query_short:<45} {r['chunks_no_filter_count']:<12} {r['chunks_with_filter_count']:<12} {improvement:<10}")
    
    print("="*70)
    print("\n💡 ВЫВОД:")
    print("   - Фильтрация отсекает нерелевантные чанки")
    print("   - Реранкинг поднимает релевантные результаты вверх")
    print("   - Query Rewrite помогает при пустых результатах")
    
    return results


def run_full_test():
    """Полное тестирование с 10 запросами"""
    
    rag = RAGReranker(threshold=0.3, top_k_before=10, top_k_after=3)
    
    test_queries = [
        "стоимость доставки ПЭК Москва Казань 50 кг",
        "максимальный вес посылки СДЭК",
        "обязанности фрахтователя постановление правительства",
        "сколько стоит отправить посылку 1 кг Почтой России",
        "какой URL у API ПЭК для расчёта стоимости",
        "как передать вес груза в API ПЭК",
        "какой формат ответа возвращает API ПЭК",
        "что такое фрахтовщик по закону",
        "правила перевозки опасных грузов",
        "стоимость доставки ПЭК Москва Санкт-Петербург 100 кг",
    ]
    
    print("="*70)
    print("🚚 ЛОГИСТ-АГЕНТ: УЛУЧШЕННЫЙ RAG С РЕРАНКИНГОМ")
    print("="*70)
    
    all_results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📌 {i}. {query}")
        print("-"*50)
        
        chunks = rag.search(query)
        
        if chunks:
            print(f"   ✅ Найдено {len(chunks)} релевантных чанков:")
            for j, c in enumerate(chunks, 1):
                print(f"      [{j}] {c['filename']} (score: {c['score']:.3f})")
                print(f"          {c['text'][:150]}...")
        else:
            print("   ❌ Релевантных чанков не найдено")
        
        all_results.append({
            "query": query,
            "found_count": len(chunks),
            "top_chunk": chunks[0]['filename'] if chunks else None,
            "top_score": chunks[0]['score'] if chunks else None
        })
    
    # Статистика
    print("\n" + "="*70)
    print("📊 СТАТИСТИКА")
    print("="*70)
    
    found = sum(1 for r in all_results if r['found_count'] > 0)
    print(f"   Успешных поисков: {found}/{len(test_queries)} ({found/len(test_queries)*100:.0f}%)")
    
    avg_score = sum(r['top_score'] for r in all_results if r['top_score']) / found if found else 0
    print(f"   Средний скор топ-результата: {avg_score:.3f}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG с реранкингом")
    parser.add_argument("--compare", action="store_true", 
                       help="Сравнение с фильтрацией и без")
    parser.add_argument("--test", action="store_true",
                       help="Полное тестирование на 10 запросах")
    
    args = parser.parse_args()
    
    if args.compare:
        run_comparison()
    elif args.test:
        run_full_test()
    else:
        # По умолчанию запускаем сравнение
        run_comparison()