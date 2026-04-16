#!/usr/bin/env python3
"""
RAG Retriever для поиска релевантных чанков в индексе.
Использует sentence-transformers и FAISS (совместим с indexer.py)
"""

import os
import sqlite3
import pickle
import numpy as np
from typing import List, Dict, Optional

# Проверка зависимостей
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError as e:
    FAISS_AVAILABLE = False
    print(f"⚠️ Предупреждение: {e}")
    print("   Установите: pip install sentence-transformers faiss-cpu")


class RAGRetriever:
    """
    Поиск релевантных чанков с использованием нейросетевых эмбеддингов.
    Совместим с indexer.py (sentence-transformers + FAISS)
    """
    
    def __init__(self, 
                 db_path: str = "metadata.db",
                 index_path: str = "faiss_index_sentence",
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Инициализация RAG Retriever.
        
        Args:
            db_path: Путь к SQLite с метаданными
            index_path: Путь к FAISS индексу
            model_name: Имя модели sentence-transformers
        """
        self.db_path = db_path
        self.index_path = index_path
        self.model_name = model_name
        
        self.index = None
        self.model = None
        self.db_conn = None
        
        self._load_components()
    
    def _load_components(self):
        """Загружает модель, FAISS индекс и БД"""
        print("🔧 Загрузка RAG Retriever...")
        
        # 1. Загрузка модели эмбеддингов
        if FAISS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                print(f"  ✓ Модель загружена (размерность: {self.model.get_sentence_embedding_dimension()})")
            except Exception as e:
                print(f"  ⚠️ Ошибка загрузки модели: {e}")
                self.model = None
        else:
            self.model = None
        
        # 2. Загрузка FAISS индекса
        if FAISS_AVAILABLE and os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"  ✓ FAISS индекс загружен: {self.index.ntotal} векторов")
            except Exception as e:
                print(f"  ⚠️ Ошибка загрузки FAISS: {e}")
                self.index = None
        else:
            print(f"  ⚠️ FAISS индекс не найден: {self.index_path}")
            self.index = None
        
        # 3. Подключение к SQLite
        if os.path.exists(self.db_path):
            try:
                self.db_conn = sqlite3.connect(self.db_path)
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chunks")
                total_chunks = cursor.fetchone()[0]
                print(f"  ✓ База данных загружена: {total_chunks} чанков")
            except Exception as e:
                print(f"  ⚠️ Ошибка подключения к БД: {e}")
                self.db_conn = None
        else:
            print(f"  ⚠️ База данных не найдена: {self.db_path}")
            self.db_conn = None
        
        # Проверка готовности
        if self.model and self.index and self.db_conn:
            print("✅ RAG Retriever готов к работе")
        else:
            print("⚠️ RAG Retriever работает в ограниченном режиме")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Ищет топ-k релевантных чанков для запроса."""
        
        if self.index is None or self.model is None or self.db_conn is None:
            return self._fallback_search(query, top_k)
        
        try:
            query_vector = self.model.encode([query])[0]
            query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            # Поиск возвращает (score, chunk_id)
            results_with_ids = self.index.search(query_vector, top_k * 2)
            
            # Получаем чанки из БД по ID
            results = []
            cursor = self.db_conn.cursor()
            
            for score, chunk_id in results_with_ids:
                cursor.execute(
                    "SELECT text, filename FROM chunks WHERE id = ?", 
                    (chunk_id,)
                )
                row = cursor.fetchone()
                if row:
                    results.append({
                        'text': row[0],
                        'filename': row[1],
                        'score': score
                    })
            
            return results[:top_k]
            
        except Exception as e:
            print(f"⚠️ Ошибка поиска: {e}")
            return self._fallback_search(query, top_k)
    
    def _fallback_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Упрощённый поиск (без эмбеддингов)"""
        if not self.db_conn:
            return []
        
        try:
            query_words = set(query.lower().split())
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT id, text, filename FROM chunks")
            all_chunks = cursor.fetchall()
            
            results = []
            for chunk_id, text, filename in all_chunks:
                text_lower = text.lower()
                matched = sum(1 for word in query_words if word in text_lower)
                if matched > 0:
                    score = matched / len(query_words) if query_words else 0
                    results.append({
                        'id': chunk_id,
                        'text': text,
                        'filename': filename,
                        'score': score
                    })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"⚠️ Ошибка fallback поиска: {e}")
            return []
    
    def get_index_stats(self) -> Dict:
        """Возвращает статистику индекса."""
        stats = {
            "model": self.model_name if self.model else None,
            "faiss_loaded": self.index is not None,
            "vectors_count": self.index.ntotal if self.index else 0,
            "dimension": self.index.d if self.index else 0
        }
        
        if self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chunks")
                stats["total_chunks"] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT filename) FROM chunks")
                stats["total_files"] = cursor.fetchone()[0]
            except Exception as e:
                stats["db_error"] = str(e)
        
        return stats
    
    def close(self):
        """Закрывает соединения."""
        if self.db_conn:
            self.db_conn.close()


# ============================================================
# ДЕМОНСТРАЦИЯ РАБОТЫ
# ============================================================

def demo():
    """Демонстрация работы RAG Retriever"""
    print("="*60)
    print("🚚 ДЕМОНСТРАЦИЯ RAG RETRIEVER")
    print("="*60)
    
    retriever = RAGRetriever()
    
    # Статистика
    print("\n📊 Статистика индекса:")
    stats = retriever.get_index_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Тестовые запросы
    test_queries = [
        "стоимость доставки ПЭК Москва Казань",
        "максимальный вес посылки СДЭК",
        "обязанности фрахтователя постановление",
        "какой URL у API ПЭК для расчёта стоимости",
    ]
    
    print("\n🔍 ТЕСТОВЫЕ ЗАПРОСЫ:")
    print("-"*60)
    
    for query in test_queries:
        print(f"\n📌 Запрос: '{query}'")
        results = retriever.search(query, top_k=2)
        
        if results:
            for i, r in enumerate(results, 1):
                print(f"   [{i}] Score: {r['score']:.4f} | Файл: {r['filename']}")
                print(f"       {r['text'][:120]}...")
        else:
            print("   ❌ Результаты не найдены")
    
    retriever.close()


if __name__ == "__main__":
    demo()