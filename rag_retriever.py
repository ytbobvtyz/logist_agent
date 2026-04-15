#!/usr/bin/env python3
"""
RAG Retriever для поиска релевантных чанков в индексе.
Использует TF-IDF для семантического поиска (легковесная версия).
"""

import sqlite3
import json
from typing import List, Dict, Optional


class RAGRetriever:
    """Класс для поиска релевантных чанков в индексе."""
    
    def __init__(self, db_path: str = "metadata.db"):
        """
        Инициализация RAG Retriever.
        
        Args:
            db_path: Путь к базе метаданных
        """
        self.db_path = db_path
        self.db_conn = None
        
        # Загружаем компоненты
        self._load_components()
    
    def _load_components(self):
        """Загружает базу данных и проверяет наличие индексов."""
        try:
            # Подключаемся к SQLite базе
            self.db_conn = sqlite3.connect(self.db_path)
            
            # Проверяем наличие таблицы
            if self.db_conn:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
                table_exists = cursor.fetchone() is not None
                
                if table_exists:
                    cursor.execute("SELECT COUNT(*) FROM chunks")
                    total_chunks = cursor.fetchone()[0]
                    print(f"✅ RAG Retriever успешно загружен")
                    print(f"   Всего чанков в базе: {total_chunks}")
                else:
                    print("⚠️ Таблица chunks не найдена в базе данных")
                    print("   Запустите TF-IDF-indexer.py для создания индексов")
            
        except Exception as e:
            print(f"⚠️ Ошибка загрузки RAG Retriever: {e}")
            print("   Возможно, база данных не создана. Запустите TF-IDF-indexer.py для создания индексов.")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Ищет топ-k релевантных чанков для запроса.
        Использует простой текстовый поиск по базе данных.
        
        Args:
            query: Текстовый запрос
            top_k: Количество возвращаемых результатов
            
        Returns:
            Список словарей с информацией о чанках:
            [
                {
                    "text": "текст чанка",
                    "filename": "pecom.txt",
                    "score": 0.85
                }
            ]
        """
        if not self.db_conn:
            print("⚠️ RAG Retriever не инициализирован")
            return []
        
        try:
            # Простой текстовый поиск по ключевым словам
            query_words = query.lower().split()
            
            # Ищем чанки, содержащие ключевые слова из запроса
            if self.db_conn:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT id, text, filename FROM chunks")
                all_chunks = cursor.fetchall()
                
                results = []
                for chunk_id, text, filename in all_chunks:
                    text_lower = text.lower()
                    
                    # Подсчитываем количество совпавших слов
                    matched_words = sum(1 for word in query_words if word in text_lower)
                    
                    if matched_words > 0:
                        # Простая оценка релевантности
                        score = matched_words / len(query_words)
                        
                        results.append({
                            "text": text,
                            "filename": filename,
                            "score": score,
                            "matched_words": matched_words
                        })
                
                # Сортируем по убыванию оценки
                results.sort(key=lambda x: x['score'], reverse=True)
                
                return results[:top_k]
            else:
                return []
            
        except Exception as e:
            print(f"⚠️ Ошибка поиска: {e}")
            return []
    
    def get_index_stats(self) -> Dict:
        """Возвращает статистику индекса."""
        if not self.db_conn:
            return {"error": "База данных не загружена"}
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM chunks")
            total_files = cursor.fetchone()[0]
            
            return {
                "total_chunks": total_chunks,
                "total_files": total_files
            }
        except Exception as e:
            return {"error": f"Ошибка получения статистики: {e}"}
    
    def close(self):
        """Закрывает соединения."""
        if self.db_conn:
            self.db_conn.close()


# Демонстрация работы
if __name__ == "__main__":
    retriever = RAGRetriever()
    
    # Статистика
    stats = retriever.get_index_stats()
    print("📊 Статистика индекса:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Тестовый поиск
    test_queries = [
        "стоимость доставки ПЭК",
        "максимальный вес СДЭК",
        "обязанности фрахтователя"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Поиск: '{query}'")
        results = retriever.search(query, top_k=2)
        
        if results:
            for i, result in enumerate(results):
                print(f"   [{i+1}] {result['filename']} (score: {result['score']:.3f})")
                print(f"      {result['text'][:100]}...")
        else:
            print("   ❌ Результаты не найдены")
    
    retriever.close()