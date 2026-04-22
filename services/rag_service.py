"""
Сервис RAG (Retrieval-Augmented Generation) для поиска информации в документах.
Использует векторные эмбеддинги для семантического поиска.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import sqlite3

from utils.config import settings


@dataclass
class RAGResult:
    """Результат RAG поиска."""
    
    text: str
    filename: str
    similarity_score: float
    chunk_id: int
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует результат в словарь."""
        return {
            "text": self.text,
            "filename": self.filename,
            "similarity_score": self.similarity_score,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata or {}
        }


class RAGService:
    """Сервис RAG для семантического поиска в документах."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.model = None
            self.index = None
            self.metadata_db = None
            self._index_loaded = False
            
            # Загружаем модель и индекс при инициализации
            self._load_model()
            self._load_index()
    
    def _load_model(self):
        """Загружает модель эмбеддингов."""
        try:
            print("🧠 Загрузка модели эмбеддингов...")
            self.model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
            print("✅ Модель эмбеддингов загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self.model = None
    
    def _load_index(self):
        """Загружает FAISS индекс и метаданные."""
        try:
            # Проверяем существование файлов
            index_path = settings.rag_index_path
            metadata_path = settings.rag_metadata_path
            
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                print("⚠️ RAG индекс не найден. Запустите indexer.py для создания индекса.")
                return
            
            # Загружаем FAISS индекс
            print("📚 Загрузка FAISS индекса...")
            self.index = faiss.read_index(index_path)
            
            # Подключаемся к базе метаданных
            self.metadata_db = sqlite3.connect(metadata_path)
            self.metadata_db.row_factory = sqlite3.Row
            
            self._index_loaded = True
            print(f"✅ RAG индекс загружен: {self.index.ntotal} чанков")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки RAG индекса: {e}")
            self._index_loaded = False
    
    def is_available(self) -> bool:
        """
        Проверяет, доступен ли RAG сервис.
        
        Returns:
            True если доступен, иначе False
        """
        return self._index_loaded and self.model is not None and self.index is not None
    
    def search(self, query: str, top_k: int = 5) -> List[RAGResult]:
        """
        Выполняет семантический поиск по документам.
        
        Args:
            query: Поисковый запрос
            top_k: Количество возвращаемых результатов
            
        Returns:
            Список результатов поиска
        """
        if not self.is_available():
            print("⚠️ RAG сервис недоступен")
            return []
        
        try:
            # Генерируем эмбеддинг для запроса
            query_embedding = self.model.encode([query])
            query_embedding = np.array(query_embedding, dtype='float32')
            
            # Ищем в индексе
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Получаем метаданные для найденных чанков
            results = []
            cursor = self.metadata_db.cursor()
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:
                    continue  # Пропускаем пустые результаты
                
                cursor.execute(
                    "SELECT * FROM chunks WHERE id = ?",
                    (int(idx),)
                )
                row = cursor.fetchone()
                
                if row:
                    similarity_score = 1.0 / (1.0 + distance)  # Преобразуем расстояние в схожесть
                    
                    result = RAGResult(
                        text=row['text'],
                        filename=row['filename'],
                        similarity_score=similarity_score,
                        chunk_id=row['id'],
                        metadata={
                            'chunk_index': row['chunk_index'],
                            'strategy': row['strategy']
                        }
                    )
                    results.append(result)
            
            cursor.close()
            return results
            
        except Exception as e:
            print(f"❌ Ошибка поиска в RAG: {e}")
            return []
    
    def search_with_threshold(self, query: str, top_k: int = 5, 
                             similarity_threshold: float = 0.5) -> List[RAGResult]:
        """
        Выполняет поиск с порогом схожести.
        
        Args:
            query: Поисковый запрос
            top_k: Количество возвращаемых результатов
            similarity_threshold: Минимальная схожесть (0.0-1.0)
            
        Returns:
            Отфильтрованные результаты поиска
        """
        results = self.search(query, top_k * 2)  # Ищем больше, чтобы отфильтровать
        
        # Фильтруем по порогу схожести
        filtered_results = [
            result for result in results 
            if result.similarity_score >= similarity_threshold
        ]
        
        # Возвращаем top_k результатов после фильтрации
        return filtered_results[:top_k]
    
    def get_context_for_query(self, query: str, max_chars: int = 2000) -> str:
        """
        Получает контекст для запроса из документов.
        
        Args:
            query: Поисковый запрос
            max_chars: Максимальная длина контекста
            
        Returns:
            Собранный контекст
        """
        results = self.search(query, top_k=10)
        
        if not results:
            return ""
        
        # Сортируем по схожести
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Собираем контекст
        context_parts = []
        total_chars = 0
        
        for result in results:
            if total_chars + len(result.text) > max_chars:
                break
            
            context_parts.append(f"[Из {result.filename}]: {result.text}")
            total_chars += len(result.text)
        
        return "\n\n".join(context_parts)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику индекса.
        
        Returns:
            Словарь со статистикой
        """
        if not self.is_available():
            return {"error": "RAG индекс не загружен"}
        
        try:
            cursor = self.metadata_db.cursor()
            
            # Общая статистика
            cursor.execute("SELECT COUNT(*) as total_chunks FROM chunks")
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT filename) as total_files FROM chunks")
            total_files = cursor.fetchone()[0]
            
            # Статистика по стратегиям чанкинга
            cursor.execute(
                "SELECT strategy, COUNT(*) as count FROM chunks GROUP BY strategy"
            )
            strategies = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.close()
            
            return {
                "total_chunks": total_chunks,
                "total_files": total_files,
                "index_dimensions": self.index.d,
                "index_size_mb": os.path.getsize(settings.rag_index_path) / (1024 * 1024),
                "strategies": strategies,
                "available": True
            }
            
        except Exception as e:
            return {"error": str(e), "available": False}
    
    def format_results_for_display(self, results: List[RAGResult]) -> str:
        """
        Форматирует результаты поиска для отображения.
        
        Args:
            results: Список результатов поиска
            
        Returns:
            Отформатированная строка
        """
        if not results:
            return "🔍 По вашему запросу ничего не найдено."
        
        lines = ["🔍 **Найдено в документах:**", ""]
        
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. 📄 **{result.filename}** (схожесть: {result.similarity_score:.2%})")
            lines.append(f"   {result.text[:200]}..." if len(result.text) > 200 else f"   {result.text}")
            lines.append("")
        
        return "\n".join(lines)
    
    def test_search(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """
        Тестирует поисковую систему.
        
        Args:
            test_queries: Список тестовых запросов
            
        Returns:
            Результаты тестирования
        """
        if test_queries is None:
            test_queries = [
                "стоимость доставки",
                "тариф Москва Санкт-Петербург",
                "условия перевозки",
                "максимальный вес"
            ]
        
        results = {}
        
        for query in test_queries:
            search_results = self.search(query, top_k=3)
            results[query] = {
                "found": len(search_results) > 0,
                "count": len(search_results),
                "top_score": search_results[0].similarity_score if search_results else 0.0
            }
        
        return results


# Глобальный экземпляр RAG сервиса
_rag_service = None

def get_rag_service() -> RAGService:
    """Возвращает глобальный экземпляр RAG сервиса."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service