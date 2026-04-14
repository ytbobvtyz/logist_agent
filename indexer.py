#!/usr/bin/env python3
"""
RAG система для индексации документов перевозчиков

Этот скрипт создает локальный индекс для поиска по документам перевозчиков
с использованием эмбеддингов и векторного поиска.
"""

import os
import re
import time
import sqlite3
import math
from typing import List, Dict, Tuple, Optional
import numpy as np

class SentenceTransformer:
    def __init__(self, model_name=None):
        self.vocab = {}
        self.vocab_size = 0
        self.is_fitted = False
        
        # Словарь синонимов для улучшения поиска
        self.synonyms = {
            'питер': 'санкт-петербург',
            'спб': 'санкт-петербург',
            'стоит': 'стоимость',
            'отправить': 'доставка',
            'груз': 'посылка',
            'кг': 'килограмм',
            'москвы': 'москва'
        }
        
    def _normalize_word(self, word):
        """Нормализует слово и заменяет синонимы"""
        # Приводим к нижнему регистру
        word = word.lower()
        
        # Заменяем синонимы
        if word in self.synonyms:
            return self.synonyms[word]
        
        return word
        
    def _build_vocab(self, texts):
        """Строит словарь уникальных слов из всех текстов"""
        all_words = set()
        for text in texts:
            # Используем более широкий паттерн для русских слов
            words = re.findall(r'[а-яёa-z0-9]+', text.lower())
            # Нормализуем слова
            normalized_words = [self._normalize_word(word) for word in words]
            all_words.update(normalized_words)
        
        self.vocab = {word: idx for idx, word in enumerate(sorted(all_words))}
        self.vocab_size = len(self.vocab)
        self.is_fitted = True
        
    def _text_to_vector(self, text):
        """Преобразует текст в вектор TF-IDF"""
        if not self.is_fitted:
            return np.zeros(384)
            
        # Используем более широкий паттерн для русских слов
        words = re.findall(r'[а-яёa-z0-9]+', text.lower())
        # Нормализуем слова
        normalized_words = [self._normalize_word(word) for word in words]
        
        # Ключевые слова для повышения релевантности
        key_terms = ['москва', 'санкт-петербург', 'тариф', 'стоимость', 'доставка', 'кг', 'груз']
        
        word_counts = {}
        for word in normalized_words:
            if word in self.vocab:
                # Увеличиваем вес ключевых слов
                weight = 2.0 if word in key_terms else 1.0
                word_counts[word] = word_counts.get(word, 0) + weight
        
        # Простой TF-IDF (без IDF для простоты)
        vector = np.zeros(self.vocab_size)
        for word, count in word_counts.items():
            vector[self.vocab[word]] = count
        
        # Нормализация
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
        
    def encode(self, texts):
        if not self.is_fitted:
            self._build_vocab(texts)
        
        embeddings = []
        for text in texts:
            vector = self._text_to_vector(text)
            # Если размерность меньше 384, дополняем нулями
            if len(vector) < 384:
                vector = np.pad(vector, (0, 384 - len(vector)))
            elif len(vector) > 384:
                vector = vector[:384]
            embeddings.append(vector)
            
        return np.array(embeddings)
        
    def embed_text(self, text):
        return self.encode([text])[0]


class DocumentLoader:
    """Загрузчик документов перевозчиков из папки data/carriers."""
    
    def __init__(self, data_dir: str = "data/carriers"):
        self.data_dir = data_dir
    
    def load_documents(self) -> List[Dict]:
        """
        Загружает все .txt файлы из папки перевозчиков.
        
        Returns:
            Список словарей с метаданными и содержимым документов
        """
        documents = []
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Директория {self.data_dir} не существует")
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.data_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read().strip()
                    
                    documents.append({
                        'filename': filename,
                        'filepath': file_path,
                        'content': content
                    })
                    print(f"✓ Загружен {filename}")
                    
                except Exception as e:
                    print(f"✗ Ошибка при загрузке {filename}: {e}")
        
        return documents


class Chunker:
    """Реализует две стратегии чанкинга: фиксированный размер и по предложениям."""
    
    def __init__(self):
        self.fixed_params = {'chunk_size': 500, 'overlap': 50}
    
    def fixed_size_chunking(self, text: str) -> List[str]:
        """
        Разбивает текст на чанки фиксированного размера с перекрытием.
        
        Args:
            text: Входной текст для разбиения
            
        Returns:
            Список текстовых чанков
        """
        chunk_size = self.fixed_params['chunk_size']
        overlap = self.fixed_params['overlap']
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Убедимся, что чанк не обрывается посередине слова
            if end < len(text) and text[end] not in ' .,!?;:\n':
                # Найдем ближайший пробел или знак препинания
                space_pos = chunk.rfind(' ')
                if space_pos != -1:
                    chunk = chunk[:space_pos]
                    end = start + space_pos
            
            if len(chunk) > 0:
                chunks.append(chunk.strip())
            
            start += chunk_size - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def sentence_based_chunking(self, text: str) -> List[str]:
        """
        Разбивает текст на чанки по границам предложений.
        
        Args:
            text: Входной текст для разбиения
            
        Returns:
            Список текстовых чанков
        """
        # Разбиваем по окончаниям предложений (. ! ? с последующим пробелом или новой строкой)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Если добавление этого предложения сделает чанк слишком большим, сохраняем текущий чанк
            if len(current_chunk) + len(sentence) > 500:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Одно предложение слишком длинное, разбиваем его
                    if len(sentence) > 500:
                        # Разбиваем длинное предложение по словам
                        words = sentence.split()
                        temp_chunk = ""
                        for word in words:
                            if len(temp_chunk) + len(word) < 450:
                                temp_chunk += " " + word
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                    temp_chunk = word
                                else:
                                    chunks.append(word)
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                    else:
                        chunks.append(sentence.strip())
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


class EmbeddingModel:
    """Обрабатывает эмбеддинги текста с использованием sentence-transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Преобразует текст в вектор эмбеддинга.
        
        Args:
            text: Входной текст
            
        Returns:
            Numpy массив с вектором эмбеддинга
        """
        return self.model.encode([text])[0]
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Преобразует пакет текстов в векторы эмбеддингов.
        
        Args:
            texts: Список входных текстов
            
        Returns:
            Numpy массив с векторами эмбеддингов
        """
        return self.model.encode(texts)


class SimpleVectorIndex:
    """Простая реализация векторного индекса для поиска по косинусному сходству."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors = []
        self.chunk_ids = []
    
    def add_vectors(self, vectors: np.ndarray, chunk_ids: List[int]) -> None:
        """
        Добавляет векторы в индекс.
        
        Args:
            vectors: Numpy массив векторов эмбеддингов
            chunk_ids: Список ID чанков, соответствующих векторам
        """
        for i, vector in enumerate(vectors):
            # Нормализуем вектор для косинусного сходства
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            self.vectors.append(vector)
            self.chunk_ids.append(chunk_ids[i])
    
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ищет похожие векторы.
        
        Args:
            query_vector: Вектор эмбеддинга запроса
            top_k: Количество возвращаемых топ-результатов
            
        Returns:
            Кортеж из (расстояния, индексы)
        """
        # Нормализуем вектор запроса
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        # Вычисляем косинусное сходство со всеми векторами
        similarities = []
        for vector in self.vectors:
            similarity = np.dot(query_vector, vector)
            similarities.append(similarity)
        
        # Получаем топ-K результатов
        similarities = np.array(similarities)
        indices = np.argsort(similarities)[::-1][:top_k]
        distances = similarities[indices]
        
        return distances, indices
    
    def save_index(self, filepath: str) -> None:
        """Сохраняет индекс в файл."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectors': self.vectors,
                'chunk_ids': self.chunk_ids,
                'dimension': self.dimension
            }, f)
    
    def load_index(self, filepath: str) -> None:
        """Загружает индекс из файла."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectors = data['vectors']
            self.chunk_ids = data['chunk_ids']
            self.dimension = data['dimension']


class MetadataDB:
    """Управляет SQLite базой данных для метаданных чанков."""
    
    def __init__(self, db_path: str = "metadata.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Инициализирует схему базы данных."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_text TEXT NOT NULL,
                filename TEXT NOT NULL,
                strategy TEXT NOT NULL,
                chunk_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_chunk(self, chunk_text: str, filename: str, strategy: str, chunk_size: int) -> int:
        """
        Добавляет метаданные чанка в базу данных.
        
        Args:
            chunk_text: Текстовое содержимое чанка
            filename: Имя исходного файла
            strategy: Использованная стратегия чанкинга
            chunk_size: Размер чанка
            
        Returns:
            ID добавленного чанка
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO chunks (chunk_text, filename, strategy, chunk_size)
            VALUES (?, ?, ?, ?)
        ''', (chunk_text, filename, strategy, chunk_size))
        
        chunk_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return chunk_id if chunk_id is not None else 0
    
    def get_chunk(self, chunk_id: int) -> Optional[Dict]:
        """
        Получает метаданные чанка по ID.
        
        Args:
            chunk_id: ID чанка
            
        Returns:
            Словарь с метаданными чанка или None если не найден
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, chunk_text, filename, strategy, chunk_size
            FROM chunks WHERE id = ?
        ''', (chunk_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'chunk_text': row[1],
                'filename': row[2],
                'strategy': row[3],
                'chunk_size': row[4]
            }
        return None


class RAGIndexer:
    """Основная система индексации RAG, которая координирует все компоненты."""
    
    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = Chunker()
        self.embedder = EmbeddingModel()
        self.vector_index = SimpleVectorIndex()
        self.metadata_db = MetadataDB()
        
        # Статистика
        self.stats = {}
        
        # Загружаем существующий индекс если он есть
        if os.path.exists("vector_index") and os.path.exists("metadata.db"):
            try:
                self.vector_index.load_index("vector_index")
                print("✓ Загружен существующий векторный индекс")
            except Exception as e:
                print(f"⚠ Ошибка при загрузке индекса: {e}")
    
    def index_documents(self) -> Dict:
        """
        Основная функция индексации - обрабатывает документы обеими стратегиями чанкинга.
        
        Returns:
            Словарь со статистикой индексации
        """
        print("Загрузка документов...")
        documents = self.loader.load_documents()
        
        if not documents:
            raise ValueError("Не найдено документов для индексации")
        
        strategies = ['fixed', 'sentence']
        stats = {}
        
        for strategy in strategies:
            print(f"\nОбработка стратегией чанкинга: {strategy}...")
            start_time = time.time()
            
            chunks = []
            chunk_ids = []
            vectors = []
            
            for doc in documents:
                if strategy == 'fixed':
                    doc_chunks = self.chunker.fixed_size_chunking(doc['content'])
                else:
                    doc_chunks = self.chunker.sentence_based_chunking(doc['content'])
                
                for chunk in doc_chunks:
                    chunk_id = self.metadata_db.add_chunk(
                        chunk, doc['filename'], strategy, len(chunk)
                    )
                    chunks.append(chunk)
                    chunk_ids.append(chunk_id)
            
            # Генерируем эмбеддинги
            if chunks:
                embeddings = self.embedder.embed_batch(chunks)
                vectors.append(embeddings)
                
                # Добавляем в векторный индекс
                self.vector_index.add_vectors(embeddings, chunk_ids)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            stats[strategy] = {
                'chunk_count': len(chunks),
                'avg_chunk_size': sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
                'processing_time': processing_time
            }
            
            print(f"  ✓ Создано {len(chunks)} чанков")
            print(f"  ✓ Средний размер чанка: {stats[strategy]['avg_chunk_size']:.0f} символов")
            print(f"  ✓ Время обработки: {processing_time:.2f} секунд")
        
        # Сохраняем векторный индекс
        self.vector_index.save_index("vector_index")
        print("\n✓ Векторный индекс сохранен в 'vector_index'")
        
        self.stats = stats
        return stats
    
    def search(self, query: str, strategy: str = "fixed", top_k: int = 3) -> List[Dict]:
        """
        Ищет релевантные чанки на основе запроса.
        
        Args:
            query: Поисковый запрос
            strategy: Стратегия чанкинга для поиска
            top_k: Количество возвращаемых топ-результатов
            
        Returns:
            Список словарей с результатами поиска
        """
        # Нормализуем запрос
        query_lower = query.lower()
        query_terms = re.findall(r'[а-яёa-z0-9]+', query_lower)
        
        # Для коротких запросов используем точное совпадение ключевых слов
        if len(query_terms) <= 5:
            return self._keyword_search(query, strategy, top_k)
        
        # Для длинных запросов используем векторный поиск
        return self._vector_search(query, strategy, top_k)
    
    def _keyword_search(self, query: str, strategy: str, top_k: int) -> List[Dict]:
        """Поиск по точному совпадению ключевых слов"""
        query_lower = query.lower()
        query_terms = re.findall(r'[а-яёa-z0-9]+', query_lower)
        
        results = []
        
        # Получаем все чанки из базы данных
        conn = sqlite3.connect("metadata.db")
        cursor = conn.cursor()
        cursor.execute('SELECT id, chunk_text, filename, strategy, chunk_size FROM chunks')
        
        for row in cursor.fetchall():
            chunk_id, chunk_text, filename, chunk_strategy, chunk_size = row
            
            # Фильтруем по стратегии если указана
            if strategy != "all" and chunk_strategy != strategy:
                continue
            
            text_lower = chunk_text.lower()
            
            # Вычисляем оценку релевантности
            relevance_score = 0.0
            
            # Проверяем совпадение терминов
            matched_terms = 0
            for term in query_terms:
                if term in text_lower:
                    matched_terms += 1
                    relevance_score += 0.2
            
            # Повышаем оценку если найден ключевые слова
            key_terms = ['москва', 'санкт-петербург', 'тариф', 'стоимость', 'доставка', 'кг']
            for term in key_terms:
                if term in text_lower:
                    relevance_score += 0.1
            
            # Повышаем приоритет для файлов с фактическими тарифами
            if filename in ['pecom.txt', 'cdek.txt', 'post_russia.txt']:
                relevance_score += 0.3
            
            # Понижаем приоритет для технической документации
            if filename in ['pecom_api_doc.txt']:
                relevance_score -= 0.2
            
            if relevance_score > 0:
                results.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'filename': filename,
                    'strategy': chunk_strategy,
                    'chunk_size': chunk_size,
                    'similarity_score': min(1.0, relevance_score)
                })
        
        conn.close()
        
        # Сортируем по оценке релевантности
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results[:top_k]
    
    def _vector_search(self, query: str, strategy: str, top_k: int) -> List[Dict]:
        """Векторный поиск"""
        # Генерируем эмбеддинг запроса
        query_vector = self.embedder.embed_text(query)
        
        # Ищем в векторном индексе
        distances, indices = self.vector_index.search(query_vector, top_k * 2)
        
        results = []
        for distance, chunk_idx in zip(distances, indices):
            chunk_id = self.vector_index.chunk_ids[chunk_idx]
            chunk_data = self.metadata_db.get_chunk(chunk_id)
            
            if chunk_data:
                # Фильтруем по стратегии если указана
                if strategy != "all" and chunk_data['strategy'] != strategy:
                    continue
                
                results.append({
                    'chunk_id': chunk_data['id'],
                    'text': chunk_data['chunk_text'],
                    'filename': chunk_data['filename'],
                    'strategy': chunk_data['strategy'],
                    'chunk_size': chunk_data['chunk_size'],
                    'similarity_score': float(distance)
                })
        
        return results[:top_k]
    
    def print_comparison_table(self) -> None:
        """Выводит таблицу сравнения стратегий чанкинга."""
        if not self.stats:
            print("Статистика недоступна. Сначала выполните index_documents().")
            return
        
        print("\n" + "="*60)
        print("СРАВНЕНИЕ СТРАТЕГИЙ ЧАНКИНГА")
        print("="*60)
        print(f"{'Стратегия':<12} {'Кол-во чанков':<15} {'Средний размер':<15} {'Время (сек)':<12}")
        print("-"*60)
        
        for strategy, data in self.stats.items():
            print(f"{strategy:<12} {data['chunk_count']:<15} {data['avg_chunk_size']:<15.0f} {data['processing_time']:<12.2f}")
        print("="*60)


def main():
    """Основная функция для запуска процесса индексации."""
    print("RAG система для индексации документов перевозчиков")
    print("="*50)
    
    # Инициализируем индексатор
    indexer = RAGIndexer()
    
    try:
        # Индексируем документы
        stats = indexer.index_documents()
        
        # Выводим таблицу сравнения
        indexer.print_comparison_table()
        
        # Тестовый поиск
        print("\nТестирование функционала поиска...")
        test_queries = [
            "сколько стоит отправить груз 30 кг из Москвы в Питер",
            "стоимость доставки до 50 кг",
            "условия перевозки опасных грузов"
        ]
        
        for test_query in test_queries:
            print(f"\nРезультаты поиска для: '{test_query}'")
            print("-"*50)
            
            results = indexer.search(test_query, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"\nРезультат {i}:")
                print(f"  Файл: {result['filename']}")
                print(f"  Стратегия: {result['strategy']}")
                print(f"  Сходство: {result['similarity_score']:.4f}")
                print(f"  Предпросмотр текста: {result['text'][:100]}...")
        
        print("\n✓ Индексация завершена успешно!")
        print("✓ Векторный индекс сохранен в 'vector_index'")
        print("✓ Метаданные сохранены в 'metadata.db'")
        
    except Exception as e:
        print(f"Ошибка при индексации: {e}")


if __name__ == "__main__":
    main()