#!/usr/bin/env python3
"""
RAG система для индексации документов перевозчиков
Оптимизировано для i7 / 12GB RAM
"""

import os
import re
import time
import sqlite3
from typing import List, Dict, Tuple
import numpy as np

# Проверка зависимостей
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"❌ Ошибка: установи зависимости:")
    print(f"   pip install sentence-transformers faiss-cpu")
    raise e


# ============================================================
# 1. ЗАГРУЗКА ДОКУМЕНТОВ
# ============================================================

def load_documents(data_dir: str = "data/carriers") -> List[Dict]:
    """Загружает все .txt файлы из папки"""
    documents = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Папка {data_dir} не найдена")
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    documents.append({
                        'filename': filename,
                        'content': content
                    })
                    print(f"  ✓ Загружен: {filename}")
    
    print(f"\n📄 Всего загружено: {len(documents)} документов")
    return documents


# ============================================================
# 2. ЧАНКИНГ (ДВЕ СТРАТЕГИИ)
# ============================================================

def chunk_by_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Стратегия 1: Фиксированный размер с перекрытием"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        
        # Не разрываем слова
        if end < text_len and chunk[-1] not in ' .,!?;:\n)»':
            last_space = chunk.rfind(' ')
            if last_space > chunk_size // 2:
                end = start + last_space
                chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start += chunk_size - overlap
    
    return chunks


def chunk_by_sentences(text: str, max_chunk_size: int = 500) -> List[str]:
    """Стратегия 2: По границам предложений"""
    # Разбиваем на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(sentence) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Разбиваем длинное предложение
            words = sentence.split()
            temp = ""
            for word in words:
                if len(temp) + len(word) + 1 < max_chunk_size:
                    temp += " " + word if temp else word
                else:
                    if temp:
                        chunks.append(temp.strip())
                    temp = word
            if temp:
                chunks.append(temp.strip())
        
        elif len(current_chunk) + len(sentence) + 1 < max_chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# ============================================================
# 3. ЭМБЕДДИНГИ (оптимизировано для CPU/12GB RAM)
# ============================================================

def get_embedding_model():
    """Загружает лёгкую модель для CPU"""
    print("\n🔧 Загрузка модели эмбеддингов...")
    # Модель: 384 dim, ~120MB RAM, хорошо для русского
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print(f"  ✓ Модель загружена (размерность: {model.get_sentence_embedding_dimension()})")
    return model


# ============================================================
# 4. FAISS ИНДЕКС
# ============================================================

class VectorIndex:
    """FAISS индекс для поиска по косинусному сходству"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []  # список словарей с метаданными
    
    def add(self, vectors: np.ndarray, chunks: List[Dict]):
        """Добавляет векторы в индекс"""
        # Нормализуем для косинусного сходства
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunks.extend(chunks)
    
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[float, Dict]]:
        """Ищет топ-k похожих чанков"""
        query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append((float(dist), self.chunks[idx]))
        
        return results


# ============================================================
# 5. SQLite МЕТАДАННЫЕ
# ============================================================

class MetadataDB:
    """Хранилище метаданных"""
    
    def __init__(self, db_path: str = "metadata.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                filename TEXT NOT NULL,
                strategy TEXT NOT NULL,
                chunk_index INTEGER
            )
        ''')
        conn.commit()
        conn.close()
    
    def clear(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks")
        conn.commit()
        conn.close()
    
    def add_chunk(self, text: str, filename: str, strategy: str, chunk_index: int) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chunks (text, filename, strategy, chunk_index)
            VALUES (?, ?, ?, ?)
        ''', (text, filename, strategy, chunk_index))
        chunk_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return chunk_id


# ============================================================
# 6. ОСНОВНАЯ ЛОГИКА
# ============================================================

def index_strategy(strategy_name: str, chunking_func, documents: List[Dict],
                   model, db: MetadataDB) -> Tuple[VectorIndex, Dict]:
    """Индексирует документы одной стратегией"""
    print(f"\n{'='*50}")
    print(f"📊 Стратегия: {strategy_name.upper()}")
    print(f"{'='*50}")
    
    all_chunks = []
    all_texts = []
    
    start = time.time()
    
    for doc in documents:
        chunks = chunking_func(doc['content'])
        print(f"  {doc['filename']}: {len(chunks)} чанков")
        
        for i, chunk in enumerate(chunks):
            chunk_id = db.add_chunk(chunk, doc['filename'], strategy_name, i)
            all_chunks.append({
                'id': chunk_id,
                'text': chunk,
                'filename': doc['filename'],
                'strategy': strategy_name
            })
            all_texts.append(chunk)
    
    print(f"  🔄 Генерация {len(all_texts)} эмбеддингов...")
    embeddings = model.encode(all_texts, show_progress_bar=True)
    
    vector_idx = VectorIndex(embeddings.shape[1])
    vector_idx.add(embeddings, all_chunks)
    
    elapsed = time.time() - start
    
    stats = {
        'chunk_count': len(all_chunks),
        'avg_size': sum(len(c['text']) for c in all_chunks) / len(all_chunks),
        'time': elapsed
    }
    
    print(f"  ✅ {stats['chunk_count']} чанков | Средний размер: {stats['avg_size']:.0f} | {stats['time']:.2f} сек")
    
    return vector_idx, stats


def search(query: str, vector_idx: VectorIndex, model, top_k: int = 3) -> List[Dict]:
    """Поиск по индексу"""
    query_vec = model.encode([query])[0]
    results = vector_idx.search(query_vec, top_k)
    
    return [
        {'score': score, 'text': chunk['text'], 'filename': chunk['filename'], 'strategy': chunk['strategy']}
        for score, chunk in results
    ]


def compare_strategies(strategies_stats: Dict):
    """Таблица сравнения"""
    print("\n" + "="*60)
    print("📊 СРАВНЕНИЕ СТРАТЕГИЙ ЧАНКИНГА")
    print("="*60)
    print(f"{'Стратегия':<12} {'Чанков':<10} {'Средний размер':<18} {'Время (сек)':<10}")
    print("-"*60)
    for name, s in strategies_stats.items():
        print(f"{name:<12} {s['chunk_count']:<10} {s['avg_size']:<18.0f} {s['time']:<10.2f}")
    print("="*60)


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("="*60)
    print("🚚 RAG индексация документов перевозчиков")
    print("="*60)
    
    # 1. Загрузка документов
    print("\n📂 Загрузка документов...")
    documents = load_documents("data/carriers")
    if not documents:
        print("❌ Нет документов")
        return
    
    # 2. Загрузка модели
    model = get_embedding_model()
    
    # 3. Подготовка БД
    db = MetadataDB()
    db.clear()
    
    # 4. Индексация двумя стратегиями
    strategies = {
        'fixed': chunk_by_fixed_size,
        'sentence': chunk_by_sentences
    }
    
    indices = {}
    stats = {}
    
    for name, func in strategies.items():
        idx, s = index_strategy(name, func, documents, model, db)
        indices[name] = idx
        stats[name] = s
    
    # 5. Сравнение
    compare_strategies(stats)
    
    # 6. Твои тестовые запросы
    test_queries = [
        "обязанности фрахтователя и фрахтовщика",
        "условия перевозки опасных грузов",
        "тариф Москва Санкт-Петербург",
        "стоимость доставки до 50 кг"
    ]
    
    print("\n" + "="*60)
    print("🔍 ТЕСТОВЫЕ ЗАПРОСЫ")
    print("="*60)
    
    for query in test_queries:
        print(f"\n📌 Запрос: '{query}'")
        print("-"*40)
        
        for strategy_name, idx in indices.items():
            results = search(query, idx, model, top_k=2)
            if results:
                print(f"\n  [{strategy_name.upper()}]")
                for r in results:
                    print(f"    Score: {r['score']:.4f} | {r['filename']}")
                    print(f"    {r['text'][:120]}...")
            else:
                print(f"\n  [{strategy_name.upper()}] Ничего не найдено")
    
    print("\n✅ Готово!")
    print("💾 FAISS индексы в памяти, метаданные в metadata.db")


if __name__ == "__main__":
    main()