#!/usr/bin/env python3
"""
RAG система для индексации документов перевозчиков
Единая стратегия: фиксированный размер чанка 500 символов, overlap 50
"""

import pickle
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
# 2. ЧАНКИНГ (фиксированный размер)
# ============================================================

def chunk_by_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Разбиение на чанки фиксированного размера с перекрытием"""
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


# ============================================================
# 3. ЭМБЕДДИНГИ
# ============================================================

def get_embedding_model():
    """Загружает лёгкую модель для CPU"""
    print("\n🔧 Загрузка модели эмбеддингов...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print(f"  ✓ Модель загружена (размерность: {model.get_embedding_dimension()})")
    return model


# ============================================================
# 4. FAISS ИНДЕКС С MAPPING
# ============================================================

class VectorIndex:
    """FAISS индекс с сохранением mapping к chunk_id"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunk_ids = []
    
    def add(self, vectors: np.ndarray, chunk_ids: List[int]):
        """Добавляет векторы в индекс"""
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunk_ids.extend(chunk_ids)
    
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[float, int]]:
        """Возвращает (score, chunk_id)"""
        query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunk_ids):
                results.append((float(dist), self.chunk_ids[idx]))
        
        return results
    
    def save(self, path: str):
        """Сохраняет индекс и mapping"""
        faiss.write_index(self.index, path)
        
        mapping_path = path + ".mapping"
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.chunk_ids, f)
        print(f"  💾 Индекс сохранён: {path}")
        print(f"  💾 Mapping сохранён: {mapping_path}")
    
    @classmethod
    def load(cls, path: str):
        """Загружает индекс и mapping"""
        index = cls(384)
        index.index = faiss.read_index(path)
        
        mapping_path = path + ".mapping"
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                index.chunk_ids = pickle.load(f)
        
        return index


# ============================================================
# 5. SQLite МЕТАДАННЫЕ
# ============================================================

class MetadataDB:
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
    
    def add_chunk(self, text: str, filename: str, chunk_index: int) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chunks (text, filename, chunk_index)
            VALUES (?, ?, ?)
        ''', (text, filename, chunk_index))
        chunk_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return chunk_id
    
    def get_chunk_by_id(self, chunk_id: int) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT text, filename FROM chunks WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {'text': row[0], 'filename': row[1]}
        return None
    
    def get_all_chunks(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, filename FROM chunks")
        rows = cursor.fetchall()
        conn.close()
        return [{'id': r[0], 'text': r[1], 'filename': r[2]} for r in rows]


# ============================================================
# 6. ОСНОВНАЯ ЛОГИКА
# ============================================================

def index_documents(documents: List[Dict], model, db: MetadataDB) -> VectorIndex:
    """Индексирует все документы одной стратегией"""
    print(f"\n{'='*50}")
    print("📊 ИНДЕКСАЦИЯ (fixed size, 500/50)")
    print(f"{'='*50}")
    
    all_texts = []
    chunk_ids = []
    
    start = time.time()
    
    for doc in documents:
        chunks = chunk_by_fixed_size(doc['content'])
        print(f"  {doc['filename']}: {len(chunks)} чанков")
        
        for i, chunk in enumerate(chunks):
            chunk_id = db.add_chunk(chunk, doc['filename'], i)
            chunk_ids.append(chunk_id)
            all_texts.append(chunk)
    
    print(f"  🔄 Генерация {len(all_texts)} эмбеддингов...")
    embeddings = model.encode(all_texts, show_progress_bar=True)
    
    vector_idx = VectorIndex(embeddings.shape[1])
    vector_idx.add(embeddings, chunk_ids)
    
    vector_idx.save("faiss_index")
    
    elapsed = time.time() - start
    
    print(f"  ✅ {len(chunk_ids)} чанков | Средний размер: {sum(len(c) for c in all_texts) / len(all_texts):.0f} | {elapsed:.2f} сек")
    
    return vector_idx


def search(query: str, vector_idx: VectorIndex, model, db: MetadataDB, top_k: int = 3) -> List[Dict]:
    """Поиск по индексу"""
    query_vec = model.encode([query])[0]
    query_vec = np.array(query_vec, dtype=np.float32).reshape(1, -1)
    
    results_with_ids = vector_idx.search(query_vec, top_k)
    
    results = []
    for score, chunk_id in results_with_ids:
        chunk_data = db.get_chunk_by_id(chunk_id)
        if chunk_data:
            results.append({
                'score': score,
                'text': chunk_data['text'],
                'filename': chunk_data['filename'],
                'chunk_id': chunk_id
            })
    
    return results


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("="*60)
    print("🚚 RAG индексация документов перевозчиков (единая стратегия)")
    print("="*60)
    
    # 1. Загрузка документов
    print("\n📂 Загрузка документов...")
    documents = load_documents("data/carriers")
    if not documents:
        print("❌ Нет документов")
        return
    
    # 2. Загрузка модели
    model = get_embedding_model()
    
    # 3. Подготовка БД (чистая)
    db = MetadataDB()
    db.clear()
    
    # 4. Индексация
    vector_idx = index_documents(documents, model, db)
    
    # 5. Тестовые запросы
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
        
        results = search(query, vector_idx, model, db, top_k=3)
        
        if results:
            for r in results:
                print(f"    Score: {r['score']:.4f} | {r['filename']}")
                print(f"    {r['text'][:150]}...")
                print()
        else:
            print("    ❌ Ничего не найдено")
    
    print("\n✅ Готово!")
    print("💾 FAISS индекс: faiss_index + faiss_index.mapping")
    print("💾 Метаданные: metadata.db")


if __name__ == "__main__":
    main()