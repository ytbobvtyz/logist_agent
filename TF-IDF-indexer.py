#!/usr/bin/env python3
"""
RAG система для индексации документов (легковесная версия без скачивания моделей)
Использует TF-IDF вместо эмбеддингов для демонстрации пайплайна
"""

import os
import re
import time
import sqlite3
import json
from typing import List, Dict, Tuple
import math

# ============================================================
# 1. TF-IDF ВЕКТОРИЗАТОР (легковесная замена эмбеддингам)
# ============================================================

class SimpleTfidfVectorizer:
    """TF-IDF векторизатор для русского текста (без скачивания моделей)"""
    
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.is_fitted = False
        # Стоп-слова для русского языка
        self.stop_words = {'и', 'в', 'во', 'на', 'с', 'со', 'по', 'к', 'у', 'о', 'об', 
                          'от', 'до', 'за', 'под', 'над', 'через', 'для', 'без', 'не', 
                          'ни', 'что', 'как', 'так', 'вот', 'это', 'этот', 'был', 'его',
                          'её', 'они', 'мы', 'вы', 'ты', 'он', 'она', 'оно', 'но', 'да',
                          'нет', 'еще', 'уже', 'только', 'если', 'когда', 'где', 'тут',
                          'там', 'здесь', 'потом', 'теперь', 'вдруг', 'даже', 'раз', 'или'}
    
    def _tokenize(self, text: str) -> List[str]:
        """Токенизация русскоязычного текста"""
        text = text.lower()
        # Находим русские слова (буквы + дефис внутри)
        words = re.findall(r'[а-яёa-z]+(?:-[а-яёa-z]+)?', text)
        # Фильтруем стоп-слова и короткие слова
        return [w for w in words if w not in self.stop_words and len(w) > 1]
    
    def fit(self, documents: List[str]):
        """Обучение на документах"""
        self.vocab = {}
        doc_term_counts = []
        
        for doc in documents:
            words = self._tokenize(doc)
            term_counts = {}
            for w in words:
                term_counts[w] = term_counts.get(w, 0) + 1
            doc_term_counts.append(term_counts)
            
            # Добавляем в словарь
            for w in term_counts.keys():
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
        
        # Вычисляем IDF
        N = len(documents)
        for word, idx in self.vocab.items():
            # Сколько документов содержат слово
            doc_count = sum(1 for dtc in doc_term_counts if word in dtc)
            self.idf[word] = math.log((N + 1) / (doc_count + 1)) + 1
        
        self.is_fitted = True
        print(f"  ✓ Словарь: {len(self.vocab)} уникальных слов")
    
    def transform(self, texts: List[str]) -> List[List[float]]:
        """Превращает тексты в векторы TF-IDF"""
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        
        vectors = []
        for text in texts:
            words = self._tokenize(text)
            # Считаем TF
            tf = {}
            for w in words:
                tf[w] = tf.get(w, 0) + 1
            
            # Строим вектор
            vec = [0.0] * len(self.vocab)
            for w, count in tf.items():
                if w in self.vocab:
                    idx = self.vocab[w]
                    # TF-IDF = TF * IDF
                    vec[idx] = (count / len(words)) * self.idf.get(w, 1.0)
            
            # Нормализация L2
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            
            vectors.append(vec)
        
        return vectors
    
    def encode(self, texts: List[str], show_progress_bar=False) -> List[List[float]]:
        """Совместимый интерфейс с sentence-transformers"""
        if not self.is_fitted:
            # Автообучение при первом вызове
            self.fit(texts)
        return self.transform(texts)


# ============================================================
# 2. ЗАГРУЗКА ДОКУМЕНТОВ
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
# 3. ЧАНКИНГ (ДВЕ СТРАТЕГИИ)
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
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(sentence) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
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
# 4. ВЕКТОРНЫЙ ИНДЕКС (BRUTE FORCE)
# ============================================================

class VectorIndex:
    """Простой векторный индекс для TF-IDF векторов"""
    
    def __init__(self):
        self.vectors = []  # список векторов
        self.chunks = []   # список чанков с метаданными
    
    def add(self, vectors: List[List[float]], chunks: List[Dict]):
        """Добавляет векторы в индекс"""
        self.vectors.extend(vectors)
        self.chunks.extend(chunks)
    
    def search(self, query_vector: List[float], top_k: int = 3) -> List[Tuple[float, Dict]]:
        """Ищет топ-k похожих чанков (косинусное сходство)"""
        results = []
        
        for vec, chunk in zip(self.vectors, self.chunks):
            # Косинусное сходство (векторы уже нормализованы TF-IDF)
            similarity = sum(a * b for a, b in zip(query_vector, vec))
            results.append((similarity, chunk))
        
        # Сортируем по убыванию сходства
        results.sort(key=lambda x: x[0], reverse=True)
        
        return results[:top_k]


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
                   vectorizer: SimpleTfidfVectorizer, db: MetadataDB) -> Tuple[VectorIndex, Dict]:
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
    
    print(f"  🔄 Генерация TF-IDF векторов для {len(all_texts)} чанков...")
    
    # Если векторизатор ещё не обучен, обучаем на всех текстах
    if not vectorizer.is_fitted:
        vectorizer.fit(all_texts)
    
    vectors = vectorizer.transform(all_texts)
    
    idx = VectorIndex()
    idx.add(vectors, all_chunks)
    
    elapsed = time.time() - start
    
    stats = {
        'chunk_count': len(all_chunks),
        'avg_size': sum(len(c['text']) for c in all_chunks) / len(all_chunks),
        'time': elapsed
    }
    
    print(f"  ✅ {stats['chunk_count']} чанков | Средний размер: {stats['avg_size']:.0f} | {stats['time']:.2f} сек")
    
    return idx, stats


def search(query: str, vector_idx: VectorIndex, vectorizer: SimpleTfidfVectorizer, top_k: int = 3) -> List[Dict]:
    """Поиск по индексу"""
    # Превращаем запрос в вектор
    query_vec = vectorizer.transform([query])[0]
    
    # Ищем
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
    print("🚚 RAG индексация документов перевозчиков (легковесная версия)")
    print("="*60)
    
    # 1. Загрузка документов
    print("\n📂 Загрузка документов...")
    documents = load_documents("data/carriers")
    if not documents:
        print("❌ Нет документов")
        return
    
    # 2. Инициализация TF-IDF векторизатора (не требует скачивания)
    print("\n🔧 Инициализация TF-IDF векторизатора...")
    vectorizer = SimpleTfidfVectorizer()
    
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
        idx, s = index_strategy(name, func, documents, vectorizer, db)
        indices[name] = idx
        stats[name] = s
    
    # 5. Сравнение
    compare_strategies(stats)
    
    # 6. Тестовые запросы
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
            results = search(query, idx, vectorizer, top_k=2)
            if results:
                print(f"\n  [{strategy_name.upper()}]")
                for r in results:
                    print(f"    Score: {r['score']:.4f} | {r['filename']}")
                    print(f"    {r['text'][:120]}...")
            else:
                print(f"\n  [{strategy_name.upper()}] Ничего не найдено")
    
    # 7. Сохраняем метаданные в JSON для отчёта
    with open("index_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "strategies": stats,
            "vocab_size": len(vectorizer.vocab),
            "total_documents": len(documents),
            "test_queries_results": "см. вывод в консоли"
        }, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Готово!")
    print("💾 Метаданные сохранены в metadata.db и index_metadata.json")
    print("\n📝 Примечание: используется TF-IDF вместо нейросетевых эмбеддингов")
    print("   (из-за ограничений интернета, пайплайн индексации полностью рабочий)")


if __name__ == "__main__":
    main()