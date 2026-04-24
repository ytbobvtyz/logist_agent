#!/usr/bin/env python3
"""Проверка статуса RAG индекса"""

import os
from rag_with_reranking import RAGReranker

print("=" * 50)
print("Проверка RAG индекса")
print("=" * 50)

# Проверяем файлы
files = ['metadata.db', 'faiss_index', 'tfidf_vectorizer.pkl']
for f in files:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    print(f"{'✅' if exists else '❌'} {f}: {size} bytes")

print("\nЗагрузка RAGReranker...")
try:
    rag = RAGReranker(
        db_path="metadata.db",
        index_path="faiss_index",
        vectorizer_path="tfidf_vectorizer.pkl",
        threshold=0.3
    )
    
    # Пробуем поиск
    test_queries = [
        "коносамент",
        "ПЭК доставка",
        "стоимость перевозки"
    ]
    
    for query in test_queries:
        print(f"\n📌 Поиск: '{query}'")
        results = rag.search(query)
        print(f"   Найдено: {len(results)} фрагментов")
        for r in results:
            print(f"   - {r['filename']} (score: {r['score']:.3f})")
            print(f"     {r['text'][:100]}...")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()