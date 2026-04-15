#!/usr/bin/env python3
"""
TF-IDF векторизатор для русского текста (без внешних зависимостей)
"""

import re
import math
from typing import List, Dict


class SimpleTfidfVectorizer:
    """TF-IDF векторизатор для русского текста"""
    
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.is_fitted = False
        self.stop_words = {
            'и', 'в', 'во', 'на', 'с', 'со', 'по', 'к', 'у', 'о', 'об', 
            'от', 'до', 'за', 'под', 'над', 'через', 'для', 'без', 'не', 
            'ни', 'что', 'как', 'так', 'вот', 'это', 'этот', 'был', 'его',
            'её', 'они', 'мы', 'вы', 'ты', 'он', 'она', 'оно', 'но', 'да',
            'нет', 'еще', 'уже', 'только', 'если', 'когда', 'где', 'тут',
            'там', 'здесь', 'потом', 'теперь', 'вдруг', 'даже', 'раз', 'или',
            'при', 'из', 'за', 'над', 'под', 'про', 'без', 'для', 'через'
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Токенизация русскоязычного текста"""
        text = text.lower()
        words = re.findall(r'[а-яёa-z]+(?:-[а-яёa-z]+)?', text)
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
            
            for w in term_counts.keys():
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
        
        # Вычисляем IDF
        N = len(documents)
        for word, idx in self.vocab.items():
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
                    vec[idx] = (count / len(words)) * self.idf.get(w, 1.0)
            
            # Нормализация L2
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            
            vectors.append(vec)
        
        return vectors
    
    def encode(self, texts: List[str], show_progress_bar=False) -> List[List[float]]:
        """Совместимый интерфейс"""
        if not self.is_fitted:
            self.fit(texts)
        return self.transform(texts)