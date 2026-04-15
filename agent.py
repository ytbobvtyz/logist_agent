#!/usr/bin/env python3
"""
Простой агент для тестирования RAG функциональности.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

# Загружаем переменные окружения
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


class LogistAgent:
    """Агент для логистических вопросов с поддержкой RAG."""
    
    def __init__(self):
        """Инициализация агента."""
        self.rag_enabled = False
        self.retriever = None
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=60.0
        )
        
        # Пытаемся загрузить RAG retriever
        try:
            from rag_retriever import RAGRetriever
            self.retriever = RAGRetriever()
            print("✅ RAG Retriever загружен")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить RAG Retriever: {e}")
    
    def ask_without_rag(self, user_input: str) -> str:
        """Режим без RAG (прямой вызов LLM)."""
        try:
            response = self.client.chat.completions.create(
                model="openrouter/auto",
                messages=[
                    {
                        "role": "system", 
                        "content": "Ты помощник-логист. Отвечай на вопросы о логистике и перевозках."
                    },
                    {"role": "user", "content": user_input}
                ],
                max_tokens=512
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Ошибка LLM: {e}"
    
    def ask_with_rag(self, user_input: str) -> str:
        """Режим с RAG: поиск → объединение → LLM."""
        if not self.retriever:
            return self.ask_without_rag(user_input)
        
        # Поиск релевантных чанков
        chunks = self.retriever.search(user_input, top_k=3)
        
        # Объединение в промпт
        prompt = self._build_rag_prompt(user_input, chunks)
        
        try:
            response = self.client.chat.completions.create(
                model="openrouter/auto",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Ошибка LLM с RAG: {e}"
    
    def _build_rag_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Объединяет чанки с вопросом в промпт."""
        if not chunks:
            return f"""Ты помощник-логист. Ответь на вопрос:

{query}

💡 Используй свои знания, так как релевантная информация в документах не найдена."""
        
        context = "\n".join([
            f"📄 [{chunk['filename']}] {chunk['text']}"
            for chunk in chunks
        ])
        
        return f"""Ты помощник-логист.

## Инструкция:
- Если информация есть в документах ниже — используй её и отметь 📄
- Если информации нет — используй свои знания и отметь 💡
- Всегда указывай источник

## Документы с релевантной информацией:
{context}

## Вопрос пользователя:
{query}

## Твой ответ:
"""
    
    def ask(self, user_input: str) -> str:
        """Главный метод с выбором режима."""
        if self.rag_enabled:
            return self.ask_with_rag(user_input)
        else:
            return self.ask_without_rag(user_input)


# Демонстрация работы
if __name__ == "__main__":
    agent = LogistAgent()
    
    # Тестовые вопросы
    test_questions = [
        "Сколько стоит доставка груза 50 кг из Москвы в Санкт-Петербург у ПЭК?",
        "Какой максимальный вес принимает СДЭК для посылки?",
        "Что такое логистика?"
    ]
    
    for question in test_questions:
        print(f"\n🔍 Вопрос: {question}")
        print("-" * 50)
        
        # Без RAG
        agent.rag_enabled = False
        answer_without = agent.ask(question)
        print(f"❌ Без RAG:\n{answer_without[:200]}...")
        
        # С RAG
        agent.rag_enabled = True
        answer_with = agent.ask(question)
        print(f"✅ С RAG:\n{answer_with[:200]}...")
        
        print("-" * 50)