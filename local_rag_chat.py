#!/usr/bin/env python3
"""
Локальное RAG приложение на Streamlit.
Использует RAGRetriever (sentence-transformers + FAISS).
"""

import os
import sys
import asyncio
from typing import List, Dict

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from openai import AsyncOpenAI

# Импортируем ваш RAG
try:
    from rag_retriever import RAGRetriever
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"⚠️ Ошибка импорта RAGRetriever: {e}")


class LocalRAGChat:
    """Чат с использованием RAGRetriever."""
    
    def __init__(self):
        self.rag = None
        self.llm = None
        self.init_rag()
        self.init_llm()
    
    def init_rag(self):
        """Инициализирует RAGRetriever."""
        if not RAG_AVAILABLE:
            st.error("❌ RAGRetriever не доступен")
            return
        
        try:
            # Проверяем существование файлов
            if not os.path.exists("faiss_index"):
                st.error("❌ FAISS индекс не найден. Запустите сначала indexer.py")
                return
            
            if not os.path.exists("metadata.db"):
                st.error("❌ metadata.db не найден")
                return
            
            # Инициализируем RAGRetriever
            with st.spinner("🔄 Загрузка RAG индекса..."):
                self.rag = RAGRetriever(
                    db_path="metadata.db",
                    index_path="faiss_index",
                    model_name="paraphrase-multilingual-MiniLM-L12-v2"
                )
            
            # Проверяем статистику
            stats = self.rag.get_index_stats()
            if stats.get('vectors_count', 0) > 0:
                st.success(f"✅ RAG загружен: {stats['vectors_count']} векторов, {stats.get('total_chunks', 0)} чанков")
            else:
                st.warning("⚠️ Индекс пуст")
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки RAG: {e}")
            self.rag = None
    
    def init_llm(self):
        """Инициализирует Ollama."""
        try:
            self.llm = AsyncOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                timeout=90.0
            )
            st.success("✅ Ollama подключена")
        except Exception as e:
            st.error(f"❌ Ошибка Ollama: {e}")
            self.llm = None
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Поиск через RAGRetriever."""
        if not self.rag:
            return []
        
        try:
            results = self.rag.search(query, top_k=top_k)
            
            # Преобразуем формат для единообразия
            formatted = []
            for r in results:
                formatted.append({
                    'text': r.get('text', ''),
                    'filename': r.get('filename', 'unknown'),
                    'score': r.get('score', 0)
                })
            
            return formatted
        except Exception as e:
            st.error(f"Ошибка поиска: {e}")
            return []
    
    def build_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Строит промпт с контекстом."""
        if not chunks:
            return f"""Ты эксперт по логистике и транспортным документам. 
Ответь на вопрос, используя свои профессиональные знания.

Вопрос: {query}

Ответ:"""
        
        # Формируем контекст
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')[:800]
            filename = chunk.get('filename', 'unknown')
            score = chunk.get('score', 0)
            
            context_parts.append(f"[{i}] {filename} (релевантность: {score:.3f})\n{text}")
            sources.append(filename)
        
        context = "\n\n---\n\n".join(context_parts)
        
        return f"""Ты эксперт по логистике. Отвечай НА ОСНОВЕ предоставленных документов.

## ДОКУМЕНТЫ (от наиболее релевантных к менее релевантным):
{context}

## ВОПРОС: {query}

## ПРАВИЛА:
1. Используй ТОЛЬКО информацию из документов
2. Обязательно указывай источник (название документа)
3. Если информации нет - честно скажи "В документах не найдено"
4. Отвечай по-русски, развернуто

## ОТВЕТ:"""
    
    async def generate_answer(self, query: str, chunks: List[Dict]) -> str:
        """Генерирует ответ через Ollama."""
        if not self.llm:
            return "❌ Ollama не доступна. Запустите: ollama serve"
        
        prompt = self.build_prompt(query, chunks)
        
        try:
            response = await self.llm.chat.completions.create(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            # Добавляем информацию об источниках
            if chunks:
                # Группируем источники
                sources_count = {}
                for chunk in chunks:
                    filename = chunk.get('filename', 'unknown')
                    sources_count[filename] = sources_count.get(filename, 0) + 1
                
                sources_text = ", ".join([f"{name} ({count})" for name, count in sources_count.items()])
                avg_score = sum(c.get('score', 0) for c in chunks) / len(chunks)
                
                return f"""{answer}

---
📚 **Источники:** {sources_text}
📊 **Средняя релевантность:** {avg_score:.2f}"""
            else:
                return f"""{answer}

---
⚠️ **Информация не найдена в документах**"""
        
        except Exception as e:
            return f"❌ Ошибка генерации: {e}"
    
    def chat_sync(self, query: str) -> str:
        """Синхронная обертка."""
        # Поиск через RAG
        chunks = self.search(query)
        
        if chunks:
            st.info(f"📄 Найдено {len(chunks)} релевантных фрагментов")
            
            # Показываем детали поиска в раскрывающемся блоке
            with st.expander("🔍 Детали поиска (нажмите чтобы раскрыть)"):
                for i, chunk in enumerate(chunks, 1):
                    score = chunk.get('score', 0)
                    filename = chunk.get('filename', 'unknown')
                    text_preview = chunk.get('text', '')[:200]
                    
                    st.markdown(f"**{i}. {filename}** (score: {score:.3f})")
                    st.caption(text_preview + "...")
                    st.markdown("---")
        else:
            st.warning("⚠️ Релевантных фрагментов не найдено")
        
        # Генерация ответа
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.generate_answer(query, chunks))
            loop.close()
            return result
        except Exception as e:
            return f"❌ Ошибка: {e}"


def main():
    st.set_page_config(
        page_title="Local RAG Chat",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Local RAG Assistant")
    st.markdown("**Llama 3.2 + RAG (sentence-transformers + FAISS)**")
    
    # Инициализация
    if 'agent' not in st.session_state:
        st.session_state.agent = LocalRAGChat()
        st.session_state.messages = []
    
    agent = st.session_state.agent
    
    # Боковая панель
    with st.sidebar:
        st.markdown("## 📊 Статус")
        
        # Статус RAG
        if agent.rag:
            stats = agent.rag.get_index_stats()
            st.success("✅ RAG активен")
            st.metric("Векторов в индексе", stats.get('vectors_count', 0))
            st.metric("Всего чанков", stats.get('total_chunks', 0))
            st.metric("Размерность", stats.get('dimension', 0))
        else:
            st.error("❌ RAG не загружен")
            st.info("Запустите: python indexer.py")
        
        st.markdown("---")
        
        # Статус Ollama
        if agent.llm:
            st.success("✅ Ollama: llama3.2:3b")
        else:
            st.error("❌ Ollama не доступна")
            st.code("ollama serve\nollama pull llama3.2:3b")
        
        st.markdown("---")
        st.markdown("## 💡 Примеры")
        
        examples = [
            "Что такое коносамент?",
            "Стоимость доставки ПЭК Москва Казань",
            "Максимальный вес посылки СДЭК",
            "Обязанности фрахтователя",
            "API ПЭК расчет стоимости"
        ]
        
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()
        
        st.markdown("---")
        
        if st.button("🗑️ Очистить историю", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Отображение чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Поле ввода
    if prompt := st.chat_input("Задайте вопрос о логистике и документах..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Поиск в документах..."):
                response = agent.chat_sync(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main() 