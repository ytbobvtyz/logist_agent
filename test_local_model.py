#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы локальной модели Ollama.
"""

import asyncio
import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(__file__))

async def test_local_model():
    """Тестирует работу локальной модели."""
    from route_planner.enhanced_agent import EnhancedRoutePlannerAgent
    
    print("🧪 Тест 1: Локальная модель с RAG")
    print("=" * 50)
    
    try:
        # Инициализация агента с локальной моделью
        agent = EnhancedRoutePlannerAgent(
            model="llama3.2:3b",
            use_local=True,
            db_path="conversations_test.db"
        )
        
        print(f"✅ Агент инициализирован: {agent.model} (локальная: {agent.use_local})")
        
        # Проверяем подключение MCP (должно быть недоступно)
        mcp_available = await agent.connect_mcp()
        print(f"📡 MCP доступен: {mcp_available} (ожидается: False)")
        
        # Проверяем RAG
        if agent.rag_retriever:
            print("✅ RAG retriever загружен")
            
            # Тестируем поиск
            test_query = "тарифы перевозки"
            results = agent.search_with_rag(test_query, top_k=2)
            print(f"🔍 RAG поиск '{test_query}': найдено {len(results)} результатов")
        else:
            print("⚠️ RAG retriever не загружен")
        
        # Тестируем обработку сообщений
        print("\n🧪 Тест 2: Обработка запросов")
        print("=" * 50)
        
        test_messages = [
            "Привет! Как дела?",
            "Какие тарифы у ПЭК?",
            "Расстояние от Москвы до Санкт-Петербурга"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n{i}. Вопрос: {message}")
            print("-" * 30)
            
            try:
                response = await agent.process_message(message)
                print(f"Ответ: {response[:200]}...")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
        
        # Тестируем ограничения MCP
        print("\n🧪 Тест 3: Проверка ограничений MCP")
        print("=" * 50)
        
        mcp_query = "Рассчитай маршрут Москва-Питер"
        print(f"Запрос: {mcp_query}")
        
        response = await agent.process_message(mcp_query)
        print(f"Ответ: {response[:300]}...")
        
        # Проверяем, что нет вызовов MCP инструментов
        print(f"\n📊 Вызовов MCP инструментов: {len(agent.state.mcp_calls)}")
        
        print("\n✅ Все тесты завершены!")
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()

async def test_openrouter_model():
    """Тестирует работу OpenRouter модели для сравнения."""
    from route_planner.enhanced_agent import EnhancedRoutePlannerAgent
    
    print("\n\n🧪 Тест: OpenRouter модель")
    print("=" * 50)
    
    try:
        # Инициализация агента с OpenRouter
        agent = EnhancedRoutePlannerAgent(
            model="openrouter/auto",
            use_local=False,
            db_path="conversations_test2.db"
        )
        
        print(f"✅ Агент инициализирован: {agent.model} (локальная: {agent.use_local})")
        
        # Подключаем MCP
        mcp_available = await agent.connect_mcp()
        print(f"📡 MCP доступен: {mcp_available}")
        
        # Тестируем простой запрос
        response = await agent.process_message("Привет!")
        print(f"\nПростой запрос: {response[:200]}...")
        
        print("\n✅ Тест OpenRouter завершен!")
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании OpenRouter: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Основная функция тестирования."""
    print("🧪 Тестирование интеграции локальной модели Ollama")
    print("=" * 60)
    
    # Проверяем наличие OLLAMA_URL в окружении
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    print(f"OLLAMA_URL: {ollama_url}")
    
    choice = input("\nВыберите тест:\n1. Локальная модель (Ollama)\n2. OpenRouter модель\n3. Оба\nВаш выбор (1-3): ").strip()
    
    if choice in ["1", "3"]:
        await test_local_model()
    
    if choice in ["2", "3"]:
        await test_openrouter_model()
    
    print("\n🎯 Тестирование завершено!")

if __name__ == "__main__":
    asyncio.run(main())