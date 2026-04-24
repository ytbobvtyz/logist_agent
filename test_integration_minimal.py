#!/usr/bin/env python3
"""
Минимальный тест интеграции локальной модели.
Проверяет только создание объектов и базовые вызовы.
"""

import sys
import os
import asyncio

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(__file__))

async def test_llm_client():
    """Тестирует создание LLM клиентов."""
    print("🧪 Тест 1: LLM клиенты")
    print("=" * 50)
    
    try:
        from llm_client import create_llm_client, OpenRouterClient, OllamaClient
        
        # Тестируем создание клиента Ollama
        local_client = create_llm_client("llama3.2:3b", use_local=True)
        print(f"✅ Локальный клиент создан: {local_client.__class__.__name__}")
        print(f"  - Модель: {local_client.model}")
        print(f"  - Поддерживает tools: {local_client.supports_tools()}")
        
        # Тестируем создание клиента OpenRouter
        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_client = create_llm_client("openrouter/auto", use_local=False)
            print(f"✅ OpenRouter клиент создан: {openrouter_client.__class__.__name__}")
            print(f"  - Модель: {openrouter_client.model}")
            print(f"  - Поддерживает tools: {openrouter_client.supports_tools()}")
        else:
            print("⚠️ OPENROUTER_API_KEY не установлен, пропускаем тест OpenRouter")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка создания LLM клиентов: {e}")
        return False

async def test_agent_initialization():
    """Тестирует создание агента."""
    print("\n🧪 Тест 2: Инициализация агента")
    print("=" * 50)
    
    try:
        from route_planner.enhanced_agent import EnhancedRoutePlannerAgent
        
        # Используем временный файл для базы
        import tempfile
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db_path = temp_db.name
        temp_db.close()
        
        # Создаем агента с локальной моделью
        agent = EnhancedRoutePlannerAgent(
            model="llama3.2:3b",
            use_local=True,
            db_path=temp_db_path
        )
        
        print(f"✅ Агент создан: {agent.model} (локальная: {agent.use_local})")
        print(f"✅ Текущий диалог: #{agent.current_conversation.id}")
        
        # Проверяем системный промпт
        prompt = agent._get_system_prompt()
        print(f"✅ Системный промпт получен ({len(prompt)} символов)")
        print(f"  - Начинается с: {prompt[:100]}...")
        
        # Проверяем RAG (может быть None)
        print(f"✅ RAG retriever: {'загружен' if agent.rag_retriever else 'не загружен'}")
        
        # Проверяем MCP доступность
        print(f"✅ MCP доступен: {agent.state.mcp_available}")
        
        # Очистка
        os.unlink(temp_db_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка инициализации агента: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mcp_handling():
    """Тестирует обработку MCP для локальной модели."""
    print("\n🧪 Тест 3: Обработка MCP ограничений")
    print("=" * 50)
    
    try:
        from route_planner.enhanced_agent import EnhancedRoutePlannerAgent
        
        # Используем временный файл для базы
        import tempfile
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db_path = temp_db.name
        temp_db.close()
        
        # Создаем агента с локальной моделью
        agent = EnhancedRoutePlannerAgent(
            model="llama3.2:3b",
            use_local=True,
            db_path=temp_db_path
        )
        
        # Проверяем, что _process_with_mcp возвращает предупреждение
        # Создаем простой callback
        async def dummy_callback(msg):
            print(f"  ⚠️ Callback: {msg}")
        
        result = await agent._process_with_mcp(
            user_message="Рассчитай маршрут Москва-Питер",
            context="",
            conversation_id=1,
            callback=dummy_callback
        )
        
        print(f"✅ _process_with_mcp вернул результат ({len(result)} символов)")
        print(f"  - Начинается с: {result[:100]}...")
        
        # Проверяем, что не было вызовов MCP
        print(f"✅ Вызовов MCP инструментов: {len(agent.state.mcp_calls)}")
        
        # Очистка
        os.unlink(temp_db_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка обработки MCP: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_rag_handling():
    """Тестирует обработку RAG для локальной модели."""
    print("\n🧪 Тест 4: Обработка RAG запросов")
    print("=" * 50)
    
    try:
        from route_planner.enhanced_agent import EnhancedRoutePlannerAgent
        
        # Используем временный файл для базы
        import tempfile
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db_path = temp_db.name
        temp_db.close()
        
        # Создаем агента с локальной моделью
        agent = EnhancedRoutePlannerAgent(
            model="llama3.2:3b",
            use_local=True,
            db_path=temp_db_path
        )
        
        # Проверяем метод search_with_rag
        if agent.rag_retriever:
            results = agent.search_with_rag("тарифы перевозки", top_k=2)
            print(f"✅ RAG поиск вернул {len(results)} результатов")
        else:
            print("⚠️ RAG retriever не загружен, пропускаем поиск")
        
        # Проверяем метод _should_use_rag
        test_queries = [
            ("Какие тарифы у ПЭК?", True),
            ("Привет как дела?", False),
            ("Расстояние между городами", False),  # MCP query, но в локальном режиме
        ]
        
        for query, expected_rag in test_queries:
            should_use = agent._should_use_rag(query)
            print(f"  - '{query}' -> RAG: {should_use} (ожидалось: {expected_rag})")
        
        # Очистка
        os.unlink(temp_db_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка обработки RAG: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Основная функция тестирования."""
    print("🧪 Тестирование интеграции локальной модели Ollama")
    print("=" * 60)
    
    results = []
    
    results.append(await test_llm_client())
    results.append(await test_agent_initialization())
    results.append(await test_mcp_handling())
    results.append(await test_rag_handling())
    
    # Итог
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'=' * 60}")
    print(f"🎯 ИТОГ: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("✅ Все тесты пройдены успешно!")
    else:
        print(f"⚠️ {total - passed} тестов не пройдено")

if __name__ == "__main__":
    asyncio.run(main())