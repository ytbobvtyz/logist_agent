#!/usr/bin/env python3
"""
Упрощенный тест локальной модели.
"""

import asyncio
import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(__file__))

async def test_basic():
    """Базовый тест инициализации."""
    try:
        from route_planner.enhanced_agent import EnhancedRoutePlannerAgent
        
        print("🧪 Тест 1: Создание агента с локальной моделью")
        # Используем временный файл для базы данных
        import tempfile
        import os
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db_path = temp_db.name
        temp_db.close()
        
        agent = EnhancedRoutePlannerAgent(
                model="llama3.2:3b",
                use_local=True,
                db_path=temp_db_path
            )
        
        print(f"✅ Агент создан: модель={agent.model}, локальная={agent.use_local}")
        print(f"✅ Системный промпт: {agent._get_system_prompt()[:100]}...")
        
        # Проверяем RAG
        if agent.rag_retriever:
            print("✅ RAG retriever загружен")
        else:
            print("⚠️ RAG retriever не загружен")
        
        # Проверяем MCP доступность
        print(f"📡 MCP доступен: {agent.state.mcp_available}")
        
        print("\n🧪 Тест 2: LLM клиент")
        from llm_client import create_llm_client
        
        # Тестируем создание клиентов
        local_client = create_llm_client("llama3.2:3b", use_local=True)
        print(f"✅ Локальный клиент создан: {local_client.__class__.__name__}")
        print(f"✅ Поддерживает tools: {local_client.supports_tools()}")
        
        # Тестируем базовый вызов LLM
        print("\n🧪 Тест 3: Базовый вызов LLM")
        messages = [
            {"role": "system", "content": "Ты помощник. Отвечай кратко."},
            {"role": "user", "content": "Привет! Как дела?"}
        ]
        
        try:
            response = await local_client.chat_completion(messages, max_tokens=50)
            print(f"✅ Ответ получен: {response.get('content', 'Нет ответа')[:100]}...")
        except Exception as e:
            print(f"⚠️ Ошибка вызова LLM: {e}")
        
        print("\n🎯 Базовый тест завершен!")
        
        # Очистка временного файла
        os.unlink(temp_db_path)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        
        # Очистка временного файла в случае ошибки
        if 'temp_db_path' in locals():
            try:
                os.unlink(temp_db_path)
            except:
                pass

async def test_openrouter_client():
    """Тестирование OpenRouter клиента."""
    print("\n\n🧪 Тест: OpenRouter клиент")
    print("=" * 50)
    
    try:
        from llm_client import create_llm_client
        
        # Требуется API ключ
        openrouter_client = create_llm_client("openrouter/auto", use_local=False)
        print(f"✅ OpenRouter клиент создан: {openrouter_client.__class__.__name__}")
        print(f"✅ Поддерживает tools: {openrouter_client.supports_tools()}")
        
        print("✅ OpenRouter клиент готов к работе (для реальных запросов нужен API ключ)")
        
    except Exception as e:
        print(f"⚠️ Ошибка OpenRouter клиента: {e}")

async def main():
    """Основная функция."""
    print("🧪 Тестирование интеграции локальной модели Ollama")
    print("=" * 60)
    
    await test_basic()
    await test_openrouter_client()
    
    print("\n✅ Все тесты завершены!")

if __name__ == "__main__":
    asyncio.run(main())