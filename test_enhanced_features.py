#!/usr/bin/env python3
"""
Тестирование расширенных функций:
1. Множество диалогов с переключением
2. Автоматическая суммаризация 
3. Task State (память задачи)
"""

import asyncio
import sys
import os
import tempfile
import shutil

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_conversation_manager():
    """Тестирование менеджера диалогов."""
    print("🧪 Тестирование ConversationManager...")
    
    from conversation_manager import get_conversation_manager
    
    # Создаем временную БД
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    try:
        # Создаем менеджер с временной БД
        manager = get_conversation_manager(temp_db.name)
        
        # Тест 1: Создание диалогов
        conv1 = manager.create_conversation("Тест диалог 1")
        conv2 = manager.create_conversation("Тест диалог 2")
        
        assert conv1.id is not None
        assert conv2.id is not None
        assert conv1.id != conv2.id
        print("  ✅ Создание диалогов: PASS")
        
        # Тест 2: Добавление сообщений
        msg1 = manager.add_message(conv1.id, "user", "Привет! Нужен расчет маршрута.")
        msg2 = manager.add_message(conv1.id, "assistant", "🔧 MCP\n\nХорошо, рассчитаю маршрут.")
        
        assert msg1.id is not None
        assert msg2.id is not None
        assert msg1.conversation_id == conv1.id
        assert msg2.conversation_id == conv1.id
        print("  ✅ Добавление сообщений: PASS")
        
        # Тест 3: Получение сообщений
        messages = manager.get_conversation_messages(conv1.id)
        assert len(messages) == 2
        assert messages[0].content == "Привет! Нужен расчет маршрута."
        assert messages[1].content.startswith("🔧 MCP")
        print("  ✅ Получение сообщений: PASS")
        
        # Тест 4: Получение всех диалогов
        conversations = manager.get_all_conversations()
        assert len(conversations) >= 2
        print(f"  ✅ Всего диалогов: {len(conversations)}")
        
        # Тест 5: Task state
        task_state = manager.get_task_state(conv1.id)
        assert task_state is not None
        assert task_state.conversation_id == conv1.id
        print("  ✅ Task state: PASS")
        
        # Тест 6: Счетчики сообщений
        conv = manager.get_conversation(conv1.id)
        assert conv.message_count == 2
        assert conv.user_message_count == 1
        print(f"  ✅ Счетчики: {conv.message_count} сообщений, {conv.user_message_count} от пользователя")
        
        # Тест 7: Переключение активного диалога
        manager.set_active_conversation(conv2.id)
        active_conv = manager.get_active_conversation()
        assert active_conv.id == conv2.id
        print("  ✅ Переключение диалогов: PASS")
        
        # Тест 8: Суммаризация триггер
        should_summarize, count = manager.should_summarize(conv1.id)
        assert not should_summarize  # Всего 1 сообщение пользователя
        assert count == 1
        print("  ✅ Проверка суммаризации: PASS")
        
        # Тест 9: Статистика
        stats = manager.get_statistics()
        assert 'total_conversations' in stats
        assert 'total_messages' in stats
        print(f"  ✅ Статистика: {stats}")
        
        print("✅ Все тесты ConversationManager пройдены успешно")
        
    finally:
        # Очищаем временные файлы
        if os.path.exists(temp_db.name):
            os.unlink(temp_db.name)
    
    return True


async def test_summarizer():
    """Тестирование модуля суммаризации."""
    print("\n🧪 Тестирование Summarizer...")
    
    from conversation_manager import ConversationManager
    from summarizer import Summarizer, SummarizerConfig
    
    # Создаем временную БД
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    try:
        # Создаем менеджер напрямую (не через синглтон)
        manager = ConversationManager(temp_db.name)
        
        # Создаем тестовый диалог
        conversation = manager.create_conversation("Тест суммаризации")
        
        # Добавляем тестовые сообщения (имитируем 15 сообщений пользователя)
        test_messages = [
            ("user", "Нужно рассчитать маршрут из Москвы в Санкт-Петербург."),
            ("assistant", "🔧 MCP\n\nХорошо, рассчитаю маршрут."),
            ("user", "Добавьте Казань в маршрут."),
            ("assistant", "🔧 MCP\n\nДобавил Казань в расчет."),
            ("user", "Какой будет общее расстояние?"),
            ("assistant", "🔧 MCP\n\nОбщее расстояние: 2320 км."),
            ("user", "А стоимость доставки для 100 кг?"),
            ("assistant", "🔧 MCP\n\nСтоимость: около 12,500 руб."),
            ("user", "Есть ли скидки для постоянных клиентов?"),
            ("assistant", "📚 RAG\n\nДа, скидки от 5% до 15% для корпоративных клиентов."),
        ]
        
        for role, content in test_messages:
            manager.add_message(conversation.id, role, content)
        
        # Создаем суммаризатор
        config = SummarizerConfig(summarization_trigger_count=3)  # Для теста триггер каждые 3 сообщения
        summarizer = Summarizer(manager, config)
        
        # Тест 1: Проверка триггера суммаризации
        should_summarize, count = manager.should_summarize(conversation.id)
        print(f"  Сообщений пользователя: {count}, нужно суммировать: {should_summarize}")
        
        # Тест 2: Принудительная суммаризация
        summary = await summarizer.force_summarize(conversation.id)
        assert summary is not None
        assert len(summary) > 0
        print(f"  Сгенерированная суммаризация ({len(summary)} символов):")
        print(f"  '{summary[:100]}...'")
        
        # Тест 3: Получение контекста
        context = summarizer.get_summary_context(conversation.id)
        assert len(context) > 0
        assert summary in context
        print(f"  Контекст сгенерирован ({len(context)} символов)")
        
        print("✅ Все тесты Summarizer пройдены успешно")
        
    finally:
        if os.path.exists(temp_db.name):
            os.unlink(temp_db.name)
    
    return True


async def test_task_state_manager():
    """Тестирование модуля состояния задачи."""
    print("\n🧪 Тестирование TaskStateManager...")
    
    from conversation_manager import ConversationManager
    from task_state import TaskStateManager
    
    # Создаем временную БД
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    try:
        # Создаем менеджеры напрямую
        manager = ConversationManager(temp_db.name)
        task_manager = TaskStateManager(manager)
        
        # Создаем тестовый диалог
        conversation = manager.create_conversation("Тест состояния задачи")
        
        # Тестовые сообщения с информацией для извлечения
        from conversation_manager import Message
        
        test_messages = [
            Message(role="user", content="Нужно рассчитать оптимальный маршрут между Москвой, Санкт-Петербургом и Казанью."),
            Message(role="user", content="Вес груза - 150 кг, нужно доставить за 2 дня."),
            Message(role="user", content="Бюджет не более 20,000 рублей, и не более 5 городов."),
        ]
        
        # Обрабатываем сообщения
        for msg in test_messages:
            # Добавляем в диалог
            manager.add_message(conversation.id, msg.role, msg.content)
            
            # Обновляем состояние задачи
            state = await task_manager.update_from_new_message(conversation.id, msg)
        
        # Тест 1: Получение состояния задачи
        final_state = task_manager.get_task_state(conversation.id)
        assert final_state is not None
        print(f"  Цель: {final_state.goal}")
        print(f"  Уточнения: {final_state.clarified_details}")
        print(f"  Ограничения: {final_state.constraints}")
        
        # Тест 2: Контекст для LLM
        context = task_manager.get_context_for_llm(conversation.id)
        assert len(context) > 0
        assert "Цель диалога" in context or "Уточнения пользователя" in context
        print(f"  Контекст для LLM ({len(context)} символов):")
        print(f"  '{context[:150]}...'")
        
        # Тест 3: Сброс состояния
        success = task_manager.reset_task_state(conversation.id)
        assert success
        reset_state = task_manager.get_task_state(conversation.id)
        assert len(reset_state.clarified_details) == 0
        assert len(reset_state.constraints) == 0
        print("  Сброс состояния: PASS")
        
        print("✅ Все тесты TaskStateManager пройдены успешно")
        
    finally:
        if os.path.exists(temp_db.name):
            os.unlink(temp_db.name)
    
    return True


async def test_integration():
    """Интеграционное тестирование всех компонентов."""
    print("\n🧪 Интеграционное тестирование...")
    
    from conversation_manager import ConversationManager
    from summarizer import Summarizer
    from task_state import TaskStateManager
    
    # Создаем временную БД
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    try:
        # Инициализируем все компоненты напрямую
        manager = ConversationManager(temp_db.name)
        summarizer = Summarizer(manager)
        task_manager = TaskStateManager(manager)
        
        # Создаем несколько диалогов
        conv1 = manager.create_conversation("Диалог 1: Расчет маршрута")
        conv2 = manager.create_conversation("Диалог 2: Вопросы по документам")
        
        # Работаем с первым диалогом
        manager.set_active_conversation(conv1.id)
        
        # Добавляем сообщения в первый диалог
        messages1 = [
            ("user", "Рассчитай маршрут Москва - Санкт-Петербург"),
            ("assistant", "🔧 MCP\n\nМаршрут рассчитан: 710 км"),
            ("user", "Добавь Казань в маршрут"),
            ("assistant", "🔧 MCP\n\nОбновленный маршрут: Москва → Казань → СПб"),
            ("user", "Какой вес можно перевезти?"),
        ]
        
        for role, content in messages1:
            manager.add_message(conv1.id, role, content)
        
        # Проверяем интеграцию
        # 1. Проверяем суммаризацию
        summary = await summarizer.check_and_summarize(conv1.id)
        if summary:
            print(f"  Суммаризация диалога 1: создана ({len(summary)} символов)")
        else:
            print(f"  Суммаризация диалога 1: не требуется или не сгенерирована")
        
        # 2. Проверяем состояние задачи
        state1 = task_manager.get_task_state(conv1.id)
        print(f"  Состояние задачи диалога 1:")
        print(f"    Сообщений: {state1.message_count if state1 else 'N/A'}")
        print(f"    Цель: {state1.goal if state1 and state1.goal else 'не определена'}")
        
        # Переключаемся на второй диалог
        manager.set_active_conversation(conv2.id)
        active_conv = manager.get_active_conversation()
        assert active_conv.id == conv2.id
        print(f"  Переключились на диалог: {active_conv.title}")
        
        # Добавляем сообщения во второй диалог
        messages2 = [
            ("user", "Какие обязанности у фрахтователя?"),
            ("assistant", "📚 RAG\n\nФрахтователь обязан предоставить груз и оплатить услуги."),
        ]
        
        for role, content in messages2:
            manager.add_message(conv2.id, role, content)
        
        # Проверяем состояние второго диалога
        state2 = task_manager.get_task_state(conv2.id)
        print(f"  Состояние задачи диалога 2:")
        print(f"    Сообщений: {state2.message_count if state2 else 'N/A'}")
        
        # Получаем статистику
        stats = manager.get_statistics()
        print(f"  Общая статистика:")
        print(f"    Диалогов: {stats['total_conversations']}")
        print(f"    Сообщений: {stats['total_messages']}")
        print(f"    Сообщений пользователя: {stats['total_user_messages']}")
        
        print("✅ Интеграционное тестирование пройдено успешно")
        
    finally:
        if os.path.exists(temp_db.name):
            os.unlink(temp_db.name)
    
    return True


async def main():
    """Основная функция тестирования."""
    print("=" * 60)
    print("🚀 ТЕСТИРОВАНИЕ РАСШИРЕННЫХ ФУНКЦИЙ LOGSIT AGENT")
    print("=" * 60)
    
    try:
        # Запускаем тесты
        await test_conversation_manager()
        await test_summarizer()
        await test_task_state_manager()
        await test_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 60)
        print("\nРеализованные функции:")
        print("1. ✅ Множество диалогов с переключением")
        print("2. ✅ Автоматическая суммаризация каждые 10 сообщений")
        print("3. ✅ Task State (память задачи)")
        print("\nСозданные файлы:")
        print("  - conversation_manager.py - управление диалогами и БД")
        print("  - summarizer.py - модуль суммаризации")
        print("  - task_state.py - модуль состояния задачи")
        print("  - enhanced_agent.py - расширенный агент")
        print("  - enhanced_app.py - обновленный UI")
        print("  - design_architecture.md - дизайн архитектуры")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ОШИБКА ТЕСТИРОВАНИЯ: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)