#!/usr/bin/env python3
"""
Модуль автоматической суммаризации диалогов.
Суммаризирует диалоги каждые 10 сообщений пользователя.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

from conversation_manager import ConversationManager, Message


@dataclass
class SummarizerConfig:
    """Конфигурация суммаризатора."""
    
    # Триггер суммаризации
    summarization_trigger_count: int = 10  # Каждые N сообщений пользователя
    
    # Параметры LLM для суммаризации
    summarization_model: str = "openrouter/free"  # Модель для суммаризации
    max_tokens: int = 300  # Максимальная длина суммаризации
    
    # Промпт для суммаризации
    summary_prompt_template: str = """
    Суммаризируй следующий диалог с логистом-ассистентом. 
    Сфокусируйся на ключевых моментах:
    
    1. Основная цель или задача пользователя
    2. Уточненные детали (города, вес, сроки и т.д.)
    3. Предложенные решения и расчеты
    4. Текущий статус задачи
    
    Диалог:
    {dialogue_text}
    
    Краткая сводка (3-5 предложений, на русском языке):
    """
    
    # Альтернативный промпт для очень коротких диалогов
    short_summary_prompt_template: str = """
    Кратко опиши суть следующего диалога (1-2 предложения):
    
    {dialogue_text}
    
    Краткое описание:
    """


class Summarizer:
    """Класс для автоматической суммаризации диалогов."""
    
    def __init__(self, conversation_manager: ConversationManager, config: SummarizerConfig = None):
        """
        Инициализация суммаризатора.
        
        Args:
            conversation_manager: Менеджер диалогов
            config: Конфигурация суммаризатора
        """
        self.conversation_manager = conversation_manager
        self.config = config or SummarizerConfig()
        
        # Инициализация клиента OpenAI для суммаризации
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            timeout=60.0,
            max_retries=2
        )
        
        print(f"✅ Инициализирован суммаризатор (триггер: каждые {self.config.summarization_trigger_count} сообщений)")
    
    async def check_and_summarize(self, conversation_id: int) -> Optional[str]:
        """
        Проверяет, нужно ли суммировать диалог, и выполняет суммаризацию если нужно.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Текст суммаризации или None если суммаризация не требовалась
        """
        # Проверяем, нужно ли суммировать
        should_summarize, message_count = self.conversation_manager.should_summarize(conversation_id)
        
        if not should_summarize:
            return None
        
        print(f"📊 Запуск суммаризации диалога #{conversation_id} (сообщений пользователя: {message_count})")
        
        try:
            # Получаем сообщения диалога
            messages = self.conversation_manager.get_last_messages(conversation_id, count=50)
            
            if not messages:
                print(f"⚠️ Нет сообщений для суммаризации в диалоге #{conversation_id}")
                return None
            
            # Формируем текст диалога для суммаризации
            dialogue_text = self._format_dialogue_for_summary(messages)
            
            # Выбираем промпт в зависимости от длины диалога
            if len(messages) <= 6:
                prompt = self.config.short_summary_prompt_template.format(dialogue_text=dialogue_text)
            else:
                prompt = self.config.summary_prompt_template.format(dialogue_text=dialogue_text)
            
            # Генерируем суммаризацию через LLM
            summary = await self._generate_summary(prompt)
            
            if summary:
                # Сохраняем суммаризацию как системное сообщение
                self.conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="system",
                    content=f"📋 Краткая сводка диалога:\n{summary}",
                    is_summary=True
                )
                
                # Обновляем информацию о суммаризации
                self.conversation_manager.update_summary_info(conversation_id, summary)
                
                print(f"✅ Суммаризация диалога #{conversation_id} завершена")
                return summary
            
        except Exception as e:
            print(f"❌ Ошибка суммаризации диалога #{conversation_id}: {e}")
        
        return None
    
    def _format_dialogue_for_summary(self, messages: List[Message]) -> str:
        """
        Форматирует сообщения диалога для суммаризации.
        
        Args:
            messages: Список сообщений
            
        Returns:
            Отформатированный текст диалога
        """
        dialogue_lines = []
        
        for message in messages:
            if message.role == "user":
                prefix = "Пользователь"
            elif message.role == "assistant":
                prefix = "Ассистент"
            else:
                continue  # Пропускаем системные сообщения для суммаризации
            
            # Очищаем специальные метки из ответов
            content = message.content
            if "📚 RAG" in content:
                content = content.replace("📚 RAG (источники:", "").split(")", 1)[-1].strip()
            elif "🔧 MCP" in content:
                content = content.replace("🔧 MCP", "").strip()
            elif "💡" in content:
                content = content.replace("💡", "").strip()
            
            dialogue_lines.append(f"{prefix}: {content}")
        
        return "\n\n".join(dialogue_lines)
    
    async def _generate_summary(self, prompt: str) -> Optional[str]:
        """
        Генерирует суммаризацию через LLM.
        
        Args:
            prompt: Промпт с диалогом
            
        Returns:
            Текст суммаризации или None при ошибке
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.config.summarization_model,
                messages=[
                    {"role": "system", "content": "Ты - помощник для суммаризации диалогов. Создавай краткие, информативные сводки на русском языке."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=0.3  # Низкая температура для более предсказуемых суммаризаций
            )
            
            if response.choices and response.choices[0].message.content:
                summary = response.choices[0].message.content.strip()
                
                # Убираем возможные префиксы типа "Краткая сводка:"
                for prefix in ["Краткая сводка:", "Краткое описание:", "Сводка:", "Summary:"]:
                    if summary.startswith(prefix):
                        summary = summary[len(prefix):].strip()
                
                return summary
            
        except Exception as e:
            print(f"❌ Ошибка генерации суммаризации через LLM: {e}")
        
        return None
    
    def get_summary_for_conversation(self, conversation_id: int) -> Optional[str]:
        """
        Получает сохраненную суммаризацию для диалога.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Текст последней суммаризации или None
        """
        task_state = self.conversation_manager.get_task_state(conversation_id)
        if task_state and task_state.last_summary:
            return task_state.last_summary
        
        return None
    
    async def force_summarize(self, conversation_id: int) -> Optional[str]:
        """
        Принудительно выполняет суммаризацию диалога.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Текст суммаризации
        """
        print(f"🔧 Принудительная суммаризация диалога #{conversation_id}")
        
        # Получаем все сообщения диалога
        messages = self.conversation_manager.get_conversation_messages(conversation_id)
        
        if not messages:
            print(f"⚠️ Нет сообщений в диалоге #{conversation_id}")
            return None
        
        # Форматируем диалог
        dialogue_text = self._format_dialogue_for_summary(messages)
        
        # Генерируем промпт
        if len(messages) <= 10:
            prompt = self.config.short_summary_prompt_template.format(dialogue_text=dialogue_text)
        else:
            prompt = self.config.summary_prompt_template.format(dialogue_text=dialogue_text)
        
        # Генерируем суммаризацию
        summary = await self._generate_summary(prompt)
        
        if summary:
            # Сохраняем суммаризацию
            self.conversation_manager.update_summary_info(conversation_id, summary)
            print(f"✅ Принудительная суммаризация диалога #{conversation_id} завершена")
        
        return summary
    
    def get_summary_context(self, conversation_id: int) -> str:
        """
        Получает контекст с суммаризацией для включения в промпт LLM.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Строка контекста с суммаризацией
        """
        summary = self.get_summary_for_conversation(conversation_id)
        
        if not summary:
            return ""
        
        # Получаем информацию о диалоге для контекста
        conversation = self.conversation_manager.get_conversation(conversation_id)
        task_state = self.conversation_manager.get_task_state(conversation_id)
        
        context_lines = []
        
        if conversation:
            context_lines.append(f"Текущий диалог: {conversation.title}")
        
        if task_state and task_state.message_count > 20:
            context_lines.append(f"В диалоге более {task_state.message_count} сообщений, используй сводку ниже.")
        
        context_lines.append(f"📋 Краткая сводка предыдущего обсуждения:\n{summary}")
        
        return "\n\n".join(context_lines)


# Синглтон для глобального доступа
_summarizer: Optional[Summarizer] = None

def get_summarizer(conversation_manager: ConversationManager = None) -> Summarizer:
    """Получает глобальный экземпляр суммаризатора."""
    global _summarizer
    if _summarizer is None:
        if conversation_manager is None:
            from conversation_manager import get_conversation_manager
            conversation_manager = get_conversation_manager()
        _summarizer = Summarizer(conversation_manager)
    return _summarizer


# Тестовые функции
async def test_summarizer():
    """Тестирование модуля суммаризации."""
    print("🧪 Тестирование модуля суммаризации...")
    
    from conversation_manager import get_conversation_manager
    
    # Создаем менеджер диалогов
    manager = get_conversation_manager("test_conversations.db")
    
    # Создаем тестовый диалог
    conversation = manager.create_conversation("Тестовый диалог для суммаризации")
    
    # Добавляем тестовые сообщения
    test_messages = [
        ("user", "Привет! Мне нужно рассчитать маршрут между Москвой и Санкт-Петербургом."),
        ("assistant", "🔧 MCP\n\nЯ помогу рассчитать оптимальный маршрут между Москвой и Санкт-Петербургом. Для этого мне нужно получить координаты этих городов и рассчитать расстояние."),
        ("user", "Отлично! Также добавьте Казань в маршрут."),
        ("assistant", "🔧 MCP\n\nБуду рассчитывать маршрут по трем городам: Москва, Санкт-Петербург и Казань. Сначала получу координаты всех городов."),
        ("user", "Какой будет общее расстояние?"),
        ("assistant", "🔧 MCP\n\nРассчитал оптимальный маршрут: Москва → Казань → Санкт-Петербург. Общее расстояние: 2320 км."),
        ("user", "Спасибо! А сколько будет стоить доставка груза весом 100 кг?"),
        ("assistant", "🔧 MCP\n\nДля расчета стоимости доставки через ПЭК мне нужны ID городов отправления и назначения. Рассчитываю стоимость для маршрута Москва → Санкт-Петербург с грузом 100 кг."),
        ("user", "Отлично, жду результат."),
        ("assistant", "🔧 MCP\n\nСтоимость доставки 100 кг из Москвы в Санкт-Петербург через ПЭК: примерно 12,500 рублей."),
        ("user", "А есть ли скидки для регулярных перевозок?"),
        ("assistant", "📚 RAG (источники: pecom.txt)\n\nСогласно документации ПЭК, для корпоративных клиентов предусмотрены скидки от 5% до 15% в зависимости от объема перевозок. Для получения точной информации рекомендуется обратиться в отдел продаж ПЭК."),
    ]
    
    for role, content in test_messages:
        manager.add_message(conversation.id, role, content)
    
    # Создаем суммаризатор
    summarizer = Summarizer(manager)
    
    # Проверяем, нужно ли суммировать
    should_summarize, count = manager.should_summarize(conversation.id)
    print(f"  Диалог #{conversation.id}: {count} сообщений пользователя, нужно суммировать: {should_summarize}")
    
    # Принудительно суммируем
    summary = await summarizer.force_summarize(conversation.id)
    
    if summary:
        print(f"  Сгенерированная суммаризация:\n  {summary}")
    else:
        print("  ❌ Не удалось сгенерировать суммаризацию")
    
    # Получаем контекст с суммаризацией
    context = summarizer.get_summary_context(conversation.id)
    print(f"  Контекст для LLM:\n  {context[:200]}...")
    
    # Очищаем тестовую БД
    import os
    if os.path.exists("test_conversations.db"):
        os.remove("test_conversations.db")
    
    print("✅ Тест завершен")


if __name__ == "__main__":
    # Запуск теста при прямом выполнении
    asyncio.run(test_summarizer())