#!/usr/bin/env python3
"""
Модуль автоматической суммаризации диалогов.
Суммаризирует диалоги каждые 10 сообщений пользователя.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from openai import AsyncOpenAI
import os

from utils.config import settings
from core.conversation_manager import ConversationManager, Message, get_conversation_manager


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
    
    Краткая суть:
    """


class Summarizer:
    """Суммаризатор диалогов."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.config = SummarizerConfig()
            self.conversation_manager = get_conversation_manager()
            
            # Инициализируем OpenAI клиент
            self.client = AsyncOpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
    
    async def summarize_conversation(self, conversation_id: int) -> Optional[str]:
        """
        Суммаризирует диалог.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Суммаризация или None при ошибке
        """
        try:
            # Получаем сообщения диалога
            messages = self.conversation_manager.get_messages(conversation_id)
            if not messages:
                return None
            
            # Форматируем диалог для суммаризации
            dialogue_text = self._format_dialogue_for_summary(messages)
            
            # Выбираем промпт в зависимости от длины диалога
            if len(messages) <= 5:
                prompt = self.config.short_summary_prompt_template.format(
                    dialogue_text=dialogue_text
                )
            else:
                prompt = self.config.summary_prompt_template.format(
                    dialogue_text=dialogue_text
                )
            
            # Генерируем суммаризацию через LLM
            summary = await self._generate_summary(prompt)
            
            if summary:
                # Сохраняем суммаризацию как системное сообщение
                self.conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="system",
                    content=f"📋 Сводка диалога: {summary}",
                    is_summary=True
                )
                
                # Обновляем состояние задачи
                self.conversation_manager.update_task_state(
                    conversation_id,
                    last_summary=summary,
                    last_summarized_at_message=self.conversation_manager.get_user_message_count(conversation_id)
                )
                
                print(f"✅ Диалог {conversation_id} суммаризирован")
                return summary
            
            return None
            
        except Exception as e:
            print(f"❌ Ошибка суммаризации диалога {conversation_id}: {e}")
            return None
    
    def _format_dialogue_for_summary(self, messages: List[Message]) -> str:
        """
        Форматирует диалог для суммаризации.
        
        Args:
            messages: Список сообщений
            
        Returns:
            Отформатированный текст диалога
        """
        formatted_lines = []
        
        for msg in messages:
            if msg.is_summary:
                continue  # Пропускаем предыдущие суммаризации
            
            role_emoji = "👤" if msg.role == "user" else "🤖"
            role_name = "Пользователь" if msg.role == "user" else "Ассистент"
            
            formatted_lines.append(f"{role_emoji} {role_name}: {msg.content}")
        
        return "\n\n".join(formatted_lines)
    
    async def _generate_summary(self, prompt: str) -> Optional[str]:
        """
        Генерирует суммаризацию через LLM.
        
        Args:
            prompt: Промпт для суммаризации
            
        Returns:
            Сгенерированная суммаризация или None при ошибке
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.config.summarization_model,
                messages=[
                    {"role": "system", "content": "Ты помощник для суммаризации диалогов с логистом."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=0.3,  # Низкая температура для более детерминированных результатов
                top_p=0.9
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Очищаем суммаризацию от возможных маркеров
            summary = summary.replace("Краткая сводка:", "").replace("Краткая суть:", "").strip()
            
            return summary
            
        except Exception as e:
            print(f"❌ Ошибка генерации суммаризации через LLM: {e}")
            return None
    
    def should_summarize(self, conversation_id: int) -> Tuple[bool, int]:
        """
        Проверяет, нужно ли суммировать диалог.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Кортеж (нужна ли суммаризация, количество сообщений пользователя)
        """
        return self.conversation_manager.should_summarize(conversation_id)
    
    async def check_and_summarize(self, conversation_id: int) -> Tuple[bool, Optional[str]]:
        """
        Проверяет и при необходимости суммирует диалог.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Кортеж (была ли выполнена суммаризация, текст суммаризации)
        """
        should_summarize, user_count = self.should_summarize(conversation_id)
        
        if should_summarize:
            summary = await self.summarize_conversation(conversation_id)
            return True, summary
        
        return False, None
    
    def get_summary_for_conversation(self, conversation_id: int) -> Optional[str]:
        """
        Получает последнюю суммаризацию диалога.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Последняя суммаризация или None
        """
        task_state = self.conversation_manager.get_task_state(conversation_id)
        if task_state and task_state.last_summary:
            return task_state.last_summary
        
        # Ищем последнее сообщение-суммаризацию
        messages = self.conversation_manager.get_recent_messages(conversation_id, count=20)
        for msg in messages:
            if msg.is_summary and msg.role == "system":
                # Извлекаем суммаризацию из сообщения
                content = msg.content
                if "📋 Сводка диалога:" in content:
                    return content.replace("📋 Сводка диалога:", "").strip()
                return content
        
        return None
    
    async def force_summarize(self, conversation_id: int) -> Optional[str]:
        """
        Принудительно суммирует диалог независимо от счетчика.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Суммаризация или None при ошибке
        """
        return await self.summarize_conversation(conversation_id)
    
    def update_config(self, **kwargs):
        """
        Обновляет конфигурацию суммаризатора.
        
        Args:
            **kwargs: Параметры конфигурации
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_config(self) -> SummarizerConfig:
        """
        Возвращает текущую конфигурацию.
        
        Returns:
            Конфигурация суммаризатора
        """
        return self.config


# Глобальный экземпляр суммаризатора
_summarizer = None

def get_summarizer(conversation_manager: Optional[ConversationManager] = None) -> Summarizer:
    """
    Возвращает глобальный экземпляр суммаризатора.
    
    Args:
        conversation_manager: Менеджер диалогов (опционально)
        
    Returns:
        Суммаризатор
    """
    global _summarizer
    if _summarizer is None:
        _summarizer = Summarizer()
        if conversation_manager:
            _summarizer.conversation_manager = conversation_manager
    return _summarizer