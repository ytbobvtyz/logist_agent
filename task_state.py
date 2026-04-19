#!/usr/bin/env python3
"""
Модуль Task State (память задачи).
Отслеживает состояние задачи в диалоге:
- Что пользователь уже уточнил
- Какие ограничения/термины зафиксированы
- Что является целью диалога
"""

import json
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

from conversation_manager import Message, ConversationManager


@dataclass
class TaskState:
    """Структура состояния задачи."""
    
    # Основные поля
    goal: str = ""  # Цель диалога
    
    # Уточненные детали
    clarified_details: List[str] = field(default_factory=list)  # Что уже уточнил пользователь
    
    # Ограничения и термины
    constraints: Dict[str, Any] = field(default_factory=dict)  # Ограничения (макс. города, вес и т.д.)
    terms: Dict[str, str] = field(default_factory=dict)  # Термины и их определения
    
    # Статистика
    message_count: int = 0  # Общее количество сообщений
    user_message_count: int = 0  # Количество сообщений пользователя
    last_updated: str = ""  # Время последнего обновления
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь."""
        return {
            "goal": self.goal,
            "clarified_details": self.clarified_details,
            "constraints": self.constraints,
            "terms": self.terms,
            "message_count": self.message_count,
            "user_message_count": self.user_message_count,
            "last_updated": self.last_updated or datetime.now().isoformat()
        }
    
    def to_context_string(self) -> str:
        """Формирует строку контекста для включения в промпт LLM."""
        lines = []
        
        if self.goal:
            lines.append(f"Цель диалога: {self.goal}")
        
        if self.clarified_details:
            details_text = "\n".join([f"- {detail}" for detail in self.clarified_details])
            lines.append(f"Уточнения пользователя:\n{details_text}")
        
        if self.constraints:
            constraints_text = "\n".join([f"- {key}: {value}" for key, value in self.constraints.items()])
            lines.append(f"Ограничения:\n{constraints_text}")
        
        if self.terms:
            terms_text = "\n".join([f"- {term}: {definition}" for term, definition in self.terms.items()])
            lines.append(f"Определения терминов:\n{terms_text}")
        
        if self.message_count > 10:
            lines.append(f"В диалоге {self.message_count} сообщений.")
        
        return "\n\n".join(lines)
    
    def merge(self, other: 'TaskState') -> 'TaskState':
        """Объединяет два состояния задачи."""
        merged = TaskState()
        
        # Объединяем цели (приоритет у более специфичной)
        if other.goal and (not self.goal or len(other.goal) > len(self.goal)):
            merged.goal = other.goal
        else:
            merged.goal = self.goal
        
        # Объединяем уточнения (уникальные)
        merged.clarified_details = list(set(self.clarified_details + other.clarified_details))
        
        # Объединяем ограничения (побеждают более строгие)
        merged.constraints = self.constraints.copy()
        for key, value in other.constraints.items():
            if key not in merged.constraints or self._is_more_constraining(value, merged.constraints[key]):
                merged.constraints[key] = value
        
        # Объединяем термины (побеждают более полные определения)
        merged.terms = self.terms.copy()
        for term, definition in other.terms.items():
            if term not in merged.terms or len(definition) > len(merged.terms[term]):
                merged.terms[term] = definition
        
        # Статистика - берем максимум
        merged.message_count = max(self.message_count, other.message_count)
        merged.user_message_count = max(self.user_message_count, other.user_message_count)
        merged.last_updated = max(self.last_updated, other.last_updated)
        
        return merged
    
    def _is_more_constraining(self, new_value: Any, old_value: Any) -> bool:
        """Определяет, является ли новое значение более строгим ограничением."""
        # Простая эвристика для чисел (меньшее число - более строгое ограничение)
        if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
            return new_value < old_value
        
        # Для строк - более короткая строка обычно более специфична
        if isinstance(new_value, str) and isinstance(old_value, str):
            return len(new_value) < len(old_value)
        
        # По умолчанию считаем новое значение более ограничивающим
        return True


class TaskStateExtractor:
    """Класс для извлечения информации о состоянии задачи из сообщений."""
    
    def __init__(self):
        """Инициализация экстрактора."""
        
        # Паттерны для извлечения уточнений
        self.detail_patterns = [
            # Города
            (r'(?:город[а]?|маршрут)[\s\w]*:?\s*([А-Я][а-я]+(?:\s+[А-Я][а-я]+)*)', self._extract_cities),
            
            # Вес груза
            (r'(?:вес|масса)[\s\w]*:?\s*(\d+)\s*(?:кг|килограмм)', self._extract_weight),
            
            # Сроки
            (r'(?:срок|время|доставк[аи])[\s\w]*:?\s*(\d+)\s*(?:час|день|сутк)', self._extract_time),
            
            # Стоимость
            (r'(?:стоимость|цена|бюджет)[\s\w]*:?\s*(\d+)\s*(?:руб|₽)', self._extract_cost),
        ]
        
        # Паттерны для ограничений
        self.constraint_patterns = [
            (r'(?:не более|максимум|до)\s+(\d+)\s*(?:город|город[ао]в)', 'max_cities'),
            (r'(?:макс|не более)\s+(\d+)\s*(?:кг|килограмм)', 'max_weight'),
            (r'(?:в течение|не более)\s+(\d+)\s*(?:час|часов|день|дней)', 'time_limit'),
        ]
        
        # Термины из области логистики
        self.logistics_terms = {
            'фрахтователь': 'Заказчик перевозки, нанимающий перевозчика',
            'фрахтовщик': 'Перевозчик, оказывающий услуги по перевозке',
            'экспедитор': 'Организация, организующая перевозку грузов',
            'тариф': 'Ставка оплаты за перевозку груза',
            'грузовладелец': 'Владелец груза, отправляемый к получателю',
            'транспортная накладная': 'Документ, сопровождающий груз при перевозке',
            'экспедиторская расписка': 'Документ, подтверждающий прием груза к перевозке',
        }
        
        # Ключевые слова для определения цели
        self.goal_keywords = {
            'рассчитать': ['расчет', 'посчитать', 'вычислить'],
            'найти': ['найти', 'поиск', 'определить'],
            'узнать': ['узнать', 'выяснить', 'получить информацию'],
            'спланировать': ['спланировать', 'планирование', 'организовать'],
        }
    
    def extract_from_message(self, message: Message) -> TaskState:
        """
        Извлекает информацию о состоянии задачи из одного сообщения.
        
        Args:
            message: Сообщение для анализа
            
        Returns:
            TaskState с извлеченной информацией
        """
        state = TaskState()
        content = message.content.lower()
        
        # Извлекаем информацию только из сообщений пользователя
        if message.role != 'user':
            return state
        
        # Извлекаем уточнения
        for pattern, extractor in self.detail_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                details = extractor(match)
                if details:
                    state.clarified_details.extend(details)
        
        # Извлекаем ограничения
        for pattern, constraint_name in self.constraint_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                value = match.group(1)
                try:
                    state.constraints[constraint_name] = int(value)
                except ValueError:
                    state.constraints[constraint_name] = value
        
        # Определяем термины
        for term, definition in self.logistics_terms.items():
            if term in content:
                state.terms[term] = definition
        
        # Пытаемся определить цель (простая эвристика)
        if not state.goal:
            state.goal = self._extract_goal(content, message.content)
        
        # Обновляем статистику
        state.user_message_count = 1
        state.message_count = 1
        state.last_updated = datetime.now().isoformat()
        
        return state
    
    def _extract_cities(self, match) -> List[str]:
        """Извлекает города из текста."""
        cities_text = match.group(1)
        # Простая разбивка на слова, начинающиеся с заглавной буквы
        cities = re.findall(r'[А-Я][а-я]+', cities_text)
        return [f"Город: {city}" for city in cities]
    
    def _extract_weight(self, match) -> List[str]:
        """Извлекает вес груза."""
        weight = match.group(1)
        return [f"Вес груза: {weight} кг"]
    
    def _extract_time(self, match) -> List[str]:
        """Извлекает сроки."""
        time_value = match.group(1)
        unit = 'часов' if 'час' in match.group(0) else 'дней'
        return [f"Срок: {time_value} {unit}"]
    
    def _extract_cost(self, match) -> List[str]:
        """Извлекает стоимость."""
        cost = match.group(1)
        return [f"Бюджет: {cost} руб"]
    
    def _extract_goal(self, content_lower: str, original_content: str) -> str:
        """Извлекает цель из сообщения."""
        # Ищем ключевые слова цели
        goal_types = []
        
        for goal_type, keywords in self.goal_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    goal_types.append(goal_type)
                    break
        
        if not goal_types:
            return ""
        
        # Берем первую найденную цель
        goal_type = goal_types[0]
        
        # Пытаемся найти объект цели
        goal_object = self._find_goal_object(content_lower, original_content)
        
        if goal_object:
            return f"{goal_type} {goal_object}"
        else:
            return goal_type
    
    def _find_goal_object(self, content_lower: str, original_content: str) -> str:
        """Находит объект цели в сообщении."""
        # Паттерны для поиска объектов
        patterns = [
            r'рассчитать (?:стоимость|цену|тариф|расходы)',
            r'найти (?:оптимальный|лучший|кратчайший) (?:маршрут|путь)',
            r'узнать (?:правила|условия|требования)',
            r'спланировать (?:доставку|перевозку|маршрут)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content_lower)
            if match:
                # Берем соответствующий фрагмент из оригинального текста
                start, end = match.span()
                return original_content[start:end]
        
        # Если не нашли паттерн, ищем существительные после глагола
        verb_patterns = [
            r'рассчитать\s+([^\.,!?]+)',
            r'найти\s+([^\.,!?]+)',
            r'узнать\s+([^\.,!?]+)',
            r'спланировать\s+([^\.,!?]+)',
        ]
        
        for pattern in verb_patterns:
            match = re.search(pattern, content_lower)
            if match:
                return match.group(1).strip()
        
        return ""


class TaskStateManager:
    """Менеджер состояния задачи."""
    
    def __init__(self, conversation_manager: ConversationManager):
        """
        Инициализация менеджера состояния задачи.
        
        Args:
            conversation_manager: Менеджер диалогов
        """
        self.conversation_manager = conversation_manager
        self.extractor = TaskStateExtractor()
        
        # Инициализация клиента OpenAI для анализа (опционально)
        self.use_llm_analysis = False
        if os.getenv("OPENROUTER_API_KEY"):
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                timeout=30.0,
                max_retries=1
            )
            self.use_llm_analysis = True
        
        print("✅ Инициализирован менеджер состояния задачи")
    
    async def update_from_new_message(self, conversation_id: int, message: Message) -> TaskState:
        """
        Обновляет состояние задачи на основе нового сообщения.
        
        Args:
            conversation_id: ID диалога
            message: Новое сообщение
            
        Returns:
            Обновленное состояние задачи
        """
        # Получаем текущее состояние
        db_task_state = self.conversation_manager.get_task_state(conversation_id)
        if not db_task_state:
            # Создаем начальное состояние
            current_state = TaskState()
        else:
            current_state = TaskState(
                goal=db_task_state.goal or "",
                clarified_details=db_task_state.clarified_details.copy() if db_task_state.clarified_details else [],
                constraints=db_task_state.constraints.copy() if db_task_state.constraints else {},
                terms={},  # Пока не храним terms в БД
                message_count=db_task_state.message_count,
                user_message_count=db_task_state.message_count,  # Используем общий счетчик как приближение
                last_updated=db_task_state.updated_at or ""
            )
        
        # Извлекаем информацию из нового сообщения
        new_info = self.extractor.extract_from_message(message)
        
        # Объединяем состояния
        updated_state = current_state.merge(new_info)
        
        # Обновляем статистику
        updated_state.message_count = current_state.message_count + 1
        if message.role == 'user':
            updated_state.user_message_count = current_state.user_message_count + 1
        else:
            updated_state.user_message_count = current_state.user_message_count
        
        updated_state.last_updated = datetime.now().isoformat()
        
        # Используем LLM для улучшения анализа (опционально)
        if self.use_llm_analysis and message.role == 'user':
            await self._enhance_with_llm(conversation_id, message, updated_state)
        
        # Сохраняем обновленное состояние в БД
        self._save_to_db(conversation_id, updated_state)
        
        return updated_state
    
    async def _enhance_with_llm(self, conversation_id: int, message: Message, state: TaskState):
        """
        Использует LLM для улучшения анализа состояния задачи.
        
        Args:
            conversation_id: ID диалога
            message: Сообщение для анализа
            state: Текущее состояние задачи
        """
        try:
            # Строим промпт для анализа
            prompt = self._build_llm_analysis_prompt(message, state)
            
            response = await self.client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {"role": "system", "content": "Ты помощник для анализа диалогов с логистическим ассистентом. Извлекай ключевую информацию о состоянии задачи."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            if response.choices and response.choices[0].message.content:
                analysis = response.choices[0].message.content.strip()
                self._parse_llm_analysis(analysis, state)
                
        except Exception as e:
            # Игнорируем ошибки LLM анализа - используем только rule-based подход
            pass
    
    def _build_llm_analysis_prompt(self, message: Message, current_state: TaskState) -> str:
        """Строит промпт для LLM анализа."""
        context = f"Сообщение пользователя: {message.content}"
        
        if current_state.goal:
            context += f"\n\nТекущая цель: {current_state.goal}"
        
        if current_state.clarified_details:
            details = "\n".join(current_state.clarified_details[:5])
            context += f"\n\nУже уточнено:\n{details}"
        
        prompt = f"""
        Проанализируй сообщение пользователя в контексте диалога с логистическим ассистентом.
        
        {context}
        
        Определи:
        1. Какую новую информацию уточнил пользователь?
        2. Нужно ли обновить цель диалога?
        3. Есть ли новые ограничения или требования?
        
        Ответь в формате JSON:
        {{
            "new_details": ["деталь 1", "деталь 2"],
            "updated_goal": "обновленная цель или пустая строка",
            "new_constraints": {{"ключ": "значение"}}
        }}
        """
        
        return prompt
    
    def _parse_llm_analysis(self, analysis: str, state: TaskState):
        """Парсит результат LLM анализа."""
        try:
            # Ищем JSON в ответе
            json_match = re.search(r'\{.*\}', analysis, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Обновляем детали
                if 'new_details' in data and isinstance(data['new_details'], list):
                    state.clarified_details.extend(data['new_details'])
                    # Удаляем дубликаты
                    state.clarified_details = list(set(state.clarified_details))
                
                # Обновляем цель
                if 'updated_goal' in data and data['updated_goal']:
                    state.goal = data['updated_goal']
                
                # Обновляем ограничения
                if 'new_constraints' in data and isinstance(data['new_constraints'], dict):
                    state.constraints.update(data['new_constraints'])
        
        except (json.JSONDecodeError, KeyError) as e:
            # Игнорируем ошибки парсинга
            pass
    
    def _save_to_db(self, conversation_id: int, state: TaskState):
        """Сохраняет состояние задачи в БД через ConversationManager."""
        self.conversation_manager.update_task_state(
            conversation_id,
            goal=state.goal,
            clarified_details=state.clarified_details,
            constraints=state.constraints,
            last_summary="",  # Суммаризация обрабатывается отдельно
            message_count=state.message_count,
            last_summarized_at_message=state.user_message_count  # Приближение
        )
    
    def get_task_state(self, conversation_id: int) -> Optional[TaskState]:
        """
        Получает состояние задачи для диалога.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            TaskState или None
        """
        db_state = self.conversation_manager.get_task_state(conversation_id)
        if not db_state:
            return None
        
        # Преобразуем из формата БД
        constraints = {}
        if db_state.constraints:
            try:
                if isinstance(db_state.constraints, str):
                    constraints = json.loads(db_state.constraints)
                elif isinstance(db_state.constraints, dict):
                    constraints = db_state.constraints
                else:
                    constraints = {}
            except (json.JSONDecodeError, TypeError):
                constraints = {}
        
        return TaskState(
            goal=db_state.goal or "",
            clarified_details=db_state.clarified_details or [],
            constraints=constraints,
            terms={},  # Пока не храним термины в БД
            message_count=db_state.message_count,
            user_message_count=db_state.message_count,  # Приближение
            last_updated=db_state.updated_at or ""
        )
    
    def get_context_for_llm(self, conversation_id: int) -> str:
        """
        Получает контекст состояния задачи для включения в промпт LLM.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Строка контекста
        """
        state = self.get_task_state(conversation_id)
        if not state:
            return ""
        
        return state.to_context_string()
    
    def reset_task_state(self, conversation_id: int) -> bool:
        """
        Сбрасывает состояние задачи для диалога.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            True если успешно
        """
        try:
            self.conversation_manager.update_task_state(
                conversation_id,
                goal="",
                clarified_details=[],
                constraints={},
                last_summary="",
                message_count=0,
                last_summarized_at_message=0
            )
            return True
        except Exception as e:
            print(f"❌ Ошибка сброса состояния задачи: {e}")
            return False


# Синглтон для глобального доступа
_task_state_manager: Optional[TaskStateManager] = None

def get_task_state_manager(conversation_manager: ConversationManager = None) -> TaskStateManager:
    """Получает глобальный экземпляр менеджера состояния задачи."""
    global _task_state_manager
    if _task_state_manager is None:
        if conversation_manager is None:
            from conversation_manager import get_conversation_manager
            conversation_manager = get_conversation_manager()
        _task_state_manager = TaskStateManager(conversation_manager)
    return _task_state_manager


# Тестовые функции
async def test_task_state_manager():
    """Тестирование модуля состояния задачи."""
    print("🧪 Тестирование модуля состояния задачи...")
    
    from conversation_manager import get_conversation_manager
    
    # Создаем менеджер диалогов
    manager = get_conversation_manager("test_task_state.db")
    
    # Создаем тестовый диалог
    conversation = manager.create_conversation("Тест состояния задачи")
    
    # Создаем менеджер состояния задачи
    task_manager = TaskStateManager(manager)
    
    # Тестовые сообщения
    test_messages = [
        ("user", "Привет! Мне нужно рассчитать оптимальный маршрут между Москвой, Санкт-Петербургом и Казанью."),
        ("assistant", "🔧 MCP\n\nХорошо, рассчитаю маршрут по трем городам."),
        ("user", "Вес груза - 150 кг, нужно доставить в течение 2 дней."),
        ("assistant", "🔧 MCP\n\nУчитываю вес 150 кг и срок 2 дня."),
        ("user", "Максимальный бюджет - 20 000 рублей, и не более 5 городов в маршруте."),
    ]
    
    # Обрабатываем сообщения
    for role, content in test_messages:
        message = Message(role=role, content=content)
        
        # Добавляем в диалог
        manager.add_message(conversation.id, role, content)
        
        # Обновляем состояние задачи
        if role == 'user':
            state = await task_manager.update_from_new_message(conversation.id, message)
    
    # Получаем финальное состояние
    final_state = task_manager.get_task_state(conversation.id)
    
    if final_state:
        print(f"  Цель: {final_state.goal}")
        print(f"  Уточнения: {final_state.clarified_details}")
        print(f"  Ограничения: {final_state.constraints}")
        print(f"  Контекст для LLM:\n  {final_state.to_context_string()[:200]}...")
    else:
        print("  ❌ Не удалось получить состояние задачи")
    
    # Очищаем тестовую БД
    import os
    if os.path.exists("test_task_state.db"):
        os.remove("test_task_state.db")
    
    print("✅ Тест завершен")


if __name__ == "__main__":
    # Запуск теста при прямом выполнении
    asyncio.run(test_task_state_manager())