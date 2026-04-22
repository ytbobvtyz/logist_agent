#!/usr/bin/env python3
"""
Модуль управления состоянием задачи (task state).
Отслеживание цели диалога, уточнений и ограничений.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from core.conversation_manager import ConversationManager, TaskState as ConversationTaskState, get_conversation_manager


@dataclass
class TaskStateConfig:
    """Конфигурация менеджера состояния задачи."""
    
    # Ключевые слова для извлечения цели
    goal_keywords: List[str] = None
    
    # Паттерны для извлечения деталей
    city_pattern: str = r'\b(?:Москва|Санкт-Петербург|Казань|Нижний Новгород|Екатеринбург|Новосибирск|Краснодар|Сочи|Ростов|Владивосток)\b'
    weight_pattern: str = r'\b(\d+(?:\.\d+)?)\s*(?:кг|килограмм|kg)\b'
    time_pattern: str = r'\b(\d+)\s*(?:час|часов|день|дней|суток)\b'
    
    # Минимальная уверенность для обновления состояния
    min_confidence: float = 0.7
    
    def __post_init__(self):
        if self.goal_keywords is None:
            self.goal_keywords = [
                "рассчитать", "найти", "построить", "оптимизировать",
                "маршрут", "доставка", "стоимость", "расстояние",
                "план", "логистика", "перевозка"
            ]


class TaskStateManager:
    """Менеджер состояния задачи."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.config = TaskStateConfig()
            self.conversation_manager = get_conversation_manager()
    
    def get_task_state(self, conversation_id: int) -> Optional[ConversationTaskState]:
        """
        Получает состояние задачи для диалога.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Состояние задачи или None, если не найдено
        """
        return self.conversation_manager.get_task_state(conversation_id)
    
    def update_from_message(self, conversation_id: int, message: str, 
                           role: str = "user") -> bool:
        """
        Обновляет состояние задачи на основе сообщения.
        
        Args:
            conversation_id: ID диалога
            message: Текст сообщения
            role: Роль отправителя
            
        Returns:
            True если состояние обновлено, иначе False
        """
        if role != "user":
            return False  # Обновляем только на основе сообщений пользователя
        
        try:
            # Получаем текущее состояние
            task_state = self.get_task_state(conversation_id)
            if not task_state:
                return False
            
            # Извлекаем информацию из сообщения
            extracted_info = self._extract_information(message)
            
            # Обновляем состояние
            updated = self._update_state(task_state, extracted_info)
            
            if updated:
                # Сохраняем обновленное состояние
                return self.conversation_manager.update_task_state(
                    conversation_id,
                    clarified_details=task_state.clarified_details,
                    constraints=task_state.constraints,
                    goal=task_state.goal
                )
            
            return False
            
        except Exception as e:
            print(f"❌ Ошибка обновления состояния задачи: {e}")
            return False
    
    def _extract_information(self, message: str) -> Dict[str, Any]:
        """
        Извлекает информацию из сообщения.
        
        Args:
            message: Текст сообщения
            
        Returns:
            Словарь с извлеченной информацией
        """
        info = {
            "cities": [],
            "weights": [],
            "times": [],
            "goal": "",
            "constraints": {}
        }
        
        # Приводим к нижнему регистру для поиска
        message_lower = message.lower()
        
        # Извлекаем города
        city_matches = re.findall(self.config.city_pattern, message, re.IGNORECASE)
        if city_matches:
            info["cities"] = list(set(city_matches))  # Убираем дубликаты
        
        # Извлекаем вес
        weight_matches = re.findall(self.config.weight_pattern, message_lower)
        if weight_matches:
            info["weights"] = [float(w) for w in weight_matches]
        
        # Извлекаем время
        time_matches = re.findall(self.config.time_pattern, message_lower)
        if time_matches:
            info["times"] = [int(t) for t in time_matches]
        
        # Извлекаем цель
        for keyword in self.config.goal_keywords:
            if keyword in message_lower:
                # Ищем предложение с ключевым словом
                sentences = re.split(r'[.!?]', message)
                for sentence in sentences:
                    if keyword in sentence.lower():
                        info["goal"] = sentence.strip()
                        break
                if info["goal"]:
                    break
        
        # Извлекаем ограничения
        constraints = self._extract_constraints(message_lower)
        if constraints:
            info["constraints"] = constraints
        
        return info
    
    def _extract_constraints(self, message: str) -> Dict[str, Any]:
        """
        Извлекает ограничения из сообщения.
        
        Args:
            message: Текст сообщения в нижнем регистре
            
        Returns:
            Словарь с ограничениями
        """
        constraints = {}
        
        # Проверяем наличие ограничений по времени
        time_patterns = [
            (r'не более (\d+) (?:час|часов|день|дней)', 'max_time'),
            (r'до (\d+) (?:час|часов|день|дней)', 'max_time'),
            (r'в течение (\d+) (?:час|часов|день|дней)', 'max_time'),
            (r'срочн[а-я]*', 'urgent'),
        ]
        
        for pattern, constraint_type in time_patterns:
            match = re.search(pattern, message)
            if match:
                if constraint_type == 'max_time':
                    constraints['max_time_hours'] = int(match.group(1))
                    if 'день' in match.group(0) or 'дней' in match.group(0):
                        constraints['max_time_hours'] *= 24
                elif constraint_type == 'urgent':
                    constraints['urgent'] = True
        
        # Проверяем ограничения по весу
        weight_patterns = [
            (r'не более (\d+(?:\.\d+)?)\s*(?:кг|килограмм)', 'max_weight'),
            (r'до (\d+(?:\.\d+)?)\s*(?:кг|килограмм)', 'max_weight'),
            (r'максимум (\d+(?:\.\d+)?)\s*(?:кг|килограмм)', 'max_weight'),
        ]
        
        for pattern, constraint_type in weight_patterns:
            match = re.search(pattern, message)
            if match:
                constraints['max_weight_kg'] = float(match.group(1))
        
        # Проверяем ограничения по количеству городов
        city_patterns = [
            (r'не более (\d+) (?:город|городов)', 'max_cities'),
            (r'до (\d+) (?:город|городов)', 'max_cities'),
            (r'максимум (\d+) (?:город|городов)', 'max_cities'),
        ]
        
        for pattern, constraint_type in city_patterns:
            match = re.search(pattern, message)
            if match:
                constraints['max_cities'] = int(match.group(1))
        
        # Проверяем бюджетные ограничения
        budget_patterns = [
            (r'не более (\d+(?:\.\d+)?)\s*(?:руб|рублей|₽)', 'max_budget'),
            (r'до (\d+(?:\.\d+)?)\s*(?:руб|рублей|₽)', 'max_budget'),
            (r'бюджет (\d+(?:\.\d+)?)\s*(?:руб|рублей|₽)', 'max_budget'),
        ]
        
        for pattern, constraint_type in budget_patterns:
            match = re.search(pattern, message)
            if match:
                constraints['max_budget_rub'] = float(match.group(1))
        
        return constraints
    
    def _update_state(self, task_state: ConversationTaskState, 
                     extracted_info: Dict[str, Any]) -> bool:
        """
        Обновляет состояние задачи на основе извлеченной информации.
        
        Args:
            task_state: Текущее состояние задачи
            extracted_info: Извлеченная информация
            
        Returns:
            True если состояние было обновлено, иначе False
        """
        updated = False
        
        # Обновляем цель
        if extracted_info["goal"] and not task_state.goal:
            task_state.goal = extracted_info["goal"]
            updated = True
            print(f"🎯 Установлена цель: {task_state.goal}")
        
        # Добавляем города
        for city in extracted_info["cities"]:
            city_detail = f"Город: {city}"
            if city_detail not in task_state.clarified_details:
                task_state.clarified_details.append(city_detail)
                updated = True
                print(f"📍 Добавлен город: {city}")
        
        # Добавляем информацию о весе
        for weight in extracted_info["weights"]:
            weight_detail = f"Вес: {weight} кг"
            if weight_detail not in task_state.clarified_details:
                task_state.clarified_details.append(weight_detail)
                updated = True
                print(f"⚖️ Добавлен вес: {weight} кг")
        
        # Добавляем информацию о времени
        for time in extracted_info["times"]:
            # Определяем единицу измерения
            unit = "часов" if time <= 48 else "дней"
            if time > 48:
                time_value = time // 24
                unit = "дней"
            else:
                time_value = time
                unit = "часов"
            
            time_detail = f"Срок: {time_value} {unit}"
            if time_detail not in task_state.clarified_details:
                task_state.clarified_details.append(time_detail)
                updated = True
                print(f"🕐 Добавлен срок: {time_value} {unit}")
        
        # Обновляем ограничения
        for key, value in extracted_info["constraints"].items():
            if key not in task_state.constraints or task_state.constraints[key] != value:
                task_state.constraints[key] = value
                updated = True
                print(f"🔒 Добавлено ограничение: {key} = {value}")
        
        # Ограничиваем количество деталей
        if len(task_state.clarified_details) > 20:
            task_state.clarified_details = task_state.clarified_details[-20:]
            print("📝 Количество деталей ограничено 20 элементами")
        
        return updated
    
    def reset_task_state(self, conversation_id: int) -> bool:
        """
        Сбрасывает состояние задачи.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            True если сброс успешен, иначе False
        """
        return self.conversation_manager.reset_task_state(conversation_id)
    
    def get_goal(self, conversation_id: int) -> str:
        """
        Получает цель диалога.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Цель диалога или пустая строка
        """
        task_state = self.get_task_state(conversation_id)
        return task_state.goal if task_state else ""
    
    def get_clarified_details(self, conversation_id: int) -> List[str]:
        """
        Получает уточненные детали диалога.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Список уточненных деталей
        """
        task_state = self.get_task_state(conversation_id)
        return task_state.clarified_details if task_state else []
    
    def get_constraints(self, conversation_id: int) -> Dict[str, Any]:
        """
        Получает ограничения диалога.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Словарь ограничений
        """
        task_state = self.get_task_state(conversation_id)
        return task_state.constraints if task_state else {}
    
    def format_task_state_for_prompt(self, conversation_id: int) -> str:
        """
        Форматирует состояние задачи для включения в промпт.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Отформатированное состояние задачи
        """
        task_state = self.get_task_state(conversation_id)
        if not task_state:
            return ""
        
        lines = []
        
        if task_state.goal:
            lines.append(f"🎯 **Цель диалога:** {task_state.goal}")
        
        if task_state.clarified_details:
            lines.append("")
            lines.append("📋 **Уточненные детали:**")
            for detail in task_state.clarified_details[-10:]:  # Последние 10 деталей
                lines.append(f"  • {detail}")
        
        if task_state.constraints:
            lines.append("")
            lines.append("🔒 **Ограничения:**")
            for key, value in task_state.constraints.items():
                lines.append(f"  • {key}: {value}")
        
        return "\n".join(lines)
    
    def update_config(self, **kwargs):
        """
        Обновляет конфигурацию менеджера состояния задачи.
        
        Args:
            **kwargs: Параметры конфигурации
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_config(self) -> TaskStateConfig:
        """
        Возвращает текущую конфигурацию.
        
        Returns:
            Конфигурация менеджера состояния задачи
        """
        return self.config


# Глобальный экземпляр менеджера состояния задачи
_task_state_manager = None

def get_task_state_manager(conversation_manager: Optional[ConversationManager] = None) -> TaskStateManager:
    """
    Возвращает глобальный экземпляр менеджера состояния задачи.
    
    Args:
        conversation_manager: Менеджер диалогов (опционально)
        
    Returns:
        Менеджер состояния задачи
    """
    global _task_state_manager
    if _task_state_manager is None:
        _task_state_manager = TaskStateManager()
        if conversation_manager:
            _task_state_manager.conversation_manager = conversation_manager
    return _task_state_manager