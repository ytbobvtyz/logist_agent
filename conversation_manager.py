#!/usr/bin/env python3
"""
Модуль управления диалогами.
Поддержка нескольких диалогов, переключение между ними, хранение в БД.
"""

import sqlite3
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class MessageRole(Enum):
    """Роли участников диалога."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Conversation:
    """Модель диалога."""
    id: int = None
    title: str = "Новый диалог"
    created_at: str = None
    updated_at: str = None
    active: bool = True
    message_count: int = 0
    user_message_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class Message:
    """Модель сообщения."""
    id: int = None
    conversation_id: int = None
    role: str = None
    content: str = ""
    timestamp: str = None
    is_summary: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskState:
    """Модель состояния задачи."""
    id: int = None
    conversation_id: int = None
    clarified_details: List[str] = None
    constraints: Dict[str, Any] = None
    goal: str = ""
    last_summary: str = ""
    message_count: int = 0
    last_summarized_at_message: int = 0
    updated_at: str = None
    
    def __post_init__(self):
        if self.clarified_details is None:
            self.clarified_details = []
        if self.constraints is None:
            self.constraints = {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Конвертируем списки и словари в JSON строки
        data['clarified_details'] = json.dumps(self.clarified_details, ensure_ascii=False)
        data['constraints'] = json.dumps(self.constraints, ensure_ascii=False)
        return data
    
    @classmethod
    def from_db(cls, db_data: Dict[str, Any]) -> 'TaskState':
        """Создает объект из данных БД."""
        clarified_details = []
        constraints = {}
        
        if db_data.get('clarified_details'):
            try:
                clarified_details = json.loads(db_data['clarified_details'])
            except json.JSONDecodeError:
                clarified_details = []
        
        if db_data.get('constraints'):
            try:
                constraints = json.loads(db_data['constraints'])
            except json.JSONDecodeError:
                constraints = {}
        
        return cls(
            id=db_data.get('id'),
            conversation_id=db_data.get('conversation_id'),
            clarified_details=clarified_details,
            constraints=constraints,
            goal=db_data.get('goal', ''),
            last_summary=db_data.get('last_summary', ''),
            message_count=db_data.get('message_count', 0),
            last_summarized_at_message=db_data.get('last_summarized_at_message', 0),
            updated_at=db_data.get('updated_at')
        )


class ConversationManager:
    """Менеджер диалогов с поддержкой нескольких контекстов."""
    
    def __init__(self, db_path: str = "conversations.db"):
        """
        Инициализация менеджера диалогов.
        
        Args:
            db_path: Путь к файлу SQLite базы данных
        """
        self.db_path = db_path
        self._current_conversation_id: Optional[int] = None
        self._init_database()
    
    def _init_database(self):
        """Инициализация таблиц базы данных."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблица диалогов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'Новый диалог',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                active BOOLEAN DEFAULT 1,
                message_count INTEGER DEFAULT 0,
                user_message_count INTEGER DEFAULT 0
            )
        ''')
        
        # Таблица сообщений
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_summary BOOLEAN DEFAULT 0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        ''')
        
        # Индекс для быстрого поиска сообщений по диалогу
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
            ON messages(conversation_id)
        ''')
        
        # Индекс для сортировки по времени
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
            ON messages(timestamp)
        ''')
        
        # Таблица состояний задач
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS task_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL UNIQUE,
                clarified_details TEXT DEFAULT '[]',
                constraints TEXT DEFAULT '{}',
                goal TEXT DEFAULT '',
                last_summary TEXT DEFAULT '',
                message_count INTEGER DEFAULT 0,
                last_summarized_at_message INTEGER DEFAULT 0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"✅ База данных диалогов инициализирована: {self.db_path}")
    
    def create_conversation(self, title: str = None) -> Conversation:
        """
        Создает новый диалог.
        
        Args:
            title: Заголовок диалога (автогенерируется если None)
            
        Returns:
            Созданный диалог
        """
        if not title:
            # Генерируем заголовок на основе текущей даты
            now = datetime.now()
            title = f"Диалог {now.strftime('%d.%m.%Y %H:%M')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (title)
            VALUES (?)
        ''', (title,))
        
        conversation_id = cursor.lastrowid
        
        # Создаем состояние задачи для диалога
        cursor.execute('''
            INSERT INTO task_states (conversation_id)
            VALUES (?)
        ''', (conversation_id,))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Создан диалог #{conversation_id}: '{title}'")
        
        # Устанавливаем как текущий
        self._current_conversation_id = conversation_id
        
        return self.get_conversation(conversation_id)
    
    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """
        Получает диалог по ID.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Объект Conversation или None если не найден
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM conversations WHERE id = ?
        ''', (conversation_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return Conversation(
            id=row['id'],
            title=row['title'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            active=bool(row['active']),
            message_count=row['message_count'],
            user_message_count=row['user_message_count']
        )
    
    def get_all_conversations(self) -> List[Conversation]:
        """
        Получает все диалоги.
        
        Returns:
            Список всех диалогов, отсортированных по дате обновления (новые первыми)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM conversations 
            ORDER BY updated_at DESC, id DESC
        ''')
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append(Conversation(
                id=row['id'],
                title=row['title'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                active=bool(row['active']),
                message_count=row['message_count'],
                user_message_count=row['user_message_count']
            ))
        
        conn.close()
        return conversations
    
    def get_active_conversation(self) -> Optional[Conversation]:
        """
        Получает текущий активный диалог.
        
        Returns:
            Активный диалог или создает новый если нет активных
        """
        if self._current_conversation_id:
            conversation = self.get_conversation(self._current_conversation_id)
            if conversation and conversation.active:
                return conversation
        
        # Ищем активный диалог
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM conversations 
            WHERE active = 1 
            ORDER BY updated_at DESC 
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            conversation = Conversation(
                id=row['id'],
                title=row['title'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                active=bool(row['active']),
                message_count=row['message_count'],
                user_message_count=row['user_message_count']
            )
            self._current_conversation_id = conversation.id
            return conversation
        
        # Если нет активных диалогов, создаем новый
        return self.create_conversation()
    
    def set_active_conversation(self, conversation_id: int) -> bool:
        """
        Устанавливает активный диалог.
        
        Args:
            conversation_id: ID диалога для активации
            
        Returns:
            True если успешно, False если диалог не найден
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        # Обновляем активный статус
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Сбрасываем активный статус у всех диалогов
        cursor.execute('UPDATE conversations SET active = 0')
        
        # Устанавливаем активный статус выбранному диалогу
        cursor.execute('''
            UPDATE conversations 
            SET active = 1, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (conversation_id,))
        
        conn.commit()
        conn.close()
        
        self._current_conversation_id = conversation_id
        print(f"✅ Активирован диалог #{conversation_id}: '{conversation.title}'")
        
        return True
    
    def add_message(self, conversation_id: int, role: str, content: str, 
                   is_summary: bool = False) -> Message:
        """
        Добавляет сообщение в диалог.
        
        Args:
            conversation_id: ID диалога
            role: Роль отправителя (user/assistant/system)
            content: Текст сообщения
            is_summary: Является ли сообщение суммаризацией
            
        Returns:
            Добавленное сообщение
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Добавляем сообщение
        cursor.execute('''
            INSERT INTO messages (conversation_id, role, content, is_summary)
            VALUES (?, ?, ?, ?)
        ''', (conversation_id, role, content, 1 if is_summary else 0))
        
        message_id = cursor.lastrowid
        
        # Обновляем счетчики сообщений в диалоге
        update_fields = ['updated_at = CURRENT_TIMESTAMP', 'message_count = message_count + 1']
        
        if role == 'user' and not is_summary:
            update_fields.append('user_message_count = user_message_count + 1')
        
        cursor.execute(f'''
            UPDATE conversations 
            SET {', '.join(update_fields)}
            WHERE id = ?
        ''', (conversation_id,))
        
        # Обновляем счетчик сообщений в состоянии задачи
        if role == 'user' and not is_summary:
            cursor.execute('''
                UPDATE task_states 
                SET message_count = message_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = ?
            ''', (conversation_id,))
        
        conn.commit()
        conn.close()
        
        # Получаем добавленное сообщение
        message = self.get_message(message_id)
        
        if message and is_summary:
            print(f"📝 Добавлена суммаризация в диалог #{conversation_id}")
        elif message:
            print(f"💬 Добавлено сообщение {role} в диалог #{conversation_id}")
        
        return message
    
    def get_message(self, message_id: int) -> Optional[Message]:
        """Получает сообщение по ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM messages WHERE id = ?
        ''', (message_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return Message(
            id=row['id'],
            conversation_id=row['conversation_id'],
            role=row['role'],
            content=row['content'],
            timestamp=row['timestamp'],
            is_summary=bool(row['is_summary'])
        )
    
    def get_conversation_messages(self, conversation_id: int, 
                                 limit: int = None, 
                                 include_summaries: bool = False) -> List[Message]:
        """
        Получает сообщения диалога.
        
        Args:
            conversation_id: ID диалога
            limit: Максимальное количество сообщений (None = все)
            include_summaries: Включать ли сообщения-суммаризации
            
        Returns:
            Список сообщений, отсортированных по времени (старые первыми)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
            SELECT * FROM messages 
            WHERE conversation_id = ?
        '''
        
        params = [conversation_id]
        
        if not include_summaries:
            query += ' AND is_summary = 0'
        
        query += ' ORDER BY timestamp ASC, id ASC'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        cursor.execute(query, params)
        
        messages = []
        for row in cursor.fetchall():
            messages.append(Message(
                id=row['id'],
                conversation_id=row['conversation_id'],
                role=row['role'],
                content=row['content'],
                timestamp=row['timestamp'],
                is_summary=bool(row['is_summary'])
            ))
        
        conn.close()
        return messages
    
    def get_last_messages(self, conversation_id: int, count: int = 20) -> List[Message]:
        """
        Получает последние N сообщений диалога.
        
        Args:
            conversation_id: ID диалога
            count: Количество последних сообщений
            
        Returns:
            Список последних сообщений (новые первыми)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM messages 
            WHERE conversation_id = ? AND is_summary = 0
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
        ''', (conversation_id, count))
        
        messages = []
        for row in cursor.fetchall():
            messages.append(Message(
                id=row['id'],
                conversation_id=row['conversation_id'],
                role=row['role'],
                content=row['content'],
                timestamp=row['timestamp'],
                is_summary=bool(row['is_summary'])
            ))
        
        conn.close()
        # Возвращаем в хронологическом порядке
        return list(reversed(messages))
    
    def get_task_state(self, conversation_id: int) -> Optional[TaskState]:
        """
        Получает состояние задачи для диалога.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Объект TaskState или None если не найден
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM task_states WHERE conversation_id = ?
        ''', (conversation_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            # Если нет состояния, создаем его
            return self._create_task_state(conversation_id)
        
        return TaskState.from_db(dict(row))
    
    def _create_task_state(self, conversation_id: int) -> TaskState:
        """Создает начальное состояние задачи."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO task_states (conversation_id)
            VALUES (?)
        ''', (conversation_id,))
        
        conn.commit()
        conn.close()
        
        return TaskState(
            id=cursor.lastrowid,
            conversation_id=conversation_id
        )
    
    def update_task_state(self, conversation_id: int, **kwargs) -> TaskState:
        """
        Обновляет состояние задачи.
        
        Args:
            conversation_id: ID диалога
            **kwargs: Поля для обновления
            
        Returns:
            Обновленное состояние задачи
        """
        task_state = self.get_task_state(conversation_id)
        if not task_state:
            return None
        
        # Обновляем поля
        for key, value in kwargs.items():
            if hasattr(task_state, key):
                setattr(task_state, key, value)
        
        # Сохраняем в БД
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data = task_state.to_dict()
        data.pop('id', None)  # Убираем ID из данных для обновления
        
        # Строим запрос обновления
        fields = ', '.join([f"{k} = ?" for k in data.keys() if k != 'conversation_id'])
        values = [v for k, v in data.items() if k != 'conversation_id']
        values.append(conversation_id)
        
        cursor.execute(f'''
            UPDATE task_states 
            SET {fields}, updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
        ''', values)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Обновлено состояние задачи для диалога #{conversation_id}")
        
        return task_state
    
    def update_task_state_from_message(self, conversation_id: int, message_content: str, 
                                      is_user: bool = True) -> TaskState:
        """
        Обновляет состояние задачи на основе нового сообщения.
        Упрощенная версия - просто увеличивает счетчик.
        
        Args:
            conversation_id: ID диалога
            message_content: Текст сообщения
            is_user: Является ли сообщение от пользователя
            
        Returns:
            Обновленное состояние задачи
        """
        if not is_user:
            return self.get_task_state(conversation_id)
        
        # Получаем текущее состояние
        task_state = self.get_task_state(conversation_id)
        
        # Обновляем только счетчик сообщений
        # В реальной реализации здесь был бы анализ содержания сообщения
        return self.update_task_state(conversation_id, 
                                     message_count=task_state.message_count + 1)
    
    def should_summarize(self, conversation_id: int) -> Tuple[bool, int]:
        """
        Проверяет, нужно ли суммировать диалог.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Tuple (нужно_ли_суммировать, текущее_количество_сообщений_пользователя)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Получаем количество сообщений пользователя
        cursor.execute('''
            SELECT user_message_count FROM conversations WHERE id = ?
        ''', (conversation_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False, 0
        
        user_message_count = row[0]
        
        # Получаем информацию о последней суммаризации
        cursor.execute('''
            SELECT last_summarized_at_message FROM task_states 
            WHERE conversation_id = ?
        ''', (conversation_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row or row[0] is None:
            last_summarized = 0
        else:
            last_summarized = row[0]
        
        # Проверяем условие: >= 10 сообщений и с последней суммаризации прошло >= 10 сообщений
        if user_message_count >= 10 and (user_message_count - last_summarized) >= 10:
            return True, user_message_count
        
        return False, user_message_count
    
    def update_summary_info(self, conversation_id: int, summary_text: str) -> bool:
        """
        Обновляет информацию о последней суммаризации.
        
        Args:
            conversation_id: ID диалога
            summary_text: Текст суммаризации
            
        Returns:
            True если успешно
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Получаем текущее количество сообщений пользователя
        cursor.execute('''
            SELECT user_message_count FROM conversations WHERE id = ?
        ''', (conversation_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False
        
        user_message_count = row[0]
        
        # Обновляем состояние задачи
        cursor.execute('''
            UPDATE task_states 
            SET last_summary = ?, 
                last_summarized_at_message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
        ''', (summary_text, user_message_count, conversation_id))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Обновлена информация о суммаризации для диалога #{conversation_id}")
        
        return True
    
    def get_summary_context(self, conversation_id: int) -> str:
        """
        Получает контекст для LLM с учетом суммаризации и состояния задачи.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Строка контекста для промпта
        """
        task_state = self.get_task_state(conversation_id)
        conversation = self.get_conversation(conversation_id)
        
        if not task_state or not conversation:
            return ""
        
        context_parts = []
        
        # Информация о диалоге
        context_parts.append(f"Текущий диалог: {conversation.title}")
        context_parts.append(f"Сообщений в диалоге: {conversation.message_count}")
        
        # Информация о состоянии задачи
        if task_state.goal:
            context_parts.append(f"Цель диалога: {task_state.goal}")
        
        if task_state.clarified_details:
            details = "\n".join([f"- {detail}" for detail in task_state.clarified_details])
            context_parts.append(f"Уточнения пользователя:\n{details}")
        
        if task_state.constraints:
            constraints = "\n".join([f"- {k}: {v}" for k, v in task_state.constraints.items()])
            context_parts.append(f"Ограничения/термины:\n{constraints}")
        
        # Краткая сводка
        if task_state.last_summary:
            context_parts.append(f"Краткая сводка диалога:\n{task_state.last_summary}")
        
        return "\n\n".join(context_parts)
    
    def delete_conversation(self, conversation_id: int) -> bool:
        """
        Удаляет диалог и все связанные данные.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            True если успешно
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Удаляем связанные данные
            cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
            cursor.execute('DELETE FROM task_states WHERE conversation_id = ?', (conversation_id,))
            cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
            
            conn.commit()
            
            # Если удаляли текущий диалог, сбрасываем указатель
            if self._current_conversation_id == conversation_id:
                self._current_conversation_id = None
            
            print(f"🗑️ Удален диалог #{conversation_id}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка удаления диалога #{conversation_id}: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def clear_all_conversations(self) -> bool:
        """Удаляет все диалоги."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM messages')
            cursor.execute('DELETE FROM task_states')
            cursor.execute('DELETE FROM conversations')
            
            conn.commit()
            
            self._current_conversation_id = None
            
            print("🗑️ Удалены все диалоги")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка удаления всех диалогов: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получает статистику по диалогам."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Количество диалогов
        cursor.execute('SELECT COUNT(*) FROM conversations')
        stats['total_conversations'] = cursor.fetchone()[0]
        
        # Количество активных диалогов
        cursor.execute('SELECT COUNT(*) FROM conversations WHERE active = 1')
        stats['active_conversations'] = cursor.fetchone()[0]
        
        # Общее количество сообщений
        cursor.execute('SELECT COUNT(*) FROM messages')
        stats['total_messages'] = cursor.fetchone()[0]
        
        # Количество сообщений пользователя
        cursor.execute('SELECT SUM(user_message_count) FROM conversations')
        stats['total_user_messages'] = cursor.fetchone()[0] or 0
        
        # Среднее количество сообщений на диалог
        if stats['total_conversations'] > 0:
            stats['avg_messages_per_conversation'] = stats['total_messages'] / stats['total_conversations']
        else:
            stats['avg_messages_per_conversation'] = 0
        
        conn.close()
        
        return stats


# Синглтон для глобального доступа
_conversation_manager: Optional[ConversationManager] = None

def get_conversation_manager(db_path: str = "conversations.db") -> ConversationManager:
    """Получает глобальный экземпляр менеджера диалогов."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager(db_path)
    return _conversation_manager