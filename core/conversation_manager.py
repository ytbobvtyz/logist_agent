#!/usr/bin/env python3
"""
Модуль управления диалогами.
Поддержка нескольких диалогов, переключение между ними, хранение в БД.
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from utils.database import db, DatabaseModel


class MessageRole(Enum):
    """Роли участников диалога."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Conversation(DatabaseModel):
    """Модель диалога."""
    
    id: Optional[int] = None
    title: str = "Новый диалог"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    active: bool = True
    message_count: int = 0
    user_message_count: int = 0
    
    @classmethod
    def create_table(cls):
        """Создает таблицу диалогов (уже создана в utils.database)."""
        pass
    
    @classmethod
    def from_db(cls, db_data: Dict[str, Any]) -> 'Conversation':
        """Создает объект из данных БД."""
        return cls(
            id=db_data.get('id'),
            title=db_data.get('title', 'Новый диалог'),
            created_at=db_data.get('created_at'),
            updated_at=db_data.get('updated_at'),
            active=bool(db_data.get('active', 1)),
            message_count=db_data.get('message_count', 0),
            user_message_count=db_data.get('user_message_count', 0)
        )
    
    def to_db(self) -> Dict[str, Any]:
        """Преобразует объект в данные для БД."""
        return {
            'title': self.title,
            'active': 1 if self.active else 0,
            'message_count': self.message_count,
            'user_message_count': self.user_message_count
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует объект в словарь."""
        return asdict(self)


@dataclass 
class Message(DatabaseModel):
    """Модель сообщения."""
    
    id: Optional[int] = None
    conversation_id: Optional[int] = None
    role: str = ""
    content: str = ""
    timestamp: Optional[str] = None
    is_summary: bool = False
    
    @classmethod
    def create_table(cls):
        """Создает таблицу сообщений (уже создана в utils.database)."""
        pass
    
    @classmethod
    def from_db(cls, db_data: Dict[str, Any]) -> 'Message':
        """Создает объект из данных БД."""
        return cls(
            id=db_data.get('id'),
            conversation_id=db_data.get('conversation_id'),
            role=db_data.get('role', ''),
            content=db_data.get('content', ''),
            timestamp=db_data.get('timestamp'),
            is_summary=bool(db_data.get('is_summary', 0))
        )
    
    def to_db(self) -> Dict[str, Any]:
        """Преобразует объект в данные для БД."""
        return {
            'conversation_id': self.conversation_id,
            'role': self.role,
            'content': self.content,
            'is_summary': 1 if self.is_summary else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует объект в словарь."""
        return asdict(self)


@dataclass
class TaskState(DatabaseModel):
    """Модель состояния задачи."""
    
    id: Optional[int] = None
    conversation_id: Optional[int] = None
    clarified_details: List[str] = None
    constraints: Dict[str, Any] = None
    goal: str = ""
    last_summary: str = ""
    message_count: int = 0
    last_summarized_at_message: int = 0
    updated_at: Optional[str] = None
    
    def __post_init__(self):
        if self.clarified_details is None:
            self.clarified_details = []
        if self.constraints is None:
            self.constraints = {}
    
    @classmethod
    def create_table(cls):
        """Создает таблицу состояний задач (уже создана в utils.database)."""
        pass
    
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
    
    def to_db(self) -> Dict[str, Any]:
        """Преобразует объект в данные для БД."""
        return {
            'conversation_id': self.conversation_id,
            'clarified_details': json.dumps(self.clarified_details, ensure_ascii=False),
            'constraints': json.dumps(self.constraints, ensure_ascii=False),
            'goal': self.goal,
            'last_summary': self.last_summary,
            'message_count': self.message_count,
            'last_summarized_at_message': self.last_summarized_at_message
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует объект в словарь."""
        data = asdict(self)
        # Конвертируем списки и словари в JSON строки для сериализации
        data['clarified_details'] = json.dumps(self.clarified_details, ensure_ascii=False)
        data['constraints'] = json.dumps(self.constraints, ensure_ascii=False)
        return data


class ConversationManager:
    """Менеджер диалогов для управления несколькими беседами."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._active_conversation_id = None
    
    def create_conversation(self, title: Optional[str] = None) -> Conversation:
        """
        Создает новый диалог.
        
        Args:
            title: Заголовок диалога (опционально)
            
        Returns:
            Созданный диалог
        """
        if title is None:
            title = f"Диалог {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        conversation = Conversation(title=title)
        
        # Сохраняем в БД
        with db.transaction() as cursor:
            cursor.execute('''
                INSERT INTO conversations (title, active, created_at, updated_at)
                VALUES (?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (title,))
            
            conversation.id = cursor.lastrowid
        
        # Создаем начальное состояние задачи
        self._create_initial_task_state(conversation.id)
        
        print(f"✅ Создан новый диалог: {title} (ID: {conversation.id})")
        return conversation
    
    def _create_initial_task_state(self, conversation_id: int) -> None:
        """Создает начальное состояние задачи для диалога."""
        task_state = TaskState(conversation_id=conversation_id)
        
        with db.transaction() as cursor:
            cursor.execute('''
                INSERT INTO task_states 
                (conversation_id, clarified_details, constraints, goal, 
                 last_summary, message_count, last_summarized_at_message, updated_at)
                VALUES (?, '[]', '{}', '', '', 0, 0, CURRENT_TIMESTAMP)
            ''', (conversation_id,))
    
    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """
        Получает диалог по ID.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Диалог или None, если не найден
        """
        results = db.execute_query(
            '''
            SELECT c.*, 
                   COUNT(m.id) as message_count,
                   SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE c.id = ?
            GROUP BY c.id
            ''',
            (conversation_id,)
        )
        
        if not results:
            return None
        
        return Conversation.from_db(results[0])
    
    def get_all_conversations(self) -> List[Conversation]:
        """
        Получает все диалоги.
        
        Returns:
            Список всех диалогов
        """
        results = db.execute_query('''
            SELECT c.*, 
                   COUNT(m.id) as message_count,
                   SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        ''')
        
        return [Conversation.from_db(row) for row in results]
    
    def get_active_conversations(self) -> List[Conversation]:
        """
        Получает активные диалоги.
        
        Returns:
            Список активных диалогов
        """
        results = db.execute_query('''
            SELECT c.*, 
                   COUNT(m.id) as message_count,
                   SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE c.active = 1
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        ''')
        
        return [Conversation.from_db(row) for row in results]
    
    def update_conversation(self, conversation_id: int, **kwargs) -> bool:
        """
        Обновляет диалог.
        
        Args:
            conversation_id: ID диалога
            **kwargs: Поля для обновления
            
        Returns:
            True если обновление успешно, иначе False
        """
        if not kwargs:
            return False
        
        # Формируем запрос на обновление
        set_clause = ', '.join([f"{key} = ?" for key in kwargs.keys()])
        query = f'''
            UPDATE conversations 
            SET {set_clause}, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        '''
        
        params = list(kwargs.values()) + [conversation_id]
        
        try:
            rows_affected = db.execute_update(query, tuple(params))
            return rows_affected > 0
        except Exception as e:
            print(f"❌ Ошибка обновления диалога: {e}")
            return False
    
    def delete_conversation(self, conversation_id: int) -> bool:
        """
        Удаляет диалог.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            True если удаление успешно, иначе False
        """
        try:
            # Удаляем связанные сообщения
            db.execute_update(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            )
            
            # Удаляем состояние задачи
            db.execute_update(
                "DELETE FROM task_states WHERE conversation_id = ?",
                (conversation_id,)
            )
            
            # Удаляем диалог
            rows_affected = db.execute_update(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            
            return rows_affected > 0
            
        except Exception as e:
            print(f"❌ Ошибка удаления диалога: {e}")
            return False
    
    def add_message(self, conversation_id: int, role: str, 
                   content: str, is_summary: bool = False) -> Optional[Message]:
        """
        Добавляет сообщение в диалог.
        
        Args:
            conversation_id: ID диалога
            role: Роль отправителя
            content: Текст сообщения
            is_summary: Является ли сообщение суммаризацией
            
        Returns:
            Добавленное сообщение или None при ошибке
        """
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            is_summary=is_summary
        )
        
        try:
            with db.transaction() as cursor:
                cursor.execute('''
                    INSERT INTO messages 
                    (conversation_id, role, content, is_summary, timestamp)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (conversation_id, role, content, 1 if is_summary else 0))
                
                message.id = cursor.lastrowid
                
                # Обновляем время последнего изменения диалога
                cursor.execute('''
                    UPDATE conversations 
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (conversation_id,))
            
            # Обновляем счетчик сообщений в состоянии задачи
            self._update_message_count(conversation_id, role)
            
            return message
            
        except Exception as e:
            print(f"❌ Ошибка добавления сообщения: {e}")
            return None
    
    def _update_message_count(self, conversation_id: int, role: str) -> None:
        """Обновляет счетчик сообщений в состоянии задачи."""
        try:
            with db.transaction() as cursor:
                # Получаем текущее состояние
                cursor.execute(
                    "SELECT message_count FROM task_states WHERE conversation_id = ?",
                    (conversation_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    new_count = result[0] + 1
                    cursor.execute('''
                        UPDATE task_states 
                        SET message_count = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE conversation_id = ?
                    ''', (new_count, conversation_id))
                    
                    # Если это сообщение пользователя, обновляем счетчик для суммаризации
                    if role == 'user':
                        cursor.execute('''
                            UPDATE task_states 
                            SET last_summarized_at_message = last_summarized_at_message,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE conversation_id = ?
                        ''', (conversation_id,))
        
        except Exception as e:
            print(f"❌ Ошибка обновления счетчика сообщений: {e}")
    
    def get_messages(self, conversation_id: int, 
                    limit: Optional[int] = None) -> List[Message]:
        """
        Получает сообщения диалога.
        
        Args:
            conversation_id: ID диалога
            limit: Максимальное количество сообщений (опционально)
            
        Returns:
            Список сообщений
        """
        query = '''
            SELECT * FROM messages 
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        '''
        
        if limit:
            query += f" LIMIT {limit}"
        
        results = db.execute_query(query, (conversation_id,))
        return [Message.from_db(row) for row in results]
    
    def get_recent_messages(self, conversation_id: int, 
                           count: int = 10) -> List[Message]:
        """
        Получает последние сообщения диалога.
        
        Args:
            conversation_id: ID диалога
            count: Количество сообщений
            
        Returns:
            Список последних сообщений
        """
        results = db.execute_query('''
            SELECT * FROM messages 
            WHERE conversation_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (conversation_id, count))
        
        # Возвращаем в правильном порядке (от старых к новым)
        return [Message.from_db(row) for row in reversed(results)]
    
    def clear_conversation(self, conversation_id: int) -> bool:
        """
        Очищает все сообщения диалога.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            True если очистка успешна, иначе False
        """
        try:
            # Удаляем сообщения
            rows_affected = db.execute_update(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            )
            
            # Сбрасываем состояние задачи
            self.reset_task_state(conversation_id)
            
            # Обновляем время изменения диалога
            db.execute_update(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (conversation_id,)
            )
            
            return rows_affected > 0
            
        except Exception as e:
            print(f"❌ Ошибка очистки диалога: {e}")
            return False
    
    def get_task_state(self, conversation_id: int) -> Optional[TaskState]:
        """
        Получает состояние задачи для диалога.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Состояние задачи или None, если не найдено
        """
        results = db.execute_query(
            "SELECT * FROM task_states WHERE conversation_id = ?",
            (conversation_id,)
        )
        
        if not results:
            return None
        
        return TaskState.from_db(results[0])
    
    def update_task_state(self, conversation_id: int, **kwargs) -> bool:
        """
        Обновляет состояние задачи.
        
        Args:
            conversation_id: ID диалога
            **kwargs: Поля для обновления
            
        Returns:
            True если обновление успешно, иначе False
        """
        if not kwargs:
            return False
        
        # Проверяем существование состояния задачи
        task_state = self.get_task_state(conversation_id)
        if not task_state:
            # Создаем новое состояние
            task_state = TaskState(conversation_id=conversation_id)
            for key, value in kwargs.items():
                setattr(task_state, key, value)
            
            db_data = task_state.to_db()
            columns = ', '.join(db_data.keys())
            placeholders = ', '.join(['?' for _ in db_data])
            
            query = f'''
                INSERT INTO task_states ({columns}, updated_at)
                VALUES ({placeholders}, CURRENT_TIMESTAMP)
            '''
            
            try:
                db.execute_update(query, tuple(db_data.values()))
                return True
            except Exception as e:
                print(f"❌ Ошибка создания состояния задачи: {e}")
                return False
        else:
            # Обновляем существующее состояние
            for key, value in kwargs.items():
                setattr(task_state, key, value)
            
            db_data = task_state.to_db()
            set_clause = ', '.join([f"{key} = ?" for key in db_data.keys()])
            
            query = f'''
                UPDATE task_states 
                SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = ?
            '''
            
            params = list(db_data.values()) + [conversation_id]
            
            try:
                rows_affected = db.execute_update(query, tuple(params))
                return rows_affected > 0
            except Exception as e:
                print(f"❌ Ошибка обновления состояния задачи: {e}")
                return False
    
    def reset_task_state(self, conversation_id: int) -> bool:
        """
        Сбрасывает состояние задачи.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            True если сброс успешен, иначе False
        """
        return self.update_task_state(
            conversation_id,
            clarified_details=[],
            constraints={},
            goal="",
            last_summary="",
            message_count=0,
            last_summarized_at_message=0
        )
    
    def get_user_message_count(self, conversation_id: int) -> int:
        """
        Получает количество сообщений пользователя в диалоге.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Количество сообщений пользователя
        """
        results = db.execute_query('''
            SELECT COUNT(*) as count FROM messages 
            WHERE conversation_id = ? AND role = 'user'
        ''', (conversation_id,))
        
        return results[0]['count'] if results else 0
    
    def should_summarize(self, conversation_id: int) -> Tuple[bool, int]:
        """
        Проверяет, нужно ли суммировать диалог.
        
        Args:
            conversation_id: ID диалога
            
        Returns:
            Кортеж (нужна ли суммаризация, количество сообщений пользователя)
        """
        task_state = self.get_task_state(conversation_id)
        if not task_state:
            return False, 0
        
        user_count = self.get_user_message_count(conversation_id)
        
        # Суммаризируем каждые 10 сообщений пользователя
        if user_count >= 10 and (user_count - task_state.last_summarized_at_message) >= 10:
            return True, user_count
        
        return False, user_count
    
    @property
    def active_conversation_id(self) -> Optional[int]:
        """Возвращает ID активного диалога."""
        return self._active_conversation_id
    
    @active_conversation_id.setter
    def active_conversation_id(self, conversation_id: Optional[int]):
        """Устанавливает активный диалог."""
        self._active_conversation_id = conversation_id
        
        if conversation_id:
            # Деактивируем другие диалоги
            self.update_conversation(conversation_id, active=1)
            
            # Находим и деактивируем другие активные диалоги
            active_conversations = self.get_active_conversations()
            for conv in active_conversations:
                if conv.id != conversation_id:
                    self.update_conversation(conv.id, active=0)


# Глобальный экземпляр менеджера диалогов
_conversation_manager = None

def get_conversation_manager() -> ConversationManager:
    """Возвращает глобальный экземпляр менеджера диалогов."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager