"""
Утилиты для работы с базой данных.
Создание подключений, управление транзакциями.
"""

import sqlite3
import json
import threading
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime

from utils.config import settings


# Alias for backward compatibility
DatabaseManager = DatabaseConnection


class DatabaseConnection:
    """Управление подключениями к базе данных."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._connection_pool = {}
        self._initialized = True
        
        # Инициализируем базу данных при первом подключении
        self._init_database()
    
    def _init_database(self) -> None:
        """Инициализирует базу данных, создавая необходимые таблицы."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Создаем таблицу диалогов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT DEFAULT 'Новый диалог',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Создаем таблицу сообщений
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
            
            # Создаем таблицу состояний задач
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    clarified_details JSON DEFAULT '[]',
                    constraints JSON DEFAULT '{}',
                    goal TEXT DEFAULT '',
                    last_summary TEXT DEFAULT '',
                    message_count INTEGER DEFAULT 0,
                    last_summarized_at_message INTEGER DEFAULT 0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            ''')
            
            # Создаем индексы для ускорения поиска
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
                ON messages(conversation_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                ON messages(timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversations_active 
                ON conversations(active)
            ''')
            
            conn.commit()
            cursor.close()
            
            print("✅ База данных инициализирована")
            
        except sqlite3.Error as e:
            print(f"❌ Ошибка инициализации базы данных: {e}")
            raise
    
    def get_connection(self, db_path: Optional[str] = None) -> sqlite3.Connection:
        """
        Возвращает подключение к базе данных.
        
        Args:
            db_path: Путь к файлу базы данных (опционально)
            
        Returns:
            Подключение к SQLite базе данных
        """
        if db_path is None:
            # Используем путь из настроек или по умолчанию
            if settings.database_url.startswith("sqlite:///"):
                db_path = settings.database_url.replace("sqlite:///", "")
            else:
                db_path = "conversations.db"
        
        # Создаем новое подключение
        conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=10.0
        )
        
        # Включаем поддержку внешних ключей
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Настраиваем подключение для возврата словарей
        conn.row_factory = sqlite3.Row
        
        return conn
    
    @contextmanager
    def transaction(self, db_path: Optional[str] = None):
        """
        Контекстный менеджер для работы с транзакциями.
        
        Args:
            db_path: Путь к файлу базы данных (опционально)
            
        Yields:
            Курсор базы данных
        """
        conn = self.get_connection(db_path)
        cursor = conn.cursor()
        
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
    
    def execute_query(self, query: str, params: Tuple = (), 
                     db_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Выполняет SQL запрос и возвращает результаты.
        
        Args:
            query: SQL запрос
            params: Параметры запроса
            db_path: Путь к файлу базы данных
            
        Returns:
            Список словарей с результатами
        """
        with self.transaction(db_path) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Преобразуем строки в словари
            return [dict(row) for row in rows]
    
    def execute_update(self, query: str, params: Tuple = (),
                      db_path: Optional[str] = None) -> int:
        """
        Выполняет SQL запрос на обновление.
        
        Args:
            query: SQL запрос
            params: Параметры запроса
            db_path: Путь к файлу базы данных
            
        Returns:
            Количество измененных строк
        """
        with self.transaction(db_path) as cursor:
            cursor.execute(query, params)
            return cursor.rowcount


# Глобальный экземпляр для работы с базой данных
db = DatabaseConnection()


def json_serializer(obj: Any) -> str:
    """Сериализатор для JSON с поддержкой datetime."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def json_deserializer(data: str) -> Any:
    """Десериализатор JSON."""
    return json.loads(data)


class DatabaseModel:
    """Базовый класс для моделей базы данных."""
    
    @classmethod
    def create_table(cls):
        """Создает таблицу для модели."""
        raise NotImplementedError("Метод create_table должен быть реализован")
    
    @classmethod
    def from_db(cls, db_data: Dict[str, Any]) -> 'DatabaseModel':
        """Создает объект из данных базы данных."""
        raise NotImplementedError("Метод from_db должен быть реализован")
    
    def to_db(self) -> Dict[str, Any]:
        """Преобразует объект в данные для базы данных."""
        raise NotImplementedError("Метод to_db должен быть реализован")