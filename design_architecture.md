# Дизайн архитектуры новых функций для Logsit Agent

## Цель
Добавить следующие функции:
1. Поддержка нескольких диалогов с переключением между ними
2. Автоматическая суммаризация после 10 сообщений пользователя и далее каждые 10 сообщений
3. "Память задачи" (task state) для отслеживания контекста диалога

## Архитектурные компоненты

### 1. Модуль управления диалогами (ConversationManager)

**Ответственность:**
- Создание, хранение и управление несколькими диалогами
- Переключение между активными диалогами
- Персистентное хранение в SQLite базе данных

**Структура базы данных:**
```
dialogs
├── conversations
│   ├── id INTEGER PRIMARY KEY
│   ├── title TEXT (автогенерируемый заголовок)
│   ├── created_at DATETIME DEFAULT CURRENT_TIMESTAMP
│   ├── updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
│   └── active BOOLEAN DEFAULT 1
├── messages
│   ├── id INTEGER PRIMARY KEY
│   ├── conversation_id INTEGER REFERENCES conversations(id)
│   ├── role TEXT (user/assistant)
│   ├── content TEXT
│   ├── timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
│   └── is_summary BOOLEAN DEFAULT 0
└── task_states
    ├── id INTEGER PRIMARY KEY
    ├── conversation_id INTEGER REFERENCES conversations(id)
    ├── clarified_details JSON (что пользователь уже уточнил)
    ├── constraints JSON (ограничения/термины)
    ├── goal TEXT (цель диалога)
    ├── last_summary TEXT (последняя суммаризация)
    └── updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
```

### 2. Модуль суммаризации (Summarizer)

**Ответственность:**
- Автоматическая суммаризация истории сообщений
- Триггер каждые 10 сообщений пользователя
- Сохранение суммаризации для context window LLM
- Использование LLM для генерации лаконичной выжимки

**Алгоритм:**
1. Счетчик сообщений пользователя на диалог
2. После 10-го сообщения → триггер суммаризации
3. Генерация краткой выжимки (3-5 предложений)
4. Сохранение в task_state.last_summary
5. Использование summary в контексте для последующих ответов

### 3. Модуль Task State (TaskStateManager)

**Ответственность:**
- Отслеживание состояния задачи в диалоге
- Экстракция уточнений, ограничений и цели
- Автоматическое обновление на основе анализа сообщений
- Предоставление контекста для LLM

**Структура данных:**
```python
{
    "clarified_details": [
        "Маршрут включает Москву, Санкт-Петербург, Казань",
        "Вес груза: 150 кг",
        "Требуется доставка до 2 дней"
    ],
    "constraints": {
        "max_cities": 5,
        "max_weight": 1000,
        "time_limit": "48 часов"
    },
    "goal": "Рассчитать оптимальный маршрут и стоимость доставки между указанными городами",
    "last_summary": "Пользователь запросил расчет маршрута...",
    "message_count": 15,
    "last_summarized_at_message": 10
}
```

### 4. Модификация RoutePlannerAgent

**Изменения:**
- Добавление ссылки на ConversationManager
- Использование текущего диалога из контекста
- Внедрение summary в промпт LLM
- Обновление task state после обработки сообщений
- Проверка триггеров суммаризации

**Обновленный промпт LLM:**
```
[Текущий диалог: {title}]
[Контекст задачи: {task_state.goal}]
[Уточнения: {clarified_details}]
[Ограничения: {constraints}]
[Краткая сводка: {last_summary}]
{стандартный промпт}
```

### 5. Обновление UI (app.py)

**Новые элементы интерфейса:**
- Список диалогов с заголовками
- Кнопка "Новый диалог"
- Переключение между диалогами (tabs или dropdown)
- Отображение текущего статуса диалога (количество сообщений, задача)
- Индикатор когда доступна новая суммаризация

**Поток данных:**
```
Пользователь → UI → ConversationManager → RoutePlannerAgent → LLM + MCP/RAG
           ↖               ↖
          TaskStateManager    Summarizer
```

## Последовательность внедрения

### Этап 1: Базовая инфраструктура
1. Создать `conversation_manager.py`
2. Создать `summarizer.py` 
3. Создать `task_state.py`
4. Обновить `requirements.txt` при необходимости

### Этап 2: Интеграция с агентом
1. Модифицировать `RoutePlannerAgent` для использования новых компонентов
2. Внедрить логику суммаризации каждые 10 сообщений
3. Добавить task state в контекст промпта

### Этап 3: UI обновления
1. Расширить `app.py` для поддержки множества диалогов
2. Добавить интерфейс переключения диалогов
3. Визуализировать task state (опционально)

### Этап 4: Тестирование и доработка
1. Минимальные интеграционные тесты
2. Тестирование персистентности данных
3. Проверка производительности с учетом summary

## Технические детали

### SQLite Схема
```sql
-- conversations table
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT DEFAULT 'Новый диалог',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT 1
);

-- messages table  
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_summary BOOLEAN DEFAULT 0,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

-- task_states table
CREATE TABLE task_states (
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
);
```

### Алгоритм суммаризации
```python
def should_summarize(conversation_id: int) -> bool:
    """
    Проверяет, нужно ли суммировать диалог.
    Возвращает True если:
    - Сообщений пользователя >= 10 И
    - С момента последней суммаризации прошло >= 10 сообщений пользователя
    """
    current_count = get_user_message_count(conversation_id)
    last_summarized = get_last_summarized_count(conversation_id)
    
    if current_count >= 10 and (current_count - last_summarized) >= 10:
        return True
    return False
```

### Обновленный процесс обработки
1. Пользователь отправляет сообщение
2. `ConversationManager` добавляет сообщение в БД
3. `TaskStateManager` анализирует сообщение и обновляет task state
4. `Summarizer` проверяет триггер суммаризации
5. Если нужно → генерирует summary через LLM → обновляет task state
6. `RoutePlannerAgent` строит промпт с контекстом (summary + task state)
7. LLM обрабатывает запрос с полным контекстом
8. Ответ сохраняется в БД
9. UI обновляется

## Риски и ограничения

### Риски:
1. Увеличение размера контекста LLM за счет summary
2. Производительность SQLite при большом количестве диалогов
3. Качество автосуммаризации может быть нестабильным

### Ограничения:
1. Суммаризация использует LLM API → дополнительные затраты
2. Task state extraction требует надежного парсинга
3. Сохранность данных зависит от SQLite

## Альтернативы

### Для суммаризации:
- Вместо LLM использовать extractive summarization (текстовые алгоритмы)
- Использовать более дешевые модели для summary

### Для хранения:
- Вместо SQLite использовать Redis (но требует инфраструктуры)
- Хранить данные в памяти с периодической сериализацией

## Дальнейшее развитие

### Возможные улучшения:
1. Ручное редактирование task state пользователем
2. Темы диалогов (логистика, документы, расчеты)
3. Экспорт истории диалогов
4. Шаблоны task state для типовых задач
5. Аналитика по диалогам

### Интеграции:
1. Веб-хуки для уведомлений о важных изменениях
2. API для управления диалогами
3. Интеграция с внешними CRM системами