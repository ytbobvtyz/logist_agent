# Развертывание Logsit Agent

## 🚀 Быстрое развертывание

### Требования
- Python 3.10+
- 2 ГБ оперативной памяти
- 1 ГБ свободного места на диске
- Доступ в интернет для API OpenRouter

### Локальная установка

```bash
# 1. Клонирование репозитория
git clone <репозиторий>
cd logsit_agent

# 2. Создание виртуального окружения
python -m venv venv

# 3. Активация виртуального окружения
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. Установка зависимостей
pip install -r requirements.txt

# 5. Настройка окружения
cp .env.example .env
# Отредактируйте .env файл, добавив ваш API ключ OpenRouter

# 6. Запуск приложения
python app/main.py
```

Приложение будет доступно по адресу: http://localhost:7872

## 🐳 Docker развертывание

### Сборка образа

```bash
# Сборка Docker образа
docker build -t logsit-agent .

# Запуск контейнера
docker run -d \
  -p 7872:7872 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.env:/app/.env \
  --name logsit-agent \
  logsit-agent
```

### Docker Compose

Создайте файл `docker-compose.yml`:

```yaml
version: '3.8'

services:
  logsit-agent:
    build: .
    ports:
      - "7872:7872"
    volumes:
      - ./data:/app/data
      - ./config/.env:/app/.env
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - YANDEX_MAPS_API_KEY=${YANDEX_MAPS_API_KEY}
    restart: unless-stopped
```

Запуск:
```bash
docker-compose up -d
```

## ☁️ Облачное развертывание

### Render.com

1. Создайте новый Web Service
2. Подключите GitHub репозиторий
3. Настройки:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app/main.py`
   - **Environment Variables**: Добавьте `OPENROUTER_API_KEY`

### Railway.app

1. Создайте новый проект
2. Добавьте переменные окружения:
   ```env
   OPENROUTER_API_KEY=ваш_ключ
   PORT=7872
   ```
3. Railway автоматически определит и запустит приложение

### Heroku

```bash
# Создание приложения
heroku create logsit-agent

# Добавление переменных окружения
heroku config:set OPENROUTER_API_KEY=ваш_ключ

# Развертывание
git push heroku main
```

## 🔧 Конфигурация

### Файл .env

```env
# Обязательные настройки
OPENROUTER_API_KEY=sk-or-v1-ваш_ключ_openrouter

# Опциональные настройки
YANDEX_MAPS_API_KEY=ваш_ключ_yandex

# Настройки приложения
APP_HOST=0.0.0.0
APP_PORT=7872
DEBUG=false

# База данных
DATABASE_URL=sqlite:///data/conversations.db

# MCP серверы (JSON строка)
MCP_SERVERS={"route_planner": "python services/mcp_server.py"}

# RAG настройки
RAG_INDEX_PATH=data/faiss_index
RAG_METADATA_PATH=data/metadata.db

# LLM настройки
DEFAULT_MODEL=openrouter/free
SUMMARIZATION_MODEL=openrouter/free

# Настройки диалогов
MAX_CONVERSATIONS=100
SUMMARIZATION_TRIGGER=10
```

### Переменные окружения

| Переменная | Обязательно | По умолчанию | Описание |
|------------|-------------|--------------|----------|
| `OPENROUTER_API_KEY` | Да | - | API ключ OpenRouter |
| `YANDEX_MAPS_API_KEY` | Нет | - | API ключ Яндекс.Карт |
| `APP_HOST` | Нет | `0.0.0.0` | Хост для веб-сервера |
| `APP_PORT` | Нет | `7872` | Порт для веб-сервера |
| `DEBUG` | Нет | `false` | Режим отладки |
| `DATABASE_URL` | Нет | `sqlite:///conversations.db` | URL базы данных |
| `MCP_SERVERS` | Нет | `{}` | JSON словарь MCP серверов |
| `RAG_INDEX_PATH` | Нет | `faiss_index` | Путь к FAISS индексу |
| `RAG_METADATA_PATH` | Нет | `metadata.db` | Путь к базе метаданных RAG |

## 🗄️ База данных

### Локальная SQLite

По умолчанию используется SQLite база данных `conversations.db` в корне проекта.

#### Миграции
```bash
# Создание резервной копии
cp conversations.db conversations.db.backup

# Восстановление из резервной копии
cp conversations.db.backup conversations.db
```

### Внешняя база данных

Для использования PostgreSQL или MySQL:

1. Измените `DATABASE_URL` в `.env`:
   ```env
   # PostgreSQL
   DATABASE_URL=postgresql://user:password@localhost/logsit
   
   # MySQL
   DATABASE_URL=mysql://user:password@localhost/logsit
   ```

2. Установите соответствующий драйвер:
   ```bash
   # PostgreSQL
   pip install psycopg2-binary
   
   # MySQL
   pip install mysqlclient
   ```

## 📊 Мониторинг и логирование

### Логи приложения

Логи выводятся в консоль и могут быть перенаправлены в файл:

```bash
# Запись логов в файл
python app/main.py 2>&1 | tee logsit.log

# Ротация логов с logrotate
# /etc/logrotate.d/logsit-agent
/home/user/logsit-agent/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

### Мониторинг здоровья

Эндпоинт здоровья (планируется):
```
GET /health
```

Ответ:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "services": {
    "llm": true,
    "database": true,
    "mcp": false,
    "rag": true
  }
}
```

### Метрики Prometheus (планируется)

```python
# Пример метрик
from prometheus_client import Counter, Gauge

requests_total = Counter('logsit_requests_total', 'Total requests')
active_conversations = Gauge('logsit_active_conversations', 'Active conversations')
```

## 🔐 Безопасность

### API ключи
- Никогда не коммитьте `.env` файл в Git
- Используйте секреты в облачных средах
- Регулярно ротируйте ключи

### Веб-безопасность
- Приложение работает только по HTTP (для локального использования)
- Для продакшена рекомендуется использовать HTTPS через reverse proxy

### Reverse Proxy (Nginx)

```nginx
# /etc/nginx/sites-available/logsit-agent
server {
    listen 80;
    server_name logsit.example.com;
    
    location / {
        proxy_pass http://localhost:7872;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # SSL конфигурация (опционально)
    listen 443 ssl;
    ssl_certificate /etc/ssl/certs/logsit.crt;
    ssl_certificate_key /etc/ssl/private/logsit.key;
}
```

## 📈 Масштабирование

### Вертикальное масштабирование
- Увеличьте память для обработки больших RAG индексов
- Увеличьте CPU для параллельной обработки MCP вызовов

### Горизонтальное масштабирование
1. Вынесите базу данных в отдельный сервис
2. Используйте shared storage для RAG индексов
3. Настройте балансировщик нагрузки

### Кэширование
```python
# Пример кэширования с Redis (планируется)
import redis
from functools import lru_cache

redis_client = redis.Redis(host='localhost', port=6379, db=0)

@lru_cache(maxsize=100)
def get_cached_response(query: str):
    cached = redis_client.get(f"response:{query}")
    if cached:
        return cached.decode()
    return None
```

## 🔄 Резервное копирование

### Автоматическое резервное копирование

Создайте скрипт `backup.sh`:

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/logsit-agent"
DATE=$(date +%Y%m%d_%H%M%S)

# Создание директории для бэкапов
mkdir -p $BACKUP_DIR

# Резервное копирование базы данных
cp conversations.db $BACKUP_DIR/conversations_$DATE.db

# Резервное копирование RAG индекса
cp -r faiss_index $BACKUP_DIR/faiss_index_$DATE
cp metadata.db $BACKUP_DIR/metadata_$DATE.db

# Удаление старых бэкапов (старше 30 дней)
find $BACKUP_DIR -type f -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR/conversations_$DATE.db"
```

Добавьте в crontab:
```bash
# Ежедневное резервное копирование в 2:00
0 2 * * * /path/to/backup.sh
```

## 🚨 Аварийное восстановление

### Восстановление из бэкапа

```bash
# Остановка приложения
pkill -f "python app/main.py"

# Восстановление базы данных
cp /backups/logsit-agent/conversations_20240101_020000.db conversations.db

# Восстановление RAG индекса
cp -r /backups/logsit-agent/faiss_index_20240101_020000 faiss_index
cp /backups/logsit-agent/metadata_20240101_020000.db metadata.db

# Запуск приложения
python app/main.py
```

### Проверка целостности

```bash
# Проверка базы данных
python -c "
import sqlite3
conn = sqlite3.connect('conversations.db')
cursor = conn.cursor()
cursor.execute('PRAGMA integrity_check')
print(cursor.fetchone())
conn.close()
"

# Проверка RAG индекса
python -c "
from services.rag_service import RAGService
rag = RAGService()
print('RAG available:', rag.is_available())
stats = rag.get_index_stats()
print('Index stats:', stats)
"
```

## 📝 Чеклист развертывания

### Перед развертыванием
- [ ] API ключ OpenRouter настроен
- [ ] `.env` файл создан и настроен
- [ ] Зависимости установлены
- [ ] Порт 7872 свободен

### После развертывания
- [ ] Приложение доступно по http://localhost:7872
- [ ] MCP серверы подключены (проверьте статус)
- [ ] RAG индекс загружен
- [ ] Можно создать новый диалог
- [ ] Сообщения обрабатываются корректно

### Регулярное обслуживание
- [ ] Резервное копирование базы данных
- [ ] Очистка старых логов
- [ ] Обновление зависимостей
- [ ] Проверка доступности API ключей

---

Следуя этому руководству, вы сможете успешно развернуть и поддерживать Logsit Agent в различных средах.