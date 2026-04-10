# 🚀 Запуск проекта

## Требования

- Python 3.10 или выше
- API ключи:
  - OpenRouter API ключ (для LLM)
  - Yandex Maps API ключ (для геокодирования)

## Пошаговая инструкция

### 1. Подготовка окружения

```bash
# Перейдите в директорию проекта
cd logsit_agent

# Создайте виртуальное окружение (если ещё не создано)
python -m venv venv

# Активируйте виртуальное окружение
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установите зависимости
pip install -r requirements.txt
```

### 2. Настройка API ключей

Создайте файл `.env` в корне проекта:

```bash
cat > .env << EOF
OPENROUTER_API_KEY=your_openrouter_api_key_here
YANDEX_MAPS_API_KEY=your_yandex_maps_api_key_here
EOF
```

#### Получение API ключей:

**OpenRouter:**
1. Зайдите на https://openrouter.ai/
2. Создайте аккаунт
3. Перейдите в раздел API Keys
4. Создайте новый ключ

**Yandex Maps:**
1. Зайдите на https://developer.tech.yandex.ru/
2. Создайте аккаунт
3. Перейдите в раздел "Кабинет разработчика"
4. Создайте новый проект и получите API ключ для Geocoder

### 3. Запуск приложения

```bash
# Перейдите в папку с кодом
cd route_planner

# Запустите Streamlit приложение
streamlit run app.py
```

Или используйте скрипт запуска (Linux/Mac):
```bash
chmod +x run.sh
./run.sh
```

### 4. Проверка работы

1. Откройте браузер и перейдите по адресу `http://localhost:8501`
2. Введите тестовый запрос: "Найди маршрут между Москвой и Санкт-Петербургом"
3. Агент должен показать расстояние между городами

## Скрипт запуска (run.sh)

```bash
#!/bin/bash

# Проверка наличия Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python 3.10 или выше."
    exit 1
fi

# Проверка наличия .env файла
if [ ! -f .env ]; then
    echo "❌ Файл .env не найден. Создайте его с API ключами."
    echo "Пример:"
    echo "OPENROUTER_API_KEY=your_key_here"
    echo "YANDEX_MAPS_API_KEY=your_key_here"
    exit 1
fi

# Активация виртуального окружения
if [ -d "venv" ]; then
    echo "📦 Активация виртуального окружения..."
    source venv/bin/activate
else
    echo "❌ Виртуальное окружение не найдено."
    echo "Создайте его командой: python3 -m venv venv"
    exit 1
fi

# Проверка установки зависимостей
echo "📦 Проверка зависимостей..."
pip show streamlit > /dev/null 2>&1
if [ $? -ne 0 ]; then echo "📦 Установка зависимостей..."; pip install -r requirements.txt; fi

# Запуск приложения
echo "🚀 Запуск приложения..."
cd route_planner
streamlit run app.py
```

## Альтернативные способы запуска

### Запуск только MCP сервера (для тестирования)

```bash
cd route_planner
python mcp_server.py
```

### Запуск агента без UI

```bash
cd route_planner
python -c "
import asyncio
from agent import RoutePlannerAgent

async def main():
    agent = RoutePlannerAgent()
    await agent.connect_mcp()
    response = await agent.process_message('Найди маршрут между Москвой и Санкт-Петербургом')
    print(response)
    await agent.disconnect_mcp()

asyncio.run(main())
"
```

## Устранение неполадок

### Ошибка: "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit
```

### Ошибка: "API ключ Яндекс.Карт не настроен"
Проверьте файл `.env` и убедитесь, что:
1. Файл находится в корне проекта
2. Переменная `YANDEX_MAPS_API_KEY` содержит корректный ключ
3. Ключ активен в кабинете разработчика Яндекс

### Ошибка: "MCP сессия не инициализирована"
Проверьте, что:
1. Python может импортировать модуль `mcp`
2. Файл `mcp_server.py` существует в папке `route_planner`
3. Нет синтаксических ошибок в `mcp_server.py`

### Ошибка подключения к OpenRouter
Проверьте, что:
1. Переменная `OPENROUTER_API_KEY` в `.env` корректна
2. У вас есть доступ к API OpenRouter
3. Интернет-соединение стабильно

## Дополнительная информация

- Приложение запускается на порту 8501 по умолчанию
- История сообщений сохраняется в сессии Streamlit
- Логи MCP вызовов доступны в боковой панели
- Для изменения порта используйте: `streamlit run app.py --server.port 8080`
