#!/bin/bash
# Скрипт для запуска enhanced приложения с поддержкой локальной модели

echo "========================================"
echo "Запуск Logsit Agent с локальной моделью"
echo "========================================"

# Проверка Ollama
echo "🔍 Проверка сервера Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Сервер Ollama доступен"
    
    # Проверка модели
    if curl -s http://localhost:11434/api/tags | grep -q "llama3.2"; then
        echo "✅ Модель llama3.2 доступна"
    else
        echo "⚠️ Модель llama3.2 не найдена"
        echo "Для загрузки модели выполните: ollama pull llama3.2:3b"
    fi
else
    echo "⚠️ Сервер Ollama недоступен"
    echo "Запустите сервер: ollama serve &"
    echo "Или используйте режим OpenRouter (снимите чекбокс 'локальная модель')"
fi

echo ""
echo "🚀 Запуск приложения..."
echo "📺 Интерфейс будет доступен по адресу: http://localhost:7872"
echo "💡 Для выхода нажмите Ctrl+C"
echo ""

# Запуск приложения
cd "$(dirname "$0")"
python3 route_planner/enhanced_app.py