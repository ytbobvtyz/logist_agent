#!/bin/bash

# Скрипт запуска умного ассистента логиста

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🗺️  Запуск умного ассистента логиста${NC}"
echo ""

# Проверка наличия Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 не найден. Установите Python 3.10 или выше.${NC}"
    exit 1
fi

# Проверка версии Python
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${YELLOW}⚠️  Рекомендуется Python 3.10 или выше. Текущая версия: $PYTHON_VERSION${NC}"
fi

# Проверка наличия .env файла
if [ ! -f .env ]; then
    echo -e "${RED}❌ Файл .env не найден.${NC}"
    echo ""
    echo "Создайте файл .env в корне проекта с API ключами:"
    echo ""
    echo "OPENROUTER_API_KEY=your_openrouter_api_key_here"
    echo "YANDEX_MAPS_API_KEY=your_yandex_maps_api_key_here"
    echo ""
    exit 1
fi

# Активация виртуального окружения
if [ -d "venv" ]; then
    echo -e "${GREEN}📦 Активация виртуального окружения...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}⚠️  Виртуальное окружение не найдено.${NC}"
    echo "Создайте его командой: python3 -m venv venv"
    echo ""
    read -p "Создать виртуальное окружение сейчас? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 -m venv venv
        source venv/bin/activate
    else
        exit 1
    fi
fi

# Проверка установки зависимостей
echo -e "${GREEN}📦 Проверка зависимостей...${NC}"
pip show streamlit > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}📦 Установка зависимостей...${NC}"
    pip install -r requirements.txt
fi

# Проверка наличия папки route_planner
if [ ! -d "route_planner" ]; then
    echo -e "${RED}❌ Папка route_planner не найдена.${NC}"
    exit 1
fi

# Проверка наличия файлов
if [ ! -f "route_planner/app.py" ]; then
    echo -e "${RED}❌ Файл route_planner/app.py не найден.${NC}"
    exit 1
fi

# Запуск приложения
echo ""
echo -e "${GREEN}🚀 Запуск приложения...${NC}"
echo -e "${GREEN}   Откройте браузер: http://localhost:8501${NC}"
echo ""
cd route_planner
streamlit run app.py
