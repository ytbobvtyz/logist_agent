# API Ключи для Logsit Agent

## Поддерживаемые провайдеры

### 1. DeepSeek (рекомендуется для РФ/СНГ)
- **Сайт**: https://platform.deepseek.com
- **Получить ключ**: https://platform.deepseek.com/api_keys
- **Модели**: `deepseek/deepseek-chat`, `deepseek/deepseek-coder`, `deepseek/deepseek-r1`
- **Статус**: Поддерживает РФ и СНГ
- **Цены**: Бесплатные тарифы доступны

### 2. OpenRouter
- **Сайт**: https://openrouter.ai
- **Получить ключ**: https://openrouter.ai/settings/keys
- **Модели**: Различные модели через единый API
- **Статус**: Могут быть региональные ограничения
- **Цены**: Есть бесплатные модели (ограниченные)

### 3. OpenAI
- **Сайт**: https://platform.openai.com
- **Получить ключ**: https://platform.openai.com/api-keys
- **Модели**: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`
- **Статус**: Могут быть региональные ограничения
- **Цены**: Платные

### 4. Qwen (Alibaba)
- **Сайт**: https://qwenlm.com
- **Получить ключ**: https://dashscope.aliyun.com/
- **Модели**: `qwen/qwen-plus`, `qwen/qwen-max`, `qwen/qwen-turbo`
- **Статус**: Поддерживает международный доступ
- **Цены**: Есть бесплатные квоты

### 5. Step Fun
- **Сайт**: https://stepfun.com
- **Получить ключ**: https://platform.stepfun.com
- **Модели**: `step/step-3.5-turbo`, `step/step-3.5-flash`
- **Статус**: Китайский провайдер, доступен в РФ

## Настройка в .env файле

Создайте или отредактируйте файл `.env` в корне проекта:

```env

# OpenRouter API (альтернатива)
OPENROUTER_API_KEY=ваш_ключ_openrouter

# Настройки моделей
DEFAULT_MODEL=deepseek/deepseek-chat
```

## Приоритет использования

Приложение будет использовать провайдеров в следующем порядке:

1. **OpenRouter** - для всех моделей

## Проверка ключей

Запустите тестовый скрипт для проверки API ключей:

```bash
python -c "
from utils.config import settings
print('OpenRouter API ключ:', '✅ Установлен' if settings.openrouter_api_key else '❌ Не установлен')
print('Модель по умолчанию:', settings.default_model)
"
```

## Решение проблем

### "unsupported_country_region_territory"
- **Причина**: Региональные ограничения OpenRouter/OpenAI
- **Решение**: Используйте DeepSeek (поддерживает РФ/СНГ)

### "API ключ не настроен"
- **Причина**: Нет ключей в `.env` файле
- **Решение**: Добавьте хотя бы один API ключ из списка выше

### "Model not found"
- **Причина**: Модель не поддерживается выбранным провайдером
- **Решение**: Проверьте правильность названия модели

## Конфигурация моделей

Список доступных моделей можно настроить в `utils/config.py`:

```python
available_models = [
    "deepseek/deepseek-chat",     # Основная модель DeepSeek
    "deepseek/deepseek-coder",    # Для программирования
    "deepseek/deepseek-r1",       # Reasoning модель
]
```