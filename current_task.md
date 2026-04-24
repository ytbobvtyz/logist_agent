1. FIX 1: при нажатии кнопки пользователь не видит, что она нажата и действия начались. необходимо заблокировать кнопку для дальнейшийх нажатий, показать пользователю, что идёт процесс работы над его запросом - DONE
2. FIX 2: необходимо проверить логику рассчёта кратчайшего расстояния. сейчас агент не справляется с корректным рассчётом при маршруте более чем с 3 точками. Нужно сделать так, чтобы он корректно считал оптимальный маршрут при задаче до 5 включительно населенных пунктов. Если пунктов больше 5 - агент должен взять только 5 первых городов в запросе пользователя и честно предупредить, что для большего числа пунктов выгрузки его мозгов не хватает - DONE
3. FIX 3: Алгоритм нахождения оптимальной дистанции работает правильно, но с одним ньюансом - он берет в качестве города отправления первый из списка. Поэтому, если рассчитан маршрут для 3 и более точек и пользователь явно не указал город отправления, то необходимо добавить в промпт вывода информацию, что "в качестве города отправления использован первый город из списка - ***" - DONE
4. FEATURE 1: Реализовать дополнительный MCP сервер для работы с api pecom.ru. Добавить агенту способность получать стоимость перевозки ООО "ПЭК" - DONE

5. FEATURE 2: реализовать индексирование RAG. (TF-IDF-indexer.py) - DONE
6. FEATURE 3: реализовать тестовый скрипт для анализа эффективности работы агента с rag и без rag - DONE
7. FEATURE 4 - снабдить приложение и агента доступом к RAG инструменту - он должен использовать RAG-функционал, добавленный в проект в дополнение к уже реализованному MCP-инструментом. Агент должен понимать, как и в каких случаях он может и должен использовать RAG, как и в каких случаях он должжен и может использовать MCP, не должно возникать конфликтов. - DONE ✅

8. FEATURE 5 - реализовать корректную логику взаимодействия агента route_planner/agent.py и сервера route_planner/pecom_server.py - агент должен провалидировать названия населённых пунктов и найти коды в sities_pec.json и передать в pecom_server ровно два кода населённых пункта +, если пользователь указал, весогабаритную информацию по грузу. Если пользователь не указал - по умолчанию берём 100 кг, 0.125 м3, габариты 0.5 * 0.5 * 0.5 м (ширина, длина, высота). 

9. FEATURE 6 - реализовать мультидиалоговое меню, историю и суммаризацию диалога. - done

10. FEATURE 7 - реализовать возможность использования локальной модели ollama для работы с rag, mcp инструменты при выборе локальной модели будут не доступны.
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"

ТЗ: Интеграция локальной модели Ollama
Цель
Добавить поддержку локальной модели llama3.2:3b через Ollama без изменения существующей логики RAG/MCP.

Архитектурное решение
Создать абстрактный слой LLMClient, который может работать с:

OpenRouter (существующая логика)

Ollama (новая локальная модель)

Необходимые изменения
1. Новый файл llm_client.py
python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json

class LLMClient(ABC):
    @abstractmethod
    async def chat_completion(self, messages: List[Dict], tools: Optional[List] = None, max_tokens: int = 1024):
        pass

class OpenRouterClient(LLMClient):
    # Перенести существующую логику из enhanced_agent.py
    
class OllamaClient(LLMClient):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.base_url = base_url
        self.model = model
        self.client = AsyncOpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama",  # Ollama не требует ключ
            timeout=120.0
        )
    
    async def chat_completion(self, messages: List[Dict], tools: Optional[List] = None, max_tokens: int = 1024):
        # Ollama не поддерживает tools, поэтому игнорируем их
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response
2. Изменения в enhanced_agent.py
python
class EnhancedRoutePlannerAgent:
    def __init__(self, model: str = "openrouter/auto", use_local: bool = False, db_path: str = "conversations.db"):
        self.use_local = use_local
        self.model_name = model
        
        if use_local:
            from llm_client import OllamaClient
            self.client = OllamaClient(model=model)  # model = "llama3.2:3b"
        else:
            self.client = AsyncOpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                timeout=120.0
            )
            self.model = model
        
        # Остальная инициализация без изменений
3. Изменения в enhanced_app.py
python
# Добавить в интерфейс:
with gr.Row():
    use_local_checkbox = gr.Checkbox(label="Использовать локальную модель Ollama (llama3.2:3b)", value=False)
    model_dropdown = gr.Dropdown(..., interactive=True, visible=True)

# Функция инициализации:
def init_agent(model: str, use_local: bool):
    async def _init():
        agent = EnhancedRoutePlannerAgent(
            model=model if not use_local else "llama3.2:3b",
            use_local=use_local
        )
        success = await agent.connect_mcp()
        return agent, success
Важные ограничения
Локальная модель НЕ поддерживает function calling (tools)

При использовании Ollama, MCP инструменты будут недоступны

Нужно явно сообщать пользователю об этом

Решение: Модифицировать _process_with_mcp для локальной модели:

python
async def _process_with_mcp(self, user_message, context, conversation_id, callback):
    if self.use_local:
        if callback:
            await callback("⚠️ Локальная модель не поддерживает инструменты MCP. Использую режим RAG/знаний.")
        # Использовать RAG или базовые знания
        return await self._process_without_tools(user_message, context, conversation_id, callback)
    
    # Существующая логика с MCP
Что НЕ нужно менять
✅ Всю логику RAG (search_with_rag, _should_use_rag)

✅ ConversationManager, Summarizer, TaskStateManager

✅ MCP серверы и оркестратор

✅ Форматирование ответов (эмодзи 🔧📚💡)

✅ Базу данных диалогов

Тестовый сценарий
python
# Тест 1: Только RAG с локальной моделью
agent = EnhancedRoutePlannerAgent(use_local=True)
response = await agent.process_message("Какие тарифы у ПЭК?")

# Ожидаемый результат:
# 💡 (локальная модель, RAG найден)
# [ответ из документов]

# Тест 2: Запрос к MCP с локальной моделью
response = await agent.process_message("Рассчитай маршрут Москва-Питер")

# Ожидаемый результат:
# ⚠️ Локальная модель не поддерживает MCP
# 💡 (ответ из знаний, без расчетов)
Что должно быть в .env (опционально)
bash
# Для локальной модели не нужен API ключ
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

Проверь что: 
в связанных модулях всегда будет испльзована та модель, которую выбрал пользователь