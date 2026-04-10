#!/usr/bin/env python3
"""
Тестирование работы приложения через Gradio клиент
"""

import requests
import json
import time

def test_app():
    """Тестирует работу приложения"""
    
    # Проверяем, что приложение запущено
    try:
        response = requests.get("http://localhost:7860")
        print(f"✅ Приложение доступно на порту 7860 (статус: {response.status_code})")
    except Exception as e:
        print(f"❌ Приложение недоступно: {e}")
        return
    
    # Пробуем отправить запрос через Gradio API
    # Сначала получаем session hash
    try:
        session_response = requests.post("http://localhost:7860/api/queue/join", 
                                        json={"fn_index": 0})
        print(f"Session response: {session_response.status_code}")
        if session_response.status_code == 200:
            print(f"Session data: {session_response.text}")
    except Exception as e:
        print(f"❌ Ошибка при получении сессии: {e}")
    
    # Пробуем прямой запрос к функции
    try:
        # Создаем тестовый запрос
        test_data = {
            "data": ["Найди маршрут между Москвой и Санкт-Петербургом", []],
            "fn_index": 0,
            "session_hash": "test123"
        }
        
        response = requests.post("http://localhost:7860/api/predict", 
                               json=test_data)
        print(f"Predict response: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Response text: {response.text}")
            
    except Exception as e:
        print(f"❌ Ошибка при отправке запроса: {e}")

if __name__ == "__main__":
    test_app()