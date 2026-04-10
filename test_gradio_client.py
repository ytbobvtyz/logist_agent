#!/usr/bin/env python3
"""
Тестирование работы приложения через Gradio клиент
"""

from gradio_client import Client

def test_app():
    """Тестирует работу приложения через Gradio Client"""
    
    try:
        # Создаем клиент для подключения к приложению
        client = Client("http://localhost:7860")
        print("✅ Клиент успешно подключен к приложению")
        
        # Получаем информацию о приложении
        print(f"API endpoints: {client.endpoints}")
        
        # Используем правильный endpoint для чат-интерфейса
        message = "Найди маршрут между Москвой и Санкт-Петербургом"
        
        # Используем endpoint /chat_response с fn_index 0
        # Согласно коду app.py, функция chat_response принимает message и history
        # История должна быть в формате List[List[Dict[str, str]]]
        empty_history = []  # пустая история для первого сообщения
        
        result = client.predict(
            message,        # сообщение пользователя
            empty_history,  # пустая история (для первого сообщения)
            api_name="/chat_response"
        )
        
        print(f"✅ Результат получен")
        
        # Результат должен содержать обновленную историю, debug log и пустую строку
        if result and len(result) >= 3:
            updated_history, debug_log, empty_string = result
            print(f"Обновленная история: {updated_history}")
            print(f"Debug log: {debug_log}")
            print(f"Пустая строка: {empty_string}")
            
            # Извлекаем ответ ассистента из истории
            if updated_history and len(updated_history) >= 2:
                last_user_msg = updated_history[-2]  # предпоследнее сообщение - пользователь
                last_assistant_msg = updated_history[-1]  # последнее сообщение - ассистент
                
                if last_assistant_msg and len(last_assistant_msg) > 0:
                    assistant_content = last_assistant_msg[0].get('content', 'Нет содержимого')
                    print(f"\n📋 Ответ ассистента:")
                    print(f"{assistant_content}")
                
    except Exception as e:
        print(f"❌ Ошибка при подключении: {e}")

if __name__ == "__main__":
    test_app()