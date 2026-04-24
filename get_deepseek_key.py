#!/usr/bin/env python3
"""
Помощник для получения DeepSeek API ключа.
"""

import webbrowser
import sys
import os
from pathlib import Path

def show_instructions():
    """Показать инструкции по получению API ключа."""
    print("=" * 70)
    print("DeepSeek API Ключ - Инструкция")
    print("=" * 70)
    print("\n1. Перейдите на сайт DeepSeek:")
    print("   https://platform.deepseek.com")
    print("\n2. Зарегистрируйтесь или войдите в аккаунт")
    print("\n3. Перейдите в раздел API Keys:")
    print("   https://platform.deepseek.com/api_keys")
    print("\n4. Нажмите 'Create new secret key'")
    print("\n5. Скопируйте созданный ключ")
    print("\n6. Откройте файл .env в редакторе:")
    print(f"   {Path(__file__).parent}/.env")
    print("\n7. Замените 'your_deepseek_api_key_here' на ваш ключ")
    print("\n8. Сохраните файл")
    print("\n9. Перезапустите приложение")
    print("\n" + "=" * 70)
    
    # Открыть сайт в браузере
    response = input("\nОткрыть сайт DeepSeek в браузере? (y/N): ")
    if response.lower() == 'y':
        webbrowser.open("https://platform.deepseek.com/api_keys")
    
    # Открыть файл .env
    response = input("\nОткрыть файл .env для редактирования? (y/N): ")
    if response.lower() == 'y':
        env_path = Path(__file__).parent / ".env"
        
        # Попробовать разные редакторы
        editors = ["nano", "vim", "gedit", "code", "notepad.exe"]
        
        for editor in editors:
            try:
                os.system(f"{editor} {env_path}")
                break
            except:
                continue
        else:
            print(f"Файл .env находится по пути: {env_path}")
            print("Откройте его вручную в любом текстовом редакторе.")

def test_current_config():
    """Проверить текущую конфигурацию."""
    print("\n" + "=" * 70)
    print("Текущая конфигурация")
    print("=" * 70)
    
    try:
        from utils.config import settings
        
        print(f"\nDeepSeek API ключ: {'✅ Установлен' if settings.deepseek_api_key else '❌ Не установлен'}")
        print(f"OpenRouter API ключ: {'✅ Установлен' if settings.openrouter_api_key else '❌ Не установлен'}")
        print(f"OpenAI API ключ: {'✅ Установлен' if settings.openai_api_key else '❌ Не установлен'}")
        print(f"\nМодель по умолчанию: {settings.default_model}")
        
        if not any([settings.deepseek_api_key, settings.openrouter_api_key, settings.openai_api_key]):
            print("\n⚠️  ВНИМАНИЕ: Не настроен ни один API ключ!")
            print("   Приложение не сможет работать без API ключа.")
            print("   Рекомендуем использовать DeepSeek для РФ/СНГ.")
            
    except Exception as e:
        print(f"\n❌ Ошибка загрузки конфигурации: {e}")

def main():
    """Основная функция."""
    print("DeepSeek API Key Setup Assistant")
    print("=" * 70)
    
    test_current_config()
    show_instructions()
    
    print("\n" + "=" * 70)
    print("После настройки API ключа, запустите:")
    print("  python app/main.py")
    print("=" * 70)

if __name__ == "__main__":
    main()