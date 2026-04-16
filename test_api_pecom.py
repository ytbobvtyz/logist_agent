#!/usr/bin/env python3
"""
Простой скрипт для подключения к публичному API ПЭК 
и получения всех доступных городов с сохранением в sities_pec.json

API endpoint: https://pecom.ru/ru/calc/towns.php
Формат ответа:
{
    "Регион1": {"id1": "Город1", "id2": "Город2", ...},
    "Регион2": {"id3": "Город3", ...}
}
"""

import json
import requests
import sys

def get_pecom_cities():
    """Получает города из API ПЭК и сохраняет в JSON файл."""
    url = "https://pecom.ru/ru/calc/towns.php"
    
    print(f"Подключаюсь к API ПЭК: {url}")
    
    response = None
    try:
        # Отправляем запрос к API
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Парсим JSON ответ
        cities_data = response.json()
        print(f"✓ Успешно получены данные")
        
        # Сохраняем в файл
        with open("sities_pec.json", "w", encoding="utf-8") as f:
            json.dump(cities_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Данные сохранены в sities_pec.json")
        
        # Выводим статистику
        print("\nСТАТИСТИКА:")
        print("-" * 40)
        
        total_regions = len(cities_data)
        total_cities = 0
        
        for region, cities in cities_data.items():
            if isinstance(cities, dict):
                total_cities += len(cities)
        
        print(f"Регионов: {total_regions}")
        print(f"Городов: {total_cities}")
        
        # Примеры данных
        print("\nПРИМЕРЫ ГОРОДОВ:")
        print("-" * 40)
        
        example_count = 0
        for region, cities in cities_data.items():
            if isinstance(cities, dict) and cities:
                city_id = list(cities.keys())[0]
                city_name = cities[city_id]
                print(f"Регион: {region}")
                print(f"  Город: {city_name} (ID: {city_id})")
                example_count += 1
                if example_count >= 3:
                    break
        
        print("\n" + "=" * 40)
        print("ВСЕ ГОРОДА УСПЕШНО СОХРАНЕНЫ!")
        print("Файл: sities_pec.json")
        print(f"Размер данных: {total_regions} регионов, {total_cities} городов")
        print("=" * 40)
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Ошибка подключения к API: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Ошибка парсинга JSON ответа: {e}")
        if response and hasattr(response, 'text'):
            print(f"Ответ сервера: {response.text[:200]}...")
        else:
            print("Не удалось получить ответ сервера")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Неожиданная ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 60)
    print("СКРИПТ ДЛЯ ПОЛУЧЕНИЯ ГОРОДОВ ПЭК")
    print("=" * 60)
    get_pecom_cities()