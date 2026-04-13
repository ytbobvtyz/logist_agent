#!/usr/bin/env python3
"""
Интерактивный тестер API ПЭК
Запускает запросы к API ПЭК для проверки работы сервиса
"""

import asyncio
import json
import httpx
from typing import Optional

# База ID городов ПЭК (встроенный словарь)
PECOM_CITY_IDS = {
    "москва": 446,
    "санкт-петербург": -203,
    "спб": -203,
    "казань": 64883,
    "нижний новгород": -195,
    "новосибирск": 49,
    "екатеринбург": 54,
    "ростов-на-дону": 39,
    "ростов": 39,
    "самара": 57,
    "омск": 52,
    "челябинск": 65,
    "волгоград": 42,
    "пермь": 55,
    "уфа": 68,
    "краснодар": 47,
    "воронеж": 41,
    "тюмень": 62,
    "тольятти": 60,
    "ижевск": 45,
    "барнаул": 33,
    "ульяновск": 63,
    "иркутск": 46,
    "хабаровск": 64,
    "владивосток": 40,
    "ярославль": 69,
    "кемерово": 348,
    "новокузнецк": 51,
    "рязань": 56,
    "липецк": 344,
    "пенза": 332,
    "киров": 48,
    "чебоксары": 66,
    "калининград": 334,
    "сочи": 594,
}


def get_city_id(city_name: str) -> Optional[int]:
    """Получает ID города из встроенного словаря."""
    normalized = city_name.lower().strip()
    return PECOM_CITY_IDS.get(normalized)


async def find_city_by_api(city_name: str) -> Optional[int]:
    """Пытается найти ID города через API ПЭК."""
    try:
        url = "http://calc.pecom.ru/bitrix/components/pecom/calc/ajax.php"
        params = {
            "action": "getCity",
            "term": city_name
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get("id")
        return None
    except Exception:
        return None


async def test_pecom_api(from_city: str, to_city: str) -> dict:
    """Тестирует API ПЭК с заданными городами."""
    
    print(f"\n🧪 Тестируем API ПЭК")
    print(f"Отправка: {from_city}")
    print(f"Доставка: {to_city}")
    print("-" * 50)
    
    # Получаем ID городов
    from_city_id = get_city_id(from_city)
    to_city_id = get_city_id(to_city)
    
    # Если город не найден в словаре, пробуем найти через API
    if from_city_id is None:
        print(f"🔍 Ищем ID города '{from_city}' через API...")
        from_city_id = await find_city_by_api(from_city)
    
    if to_city_id is None:
        print(f"🔍 Ищем ID города '{to_city}' через API...")
        to_city_id = await find_city_by_api(to_city)
    
    # Проверяем результаты поиска
    if from_city_id is None:
        return {
            "success": False,
            "error": f"Город '{from_city}' не найден",
            "available_cities": list(PECOM_CITY_IDS.keys())
        }
    
    if to_city_id is None:
        return {
            "success": False,
            "error": f"Город '{to_city}' не найден",
            "available_cities": list(PECOM_CITY_IDS.keys())
        }
    
    print(f"✅ Найдены ID городов:")
    print(f"   {from_city} → {from_city_id}")
    print(f"   {to_city} → {to_city_id}")
    
    # Параметры запроса
    weight_kg = 50.0
    length_m = 0.5
    width_m = 0.5
    height_m = 0.4
    volume = length_m * width_m * height_m
    
    # Формируем URL запроса к API ПЭК
    base_url = "http://calc.pecom.ru/bitrix/components/pecom/calc/ajax.php"
    
    # Пробуем разные форматы параметров
    params_variants = [
        # Вариант 1: исходный формат
        {
            'take[town]': from_city_id,
            'deliver[town]': to_city_id,
            'places[0][]': [
                length_m,
                width_m,
                height_m,
                volume,
                weight_kg,
                0,  # негабарит
                0,  # упаковка
            ]
        },
        # Вариант 2: с action параметром
        {
            'action': 'calc',
            'take[town]': from_city_id,
            'deliver[town]': to_city_id,
            'places[0][]': [
                length_m,
                width_m,
                height_m,
                volume,
                weight_kg,
                0,  # негабарит
                0,  # упаковка
            ]
        },
        # Вариант 3: альтернативный формат
        {
            'take[town]': from_city_id,
            'deliver[town]': to_city_id,
            'weight': weight_kg,
            'volume': volume,
            'length': length_m,
            'width': width_m,
            'height': height_m
        }
    ]
    
    for i, params in enumerate(params_variants, 1):
        print(f"\n📡 Попытка {i}: Отправляем запрос к API ПЭК...")
        print(f"URL: {base_url}")
        print(f"Параметры: {params}")
        
        try:
            async with httpx.AsyncClient() as client:
                # Добавляем заголовки для имитации браузера
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = await client.get(base_url, params=params, headers=headers, timeout=15.0)
                
                print(f"\n📥 Ответ от API:")
                print(f"Статус: {response.status_code}")
                print(f"URL запроса: {response.url}")
                print(f"Заголовки: {dict(response.headers)}")
                print(f"\nТело ответа:")
                print(response.text)
                
                response.raise_for_status()
                
                # Проверяем content-type
                content_type = response.headers.get('content-type', '')
                if 'application/json' not in content_type:
                    print(f"⚠️ API вернул не JSON ответ (content-type: {content_type})")
                    continue
                
                data = response.json()
                
                # Парсим ответ
                if "error" in data:
                    error_message = data.get("error", ["API вернул ошибку"])[0]
                    print(f"⚠️ API вернул ошибку: {error_message}")
                    continue
                
                if data.get("status") != "success":
                    error_message = data.get("message", "API вернул неуспешный статус")
                    print(f"⚠️ API вернул неуспешный статус: {error_message}")
                    continue
                
                # Ищем тариф "Авто"
                methods = data.get("methods", [])
                auto_price = None
                
                for method in methods:
                    if method.get("name") == "Авто":
                        auto_price = method.get("price")
                        break
                        
                if auto_price is None:
                    print("⚠️ В ответе API не найден тариф 'Авто'")
                    continue
                
                # Успех!
                return {
                    "success": True,
                    "from_city": from_city,
                    "to_city": to_city,
                    "weight_kg": weight_kg,
                    "cost": int(auto_price),
                    "message": f"Стоимость доставки {from_city} → {to_city}: {int(auto_price)} ₽",
                    "api_response": data
                }
                
        except httpx.HTTPStatusError as e:
            print(f"⚠️ HTTP ошибка: {e.response.status_code}")
            continue
        except httpx.RequestError as e:
            return {
                "success": False,
                "error": f"Ошибка сети при запросе к API: {e}",
                "details": str(e),
                "error_type": type(e).__name__
            }
        except json.JSONDecodeError as e:
            print(f"⚠️ Ошибка декодирования JSON: {e}")
            continue
        except Exception as e:
            print(f"⚠️ Неизвестная ошибка: {e}")
            continue
    
    # Если все попытки не удались
    return {
        "success": False,
        "error": "Все попытки запроса к API завершились неудачей",
        "available_cities": list(PECOM_CITY_IDS.keys())
    }


async def main():
    """Основная функция тестера."""
    print("🚚 Тестер API ПЭК")
    print("=" * 50)
    
    # Тестируем с предопределенными городами
    test_cases = [
        ("москва", "санкт-петербург"),
        ("москва", "казань"),
        ("спб", "москва")
    ]
    
    for from_city, to_city in test_cases:
        print(f"\n🧪 Тестируем: {from_city} → {to_city}")
        print("=" * 50)
        
        # Выполняем тест
        result = await test_pecom_api(from_city, to_city)
        
        print("\n" + "=" * 50)
        print("📊 РЕЗУЛЬТАТ ТЕСТА:")
        print("=" * 50)
        
        if result["success"]:
            print("✅ API работает корректно!")
            print(f"📦 {result['message']}")
            print(f"💰 Стоимость: {result['cost']} ₽")
        else:
            print("❌ API вернул ошибку")
            print(f"Ошибка: {result['error']}")
            if "details" in result:
                print(f"Детали: {result['details']}")
        
        print("\n" + "=" * 50)
    
    print("\n👋 Завершение работы тестера")


if __name__ == "__main__":
    asyncio.run(main())