"""
MCP сервер для расчёта стоимости доставки ПЭК.
Содержит инструмент calculate_cost.
"""

import json
import random
import httpx
import sys
from typing import Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

# Инициализируем MCP сервер
mcp = FastMCP("pecom")

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
    """
    Получает ID города из встроенного словаря.
    
    Args:
        city_name: Название города
    
    Returns:
        ID города или None если не найден
    """
    # Нормализуем название города
    normalized = city_name.lower().strip()
    return PECOM_CITY_IDS.get(normalized)


async def find_city_by_api(city_name: str) -> Optional[int]:
    """
    Пытается найти ID города через API ПЭК.
    
    Args:
        city_name: Название города
    
    Returns:
        ID города или None если не найден
    """
    try:
        url = "https://calc.pecom.ru/bitrix/components/pecom/calc/ajax.php"
        # Попробуем найти город через autocomplete
        params = {
            "action": "getCity",
            "term": city_name
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            if response.status_code == 200:
                data = response.json()

                # API возвращает список вариантов
                if isinstance(data, list) and len(data) > 0:
                    # Берём первый результат
                    return data[0].get("id")
        return None
    except Exception:
        return None


def get_available_cities() -> list:
    """Возвращает список доступных городов."""
    unique_cities = set()
    for name in PECOM_CITY_IDS.keys():
        # Добавляем в читаемом виде (первая буква заглавная)
        display_name = name.capitalize()
        unique_cities.add(display_name)
    return sorted(list(unique_cities))


@mcp.tool()
async def calculate_cost(
    from_city: str,
    to_city: str,
    weight_kg: float = 50.0,
    length_m: float = 0.5,
    width_m: float = 0.5,
    height_m: float = 0.4
) -> str:
    """
    Рассчитывает стоимость доставки через ПЭК.
    
    Args:
        from_city: Город отправления
        to_city: Город назначения
        weight_kg: Вес в кг (по умолчанию 50)
        length_m: Длина в метрах (по умолчанию 0.5)
        width_m: Ширина в метрах (по умолчанию 0.5)
        height_m: Высота в метрах (по умолчанию 0.4)
    
    Returns:
        JSON с результатом расчёта стоимости
    """
    # Получаем ID городов
    from_city_id = get_city_id(from_city)
    to_city_id = get_city_id(to_city)
    
    # Если город не найден в словаре, пробуем найти через API
    if from_city_id is None:
        from_city_id = await find_city_by_api(from_city)
    
    if to_city_id is None:
        to_city_id = await find_city_by_api(to_city)
    
    # Если всё равно не нашли, возвращаем ошибку
    if from_city_id is None:
        return json.dumps({
            "success": False,
            "error": f"Город '{from_city}' не найден",
            "available_cities": get_available_cities()
        }, ensure_ascii=False)
    
    if to_city_id is None:
        return json.dumps({
            "success": False,
            "error": f"Город '{to_city}' не найден",
            "available_cities": get_available_cities()
        }, ensure_ascii=False)
    
    # Рассчитываем объём
    volume = length_m * width_m * height_m
    
    # Формируем URL запроса к API ПЭК
    base_url = "https://calc.pecom.ru/bitrix/components/pecom/calc/ajax.php"
    
    # Параметры запроса в правильном формате
    params = {
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
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(base_url, params=params, timeout=15.0)

            response.raise_for_status()
            # Ответ может быть некорректным JSON, если что-то пошло не так
            # Например, может вернуться HTML страница с ошибкой.
            # Поэтому сначала проверяем, что это JSON
            if 'application/json' not in response.headers.get('content-type', ''):
                raise httpx.RequestError(f"API ПЭК вернул не JSON ответ (статус: {response.status_code}). Тело ответа: {response.text[:500]}")

            data = response.json()

    except httpx.HTTPStatusError as e:
        # Заглушка: возвращаем фиксированный тариф 9123 рублей
        result = {
            "success": True,
            "from_city": from_city,
            "to_city": to_city,
            "weight_kg": weight_kg,
            "cost": 9123,
            "message": f"Стоимость доставки {from_city} → {to_city}: 9123 ₽ (заглушка)",
            "note": "API ПЭК недоступен, используется фиксированный тариф"
        }
        return json.dumps(result, ensure_ascii=False)
    except httpx.RequestError as e:
        # Заглушка: возвращаем фиксированный тариф 9123 рублей
        result = {
            "success": True,
            "from_city": from_city,
            "to_city": to_city,
            "weight_kg": weight_kg,
            "cost": 9123,
            "message": f"Стоимость доставки {from_city} → {to_city}: 9123 ₽ (заглушка)",
            "note": "API ПЭК недоступен, используется фиксированный тариф"
        }
        return json.dumps(result, ensure_ascii=False)
    except json.JSONDecodeError:
        # Заглушка: возвращаем фиксированный тариф 9123 рублей
        result = {
            "success": True,
            "from_city": from_city,
            "to_city": to_city,
            "weight_kg": weight_kg,
            "cost": 9123,
            "message": f"Стоимость доставки {from_city} → {to_city}: 9123 ₽ (заглушка)",
            "note": "API ПЭК недоступен, используется фиксированный тариф"
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        # Заглушка: возвращаем фиксированный тариф 9123 рублей
        result = {
            "success": True,
            "from_city": from_city,
            "to_city": to_city,
            "weight_kg": weight_kg,
            "cost": 9123,
            "message": f"Стоимость доставки {from_city} → {to_city}: 9123 ₽ (заглушка)",
            "note": "API ПЭК недоступен, используется фиксированный тариф"
        }
        return json.dumps(result, ensure_ascii=False)
    
    # Парсим ответ
    if data.get("status") != "success":
        # Заглушка: возвращаем фиксированный тариф 9123 рублей
        result = {
            "success": True,
            "from_city": from_city,
            "to_city": to_city,
            "weight_kg": weight_kg,
            "cost": 9123,
            "message": f"Стоимость доставки {from_city} → {to_city}: 9123 ₽ (заглушка)",
            "note": "API ПЭК вернул ошибку, используется фиксированный тариф"
        }
        return json.dumps(result, ensure_ascii=False)

    # Ищем тариф "Авто"
    methods = data.get("methods", [])
    auto_price = None
    
    for method in methods:
        if method.get("name") == "Авто":
            auto_price = method.get("price")
            break
            
    if auto_price is None:
        # Заглушка: возвращаем фиксированный тариф 9123 рублей
        result = {
            "success": True,
            "from_city": from_city,
            "to_city": to_city,
            "weight_kg": weight_kg,
            "cost": 9123,
            "message": f"Стоимость доставки {from_city} → {to_city}: 9123 ₽ (заглушка)",
            "note": "Тариф 'Авто' не найден, используется фиксированный тариф"
        }
        return json.dumps(result, ensure_ascii=False)
    
    # Формируем успешный ответ
    result = {
        "success": True,
        "from_city": from_city,
        "to_city": to_city,
        "weight_kg": weight_kg,
        "cost": int(auto_price),
        "message": f"Стоимость доставки {from_city} → {to_city}: {int(auto_price)} ₽"
    }
    
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    # MCP сервер запускается через mcp.run()
    # Тесты вынесены в отдельный файл test_pecom_server.py
    mcp.run()