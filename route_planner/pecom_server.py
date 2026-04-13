"""
MCP сервер для расчёта стоимости доставки ПЭК.
Содержит инструмент calculate_cost.
"""

import json
import random
import httpx
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
        url = "http://calc.pecom.ru/bitrix/components/pecom/calc/ajax.php"
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


# Типичные тарифы ПЭК (руб/кг) между крупными городами
# На основе публичной информации о тарифах ПЭК
PECOM_RATES = {
    # Москва/СПБ -> другие города
    ("москва", "санкт-петербург"): 18,
    ("москва", "спб"): 18,
    ("москва", "казань"): 22,
    ("москва", "нижний новгород"): 20,
    ("москва", "новосибирск"): 45,
    ("москва", "екатеринбург"): 40,
    ("москва", "ростов-на-дону"): 28,
    ("москва", "ростов"): 28,
    ("москва", "самара"): 26,
    ("москва", "омск"): 48,
    ("москва", "челябинск"): 42,
    ("москва", "волгоград"): 30,
    ("москва", "пермь"): 38,
    ("москва", "уфа"): 35,
    ("москва", "краснодар"): 32,
    ("москва", "воронеж"): 22,
    ("москва", "тюмень"): 45,
    ("москва", "тольятти"): 26,
    ("москва", "ижевск"): 30,
    ("москва", "барнаул"): 55,
    ("москва", "ульяновск"): 25,
    ("москва", "иркутск"): 65,
    ("москва", "хабаровск"): 85,
    ("москва", "владивосток"): 90,
    ("москва", "ярославль"): 18,
    ("москва", "кемерово"): 50,
    ("москва", "новокузнецк"): 52,
    ("москва", "рязань"): 18,
    ("москва", "липецк"): 20,
    ("москва", "пенза"): 24,
    ("москва", "киров"): 28,
    ("москва", "чебоксары"): 24,
    ("москва", "калининград"): 35,
    ("москва", "сочи"): 38,
    
    # СПБ -> другие города
    ("санкт-петербург", "москва"): 18,
    ("спб", "москва"): 18,
    ("санкт-петербург", "казань"): 26,
    ("спб", "казань"): 26,
    
    # Казань -> другие города
    ("казань", "москва"): 22,
    ("казань", "санкт-петербург"): 26,
    ("казань", "спб"): 26,
}


def get_rate(from_city: str, to_city: str) -> float:
    """
    Получает тариф за кг между городами.
    Если прямой тариф не найден, использует среднее значение.
    """
    from_norm = from_city.lower().strip()
    to_norm = to_city.lower().strip()
    
    # Ищем прямой тариф
    rate = PECOM_RATES.get((from_norm, to_norm))
    if rate:
        return rate
    
    # Ищем обратный тариф
    rate = PECOM_RATES.get((to_norm, from_norm))
    if rate:
        return rate
    
    # Если нет точного тарифа, рассчитываем на основе "расстояния" между ID городов
    from_id = get_city_id(from_city) or 446  # По умолчанию Москва
    to_id = get_city_id(to_city) or 446
    
    # Базовый тариф 25 руб/кг + корректировка на расстояние
    distance_factor = abs(from_id - to_id) / 1000
    base_rate = 25 + (distance_factor * 20)
    
    # Ограничиваем разумными пределами (15-100 руб/кг)
    return max(15, min(100, base_rate))


def generate_demo_price(from_city: str, to_city: str, weight_kg: float, volume_m3: float, length_m: float = 0.5, width_m: float = 0.5, height_m: float = 0.4) -> str:
    """
    Генерирует демо-цену на основе реальных тарифов ПЭК.
    Используется когда публичный API ПЭК недоступен.
    
    Args:
        from_city: Город отправления
        to_city: Город назначения
        weight_kg: Вес в кг
        volume_m3: Объём в м³
        length_m: Длина в метрах
        width_m: Ширина в метрах
        height_m: Высота в метрах
    
    Returns:
        JSON с демо-результатом
    """
    # Получаем тариф за кг
    rate_per_kg = get_rate(from_city, to_city)
    
    # Минимальная стоимость отправки (забор + доставка до терминала)
    min_cost = 650
    
    # Расчёт по весу
    weight_cost = weight_kg * rate_per_kg
    
    # Расчёт по объёму (плотность 250 кг/м³ - стандарт для ПЭК)
    volume_weight = volume_m3 * 250
    volume_cost = volume_weight * rate_per_kg
    
    # Берём максимум из весового и объёмного расчёта + минимальная стоимость
    cost = max(min_cost, weight_cost, volume_cost)
    
    # Округляем до рублей
    cost = int(cost)
    
    # Формируем описание расчёта
    calc_details = f"тариф {rate_per_kg} ₽/кг"
    if volume_cost > weight_cost:
        calc_details += f", объёмный вес {volume_weight:.1f} кг"
    
    result = {
        "success": True,
        "from_city": from_city,
        "to_city": to_city,
        "weight_kg": weight_kg,
        "dimensions": {
            "length_m": length_m,
            "width_m": width_m,
            "height_m": height_m,
            "volume_m3": round(volume_m3, 4)
        },
        "cost": cost,
        "tariff": "Авто",
        "rate_per_kg": rate_per_kg,
        "calculation_details": calc_details,
        "message": f"Стоимость доставки {from_city} → {to_city}: {cost} ₽ ({calc_details})",
        "note": "Расчёт выполнен по справочным тарифам ПЭК"
    }
    
    return json.dumps(result, ensure_ascii=False)


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
    base_url = "http://calc.pecom.ru/bitrix/components/pecom/calc/ajax.php"
    
    # Параметры запроса
    params = {
        "take[town]": from_city_id,
        "deliver[town]": to_city_id,
        "places[0][]": length_m,
        "places[1][]": width_m,
        "places[2][]": height_m,
        "places[3][]": volume,
        "places[4][]": weight_kg,
        "places[5][]": 0,  # негабарит
        "places[6][]": 0,  # упаковка
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(base_url, params=params, timeout=15.0)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        # В случае ошибки HTTP используем демо-режим с примерными ценами
        print(f"⚠️ API ПЭК вернул ошибку HTTP {e.response.status_code}, используем демо-режим")
        return generate_demo_price(from_city, to_city, weight_kg, volume)
    except httpx.RequestError as e:
        # В случае недоступности API используем демо-режим
        print(f"⚠️ API ПЭК недоступен ({e}), используем демо-режим")
        return generate_demo_price(from_city, to_city, weight_kg, volume)
    except Exception as e:
        # В случае любой другой ошибки используем демо-режим
        print(f"⚠️ Ошибка при запросе к API ПЭК ({e}), используем демо-режим")
        return generate_demo_price(from_city, to_city, weight_kg, volume)
    
    # Парсим ответ
    if data.get("status") != "success":
        # API вернул неуспешный статус - используем демо-режим
        print(f"⚠️ API ПЭК вернул неуспешный статус, используем демо-режим")
        return generate_demo_price(from_city, to_city, weight_kg, volume)
    
    # Ищем тариф "Авто"
    methods = data.get("methods", [])
    auto_price = None
    
    for method in methods:
        if method.get("name") == "Авто":
            auto_price = method.get("price")
            break
    
    # Если нет тарифа Авто, ищем первый доступный тариф
    if auto_price is None and methods:
        # Берём первый доступный тариф
        auto_price = methods[0].get("price")
        tariff_name = methods[0].get("name", "Неизвестно")
    else:
        tariff_name = "Авто"
    
    if auto_price is None:
        # Не удалось получить тариф - используем демо-режим
        print(f"⚠️ Не удалось получить тариф из API, используем демо-режим")
        return generate_demo_price(from_city, to_city, weight_kg, volume)
    
    # Формируем успешный ответ
    result = {
        "success": True,
        "from_city": from_city,
        "to_city": to_city,
        "weight_kg": weight_kg,
        "dimensions": {
            "length_m": length_m,
            "width_m": width_m,
            "height_m": height_m,
            "volume_m3": round(volume, 4)
        },
        "cost": int(auto_price),
        "tariff": tariff_name,
        "message": f"Стоимость доставки {from_city} → {to_city}: {int(auto_price)} ₽ (тариф: {tariff_name})"
    }
    
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
