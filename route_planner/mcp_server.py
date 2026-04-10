"""
MCP сервер для планирования маршрутов.
Содержит 3 инструмента: geocode_batch, find_optimal_route, format_route_summary.
"""

import json
import math
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import httpx
from mcp.server.fastmcp import FastMCP

# Загружаем переменные окружения
load_dotenv()

# Инициализируем MCP сервер
mcp = FastMCP("route_planner")

# API ключ Яндекс.Карт
YANDEX_API_KEY = os.getenv("YANDEX_MAPS_API_KEY")


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Вычисляет расстояние между двумя точками на Земле по формуле гаверсинусов.
    
    Args:
        lat1, lon1: Координаты первой точки (градусы)
        lat2, lon2: Координаты второй точки (градусы)
    
    Returns:
        Расстояние в километрах
    """
    R = 6371.0  # Радиус Земли в км
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat / 2) ** 2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


async def geocode_city(city_name: str) -> Dict[str, Any]:
    """
    Геокодирует один город через Яндекс API.
    
    Args:
        city_name: Название города
    
    Returns:
        Словарь с координатами или ошибкой
    """
    if not YANDEX_API_KEY:
        return {"error": "API ключ Яндекс.Карт не настроен", "city": city_name}
    
    url = "https://geocode-maps.yandex.ru/1.x/"
    params = {
        "apikey": YANDEX_API_KEY,
        "geocode": city_name,
        "format": "json",
        "results": 1
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Извлекаем координаты из ответа
            feature_member = data["response"]["GeoObjectCollection"]["featureMember"]
            if not feature_member:
                return {"error": f"Город '{city_name}' не найден", "city": city_name}
            
            geo_object = feature_member[0]["GeoObject"]
            pos = geo_object["Point"]["pos"]
            lon, lat = map(float, pos.split())
            
            return {"name": city_name, "lat": lat, "lon": lon}
    
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            return {"error": "API ключ Яндекс.Карт не настроен или невалиден", "city": city_name}
        return {"error": f"Ошибка API Яндекс: {e}", "city": city_name}
    except Exception as e:
        return {"error": f"Ошибка геокодирования города '{city_name}': {e}", "city": city_name}


@mcp.tool()
async def geocode_batch(cities: List[str]) -> str:
    """
    Получает координаты для списка городов.
    
    Args:
        cities: Список названий городов
    
    Returns:
        JSON с координатами каждого города
    """
    if len(cities) < 2:
        return json.dumps({"error": "Укажите хотя бы два города"}, ensure_ascii=False)
    
    results = []
    errors = []
    
    for city in cities:
        result = await geocode_city(city)
        if "error" in result:
            errors.append(result)
        else:
            results.append(result)
    
    if errors:
        # Возвращаем первую ошибку
        return json.dumps(errors[0], ensure_ascii=False)
    
    return json.dumps({"cities": results}, ensure_ascii=False)


@mcp.tool()
async def find_optimal_route(coordinates_json: str) -> str:
    """
    Находит оптимальный порядок обхода точек.
    
    Args:
        coordinates_json: JSON с координатами от geocode_batch
    
    Returns:
        JSON с оптимальным маршрутом и общей дистанцией
    """
    try:
        data = json.loads(coordinates_json)
        
        if "error" in data:
            return coordinates_json  # Пробрасываем ошибку
        
        cities_data = data.get("cities", [])
        n = len(cities_data)
        
        if n < 2:
            return json.dumps({"error": "Нужно минимум 2 города для маршрута"}, ensure_ascii=False)
        
        # Строим матрицу расстояний
        dist_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = haversine(
                    cities_data[i]["lat"], cities_data[i]["lon"],
                    cities_data[j]["lat"], cities_data[j]["lon"]
                )
                dist_matrix[i][j] = d
                dist_matrix[j][i] = d
        
        # Жадный алгоритм для TSP
        # Начинаем с первого города
        visited = [False] * n
        route_indices = [0]
        visited[0] = True
        total_distance = 0.0
        
        current = 0
        for _ in range(n - 1):
            nearest = -1
            nearest_dist = float('inf')
            
            for j in range(n):
                if not visited[j] and dist_matrix[current][j] < nearest_dist:
                    nearest = j
                    nearest_dist = dist_matrix[current][j]
            
            visited[nearest] = True
            route_indices.append(nearest)
            total_distance += nearest_dist
            current = nearest
        
        # Формируем результат
        route_names = [cities_data[i]["name"] for i in route_indices]
        segments = []
        for i in range(len(route_indices) - 1):
            from_city = cities_data[route_indices[i]]
            to_city = cities_data[route_indices[i + 1]]
            distance = dist_matrix[route_indices[i]][route_indices[i + 1]]
            segments.append({
                "from": from_city["name"],
                "to": to_city["name"],
                "distance_km": round(distance, 1)
            })
        
        result = {
            "route": route_names,
            "segments": segments,
            "total_distance_km": round(total_distance, 1)
        }
        
        return json.dumps(result, ensure_ascii=False)
    
    except json.JSONDecodeError:
        return json.dumps({"error": "Неверный формат JSON"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Ошибка расчёта маршрута: {e}"}, ensure_ascii=False)


@mcp.tool()
async def format_route_summary(route_json: str) -> str:
    """
    Форматирует результат в читаемый вид.
    
    Args:
        route_json: JSON с маршрутом от find_optimal_route
    
    Returns:
        Красиво отформатированный текст маршрута
    """
    try:
        data = json.loads(route_json)
        
        if "error" in data:
            return f"❌ {data['error']}"
        
        route = data.get("route", [])
        segments = data.get("segments", [])
        total_distance = data.get("total_distance_km", 0)
        
        if not route:
            return "❌ Маршрут не найден"
        
        # Формируем красивый вывод
        lines = ["🗺️ Оптимальный маршрут:", ""]
        
        # Маршрут стрелками
        route_str = " → ".join(route)
        lines.append(route_str)
        lines.append("")
        
        # Детали по сегментам
        if segments:
            lines.append("📏 Расстояния:")
            for seg in segments:
                lines.append(f"• {seg['from']} → {seg['to']}: {seg['distance_km']} км")
            lines.append("")
        
        # Общее расстояние
        lines.append(f"📊 Общее расстояние: {total_distance} км")
        
        return "\n".join(lines)
    
    except json.JSONDecodeError:
        return "❌ Неверный формат данных маршрута"
    except Exception as e:
        return f"❌ Ошибка форматирования: {e}"


if __name__ == "__main__":
    mcp.run()
