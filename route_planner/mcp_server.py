"""
MCP сервер для планирования маршрутов.
Содержит 3 инструмента: geocode_batch, find_optimal_route, format_route_summary.
"""

import json
import math
import os
from typing import List, Dict, Any, Tuple
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


def solve_tsp_exact(dist_matrix: List[List[float]], n: int) -> Tuple[List[int], float]:
    """
    Точное решение TSP методом динамического программирования (алгоритм Хелд-Карпа).
    Работает для n ≤ 15.
    
    Args:
        dist_matrix: Матрица расстояний
        n: Количество городов
    
    Returns:
        Tuple (оптимальный маршрут, общее расстояние)
    """
    if n <= 1:
        return [0], 0.0
    
    # Алгоритм Хелд-Карпа для TSP
    # Используем битовые маски для представления подмножеств
    
    # Инициализация таблицы DP
    # dp[mask][last] = минимальное расстояние для посещения городов в mask с последним городом last
    dp = {}
    
    # Инициализация: из города 0 в каждый другой город
    for i in range(1, n):
        mask = (1 << i) | 1  # маска с городами 0 и i
        dp[(mask, i)] = dist_matrix[0][i]
    
    # Динамическое программирование
    for subset_size in range(2, n):
        for mask in range(1 << n):
            if bin(mask).count("1") != subset_size or not (mask & 1):
                continue  # пропускаем маски без города 0 или с неправильным размером
            
            for last in range(1, n):
                if not (mask & (1 << last)):
                    continue
                
                # Ищем минимальное расстояние для этого подмножества
                min_dist = float('inf')
                for prev in range(n):
                    if prev == last or not (mask & (1 << prev)):
                        continue
                    
                    prev_mask = mask ^ (1 << last)
                    if (prev_mask, prev) in dp:
                        candidate = dp[(prev_mask, prev)] + dist_matrix[prev][last]
                        if candidate < min_dist:
                            min_dist = candidate
                
                if min_dist != float('inf'):
                    dp[(mask, last)] = min_dist
    
    # Находим оптимальный маршрут
    final_mask = (1 << n) - 1
    min_total = float('inf')
    best_last = -1
    
    for last in range(1, n):
        if (final_mask, last) in dp:
            total = dp[(final_mask, last)] + dist_matrix[last][0]
            if total < min_total:
                min_total = total
                best_last = last
    
    # Восстанавливаем маршрут
    if best_last == -1:
        # Если точный алгоритм не сработал, используем жадный
        return solve_tsp_greedy(dist_matrix, n)
    
    route = []
    mask = final_mask
    current = best_last
    
    while mask != 1:  # пока не остался только город 0
        route.append(current)
        for prev in range(n):
            if prev == current or not (mask & (1 << prev)):
                continue
            
            prev_mask = mask ^ (1 << current)
            if (prev_mask, prev) in dp:
                if abs(dp[(mask, current)] - (dp[(prev_mask, prev)] + dist_matrix[prev][current])) < 1e-6:
                    mask = prev_mask
                    current = prev
                    break
    
    route.append(0)  # добавляем стартовый город
    route.reverse()
    
    return route, min_total


def solve_tsp_greedy(dist_matrix: List[List[float]], n: int) -> Tuple[List[int], float]:
    """
    Жадный алгоритм для TSP.
    
    Args:
        dist_matrix: Матрица расстояний
        n: Количество городов
    
    Returns:
        Tuple (маршрут, общее расстояние)
    """
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
    
    return route_indices, total_distance


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
    
    # Проверяем ограничение количества городов
    warning = ""
    warning_flag = False
    if len(cities) > 5:
        truncated_cities = cities[:5]
        warning = f"⚠️ Я могу обработать только первые 5 городов из {len(cities)}. "
        warning += f"Будет рассчитан маршрут: {', '.join(truncated_cities)}"
        
        # Сохраняем предупреждение для дальнейшего использования
        cities_to_process = truncated_cities
        warning_flag = True
    else:
        cities_to_process = cities
    
    results = []
    errors = []
    
    for city in cities_to_process:
        result = await geocode_city(city)
        if "error" in result:
            errors.append(result)
        else:
            results.append(result)
    
    if errors:
        # Возвращаем первую ошибку
        return json.dumps(errors[0], ensure_ascii=False)
    
    if warning_flag:
        result_data = {"cities": results, "warning": warning}
    else:
        result_data = {"cities": results}
    
    return json.dumps(result_data, ensure_ascii=False)


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
        
        # Выбираем алгоритм в зависимости от количества городов
        if n <= 5:
            # Точный алгоритм для малого количества городов
            route_indices, total_distance = solve_tsp_exact(dist_matrix, n)
        else:
            # Жадный алгоритм для большего количества городов
            route_indices, total_distance = solve_tsp_greedy(dist_matrix, n)
        
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
            "total_distance_km": round(total_distance, 1),
            "algorithm": "exact" if n <= 5 else "greedy"
        }
        
        # Добавляем предупреждение, если оно было в исходных данных
        if "warning" in data:
            result["warning"] = data["warning"]
        
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
        lines = []
        
        # Добавляем предупреждение, если оно есть
        if "warning" in data:
            lines.append(data["warning"])
            lines.append("")
        
        lines.extend(["🗺️ Оптимальный маршрут:", ""])
        
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
        
        # Информация об алгоритме
        algorithm = data.get("algorithm", "unknown")
        if algorithm == "exact":
            lines.append("")
            lines.append("✅ Использован точный алгоритм оптимизации")
        elif algorithm == "greedy":
            lines.append("")
            lines.append("⚠️ Использован жадный алгоритм (для более чем 5 городов)")
        
        return "\n".join(lines)
    
    except json.JSONDecodeError:
        return "❌ Неверный формат данных маршрута"
    except Exception as e:
        return f"❌ Ошибка форматирования: {e}"


if __name__ == "__main__":
    mcp.run()