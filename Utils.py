"""
Модуль со вспомогательными функциями
"""
import numpy as np
from typing import Dict, List, Tuple
from Methodology import DEFAULT_VALUE, COST_FUNCTION
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely import affinity

TO_RADIANS = np.pi / 180.  # degrees to radians

AVG_EARTH_RADIUS = 6371138  # Earth radius, meters

EARTH_METRICS_DIAGONAL_SR = AVG_EARTH_RADIUS * TO_RADIANS  # SQRT(Earth sphere metrics diagonal)

def calculate_base_score(x: str,
                         cost_dict: Dict[str, float] = COST_FUNCTION,
                         default_value=DEFAULT_VALUE) -> float:
    """
    x - значения поля DTP_V для одной строчки
    Функция возвращает количество баллов, соответствующее данному ДТП, согласно методике 2
    Коэффициент летальности учитывается после применения данной функции
    """

    score = cost_dict.get(x, default_value)

    return score


def count_lethal(x: Dict) -> int:
    """
    x - значения поля infoDtp для одной строчки
    Функция возвращает количество летальных исходов в данном ДТП
    """

    lethals = 0
    results = [[K_UCH['S_T'] for K_UCH in ts['ts_uch']] for ts in x['ts_info']]
    for r in results:
        for r_s in r:
            if r_s.startswith('Скончался'):
                lethals += 1

    return lethals


def if_crossroad(x: Dict) -> bool:
    """

    :param x: значения поля infoDtp для одной строчки
    :return: произошло ли данное ДТП на перекрестке
    """
    words = ['Перекресток', 'Перекрёсток', 'перекресток', 'перекрёсток', 'круг']

    results = x['sdor']
    for r in results:
        for w in words:
            if w in r:
                return True

    return False


def to_meters(coords, center_coords):
    """
    Преобразование списка координат в метрическую систему
    """

    x_center, y_center = center_coords
    cos_theta = np.cos(y_center * np.pi / 180.)

    xfact = cos_theta * EARTH_METRICS_DIAGONAL_SR
    yfact = EARTH_METRICS_DIAGONAL_SR

    return coords * np.array([xfact, yfact]), cos_theta


def to_degrees(coords, center_coords):
    """
    Обратное преобразование (метрические координаты) -> (широта, долгота)
    """

    x_center, y_center = center_coords
    cos_theta = np.cos(x_center * np.pi / 180.)

    xfact = cos_theta * EARTH_METRICS_DIAGONAL_SR
    yfact = EARTH_METRICS_DIAGONAL_SR

    return coords / np.array([xfact, yfact])

def distance(point1, point2):
    return np.sqrt(np.sum((point2 - point1) ** 2))

def to_meters_shapely(geom: BaseGeometry, cos: float or None = None) -> Tuple[BaseGeometry, float] or BaseGeometry:
    """
    Преобразование координат к метрам.
    """

    cos_ = cos or np.cos(geom.centroid.y * np.pi / 180)
    geom = affinity.scale(geom,
                          xfact=cos_ * np.pi * AVG_EARTH_RADIUS / 180,
                          yfact=np.pi * AVG_EARTH_RADIUS / 180,
                          origin=Point(0, 0))

    if cos is None:
        return geom, cos_
    else:
        return geom


def to_degrees_shapely(geom: BaseGeometry, cos: float, reversed: bool = False) -> BaseGeometry:

    """
    Преобразование координат в градусы (4326)

    """
    if not reversed:
        geom = affinity.scale(geom,
                              xfact=180 / (cos * np.pi * AVG_EARTH_RADIUS),
                              yfact=180 / (np.pi * AVG_EARTH_RADIUS),
                              origin=Point(0, 0))
    else:
        geom = affinity.scale(geom,
                              xfact=180 / (np.pi * AVG_EARTH_RADIUS),
                              yfact=180 / (cos * np.pi * AVG_EARTH_RADIUS),
                              origin=Point(0, 0))

    return geom