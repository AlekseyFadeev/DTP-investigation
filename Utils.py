"""
Модуль со вспомогательными функциями
"""
import numpy as np
from typing import Dict
from Methodology import DEFAULT_VALUE, COST_FUNCTION

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
    cos_theta = np.cos(x_center * np.pi / 180.)

    xfact = cos_theta * EARTH_METRICS_DIAGONAL_SR
    yfact = EARTH_METRICS_DIAGONAL_SR

    return coords * np.array([xfact, yfact])


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