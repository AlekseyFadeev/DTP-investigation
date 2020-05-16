"""
Модуль для быстрого поиска точек на фиксированном расстоянии от точек из базы данных (датафрейма)
В основе лежит структура данных cKDTree из библиотеки scipy
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional, List
from Methodology import DISTANCE_RADIUS
from Utils import to_meters, to_degrees
from tqdm import tqdm

from geojson import Feature, Point, FeatureCollection, dump

class Tree:

    def __init__(self,
                 dataframe: pd.DataFrame,
                 radius=DISTANCE_RADIUS,
                 center_coords: Optional[Tuple[float, float]] = None,
                 k_neighbors_limit: int = 500):

        self.dataframe = dataframe
        self.radius = radius
        self.k_neighbors = k_neighbors_limit  # максимальное количество ДТП в окрестности (нужно для ускорения поиска)

        self.coords = self.dataframe[['Lat', 'Lon']].values.astype('float32')  # массив географических координат
        self.scores = self.dataframe['Total_score'].values

        if center_coords is not None:
            self.center = center_coords
        else:
            self.center = tuple(np.mean(self.coords, axis=0))

        self.coords_m = to_meters(self.coords, self.center)  # массив координат в метрах

    def create_tree(self):
        """
        Создаем cKDTree для реализации быстрых запросов к БД
        """
        self.tree = cKDTree(self.coords_m)

    def restrict_distance(self, query_result):
        """
        Оставляем только те индексы из запроса к дереву, расстояние которых меньше чем distance
        TODO: ускорить бинарным поиском, проверить запросы с нулевым результатом
        """
        d = query_result[0]
        stop_ind = len(d) - 1
        for i in range(len(d)):
            if d[i] >= self.radius:
                stop_ind = i
                break

        return query_result[1][:stop_ind]

    def query_for_point(self,
                        point: Tuple[float, float]) -> Tuple[float, List[int]]:
        """
        Для точки point в формате (широта, долгота) возвращает ей score и список индексов ближайших к ней ДТП
        из self.dataframe
        """

        point_m = to_meters(np.asarray(point), self.center)
        indices = self.restrict_distance(np.asarray(self.tree.query(point_m, k=self.k_neighbors)).astype(int))

        score = np.sum(self.scores[indices])

        return score, indices
    
    def query_for_region(self,
                         coords_min: Tuple[float, float],
                         coords_max: Tuple[float, float],
                         grid_step: int = 100,
                         save_path: Optional[str] = None) -> np.ndarray:
        """
        Подсчет score`ов в области, ограниченной прямоугольником
        :param coords_min: координаты левого нижнего угла области
        :param coords_max: координаты правого верхнего угла области
        :param grid_step: шаг сетки в метрах
        :param save_path: если указано, то путь для сохранения результата в формате geojson
        :return: heatmap - матрица, заполненная значениями score для каждой точки сетки
        """

        c_min = to_meters(np.asarray(coords_min), self.center)
        c_max = to_meters(np.asarray(coords_max), self.center)

        X = np.arange(c_min[0], c_max[0], grid_step).astype(int)
        Y = np.arange(c_min[1], c_max[1], grid_step).astype(int)

        heat_map = np.zeros((len(X), len(Y)))

        if save_path is None:
            for i in tqdm(range(len(X))):
                for j in range(len(Y)):
                    point = [X[i], Y[j]]
                    indices = self.restrict_distance(np.asarray(self.tree.query(point, k=self.k_neighbors)).astype(int))
                    result = np.sum(self.scores[indices])
                    heat_map[i, j] = result
        else:
            json_output = []
            for i in tqdm(range(len(X))):
                for j in range(len(Y)):
                    point = [X[i], Y[j]]
                    indices = self.restrict_distance(np.asarray(self.tree.query(point, k=self.k_neighbors)).astype(int))
                    result = np.sum(self.scores[indices])
                    heat_map[i, j] = result

                    point_c = to_degrees(np.asarray(point), self.center)

                    point_json = Feature(geometry=Point((point_c[1], point_c[0])),
                                         properties={'score': result})

                    json_output.append(point_json)

            json_output = FeatureCollection(json_output)

            with open(save_path, 'w') as file:
                dump(json_output, file)

        return heat_map