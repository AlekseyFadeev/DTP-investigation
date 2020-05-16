"""
Модуль для чтения и первоначальной обработки данных
"""


import pandas as pd
from datetime import datetime
from Methodology import LETAL_COEFF
from Utils import *


def get_data(columns=[],
             path='/dtp.json'):
    """
    :param columns: поля, которые мы хотим оставить из json файла (если не указано - то оставляем все)
    :param path: путь к json файлу с данными
    :return:
    """
    if not columns:
        data = pd.read_json(path)
    else:
        data = pd.read_json(path)
        data = data[columns]
    return data


def prepare_data(data: pd.DataFrame,
                 date: datetime = datetime.now(),
                 days: int = 365) -> pd.DataFrame:
    """

    :param data: исходный датафрейм данных
    :param date: дата отсчёта
    :param days: количество дней, определяющее период акутальности ДТП
    :return: обработанный датафрейм (КОПИЯ, исходный датафрейм не изменяется!)

    Функция фильтрует данные, оставляя актуальные (период days дней до указанной даты)

    Функция создаёт дополнительные информативные столбцы в датафрейме:
        - Base score: коэффициент тяжести ДТП
        - Lethals: количество летальных исходов в данном ДТП
        - Total_score: тяжесть ДТП с учётом коэффициента летальности
        - If_crossroad: является ли место ДТП перекрестком
        - Lat & Lon: широта и долгота места ДТП
    """

    data['days_since_DTP'] = data['date'].apply(lambda x: (date - x).days)
    prepared_data = data[data['days_since_DTP'] < days]

    prepared_data['Base_score'] = prepared_data['DTP_V'].apply(calculate_base_score)
    prepared_data['Lethals'] = prepared_data['infoDtp'].apply(count_lethal)
    prepared_data['Total_score'] = prepared_data['Lethals'].apply(lambda x: LETAL_COEFF if x > 0 else 1) * prepared_data['Base_score']
    prepared_data['If_crossroad'] = prepared_data['infoDtp'].apply(if_crossroad)

    prepared_data['Lat'] = prepared_data['infoDtp'].apply(lambda x: float(x['COORD_W']))
    prepared_data['Lon'] = prepared_data['infoDtp'].apply(lambda x: float(x['COORD_L']))

    # Сейчас мы просто игнорируем нулевые координаты (ошибки заполнения)
    # По-хорошему, это TODO: через geocoder пофиксить ошибочные координаты
    prepared_data = prepared_data[(prepared_data['Lat'] > 0) | (prepared_data['Lon'] > 0)]

    prepared_data = prepared_data.reset_index(drop=True)

    return prepared_data