# Special functions for loading data
# AIRY v.0.01

import numpy as np
import pandas as pd


# All column's names:
# ['Сотрудник', 'ID (автономер в базе)', 'Фамилия', 'Имя', 'Отчество',
# 'Пол', 'Город', 'Создано (внесена анкета в базу)', 'Субъект федерации (Город)',
# 'Есть имейл (указан сервис)', 'Адрес по прописке', 'Адрес по факт.проживанию',
# 'Адрес прописки совпадает с адресом проживания', 'Алкоголь', 'Аллергены',
# 'Аллергические реакции', 'Ближ.метро', 'Виды хронический заболеваний',
# 'Возможные трудности', 'Возраст', 'Гражданство', 'График основной работы',
# 'Дата выдачи паспорта', 'Дата рождения', 'Дети', 'Должность', 'Домашний телефон',
# 'Дополнительный телефон', 'Допольнительное образование', 'Другие вредные привычки',
# 'Есть основная работа', 'Индекс по прописке', 'Индекс по факт.проживанию',
# 'ИНН', 'Источник', 'Количество и возраст детей', 'Комментарии по предпочтительному графику',
# 'Курение', 'Любимый герой сказки', 'Любимый фильм', 'Первые 4 цифры моб телефона', 'Моторика',
# 'Образование', 'Опыт проведения инвентаризаций', 'Опыт проведения инвентаризаций (описание)',
# 'Предпочтительны смены', 'Продолжительность работы', 'ПСС', 'Работа на высоте', 'Размер одежды',
# 'Район проживания (рег)', 'Самые сильные/слабые личные качества', 'Семейное положение',
# 'Учебное заведение', 'Хобби', 'Хронические заболевания', 'Что привлекает в работе']

def load_features(path='../data/Features2013.xlsx',
                  priorities=['Важный'],
                  forceAll=False):
    """Load data with required columns names"""
    # Setting possible priorities for features

    # TODO Add scanning for unique values of priority
    # Create a structure for our column names with their priorities
    column_names = dict()

    # Loading all possible priorities for this data
    list2 = pd.read_excel(path,
                          sheetname=1)
    list2['Важность (по мнению МЗ)'].fillna('NaN',
                                            inplace=True)
    # Getting unique values of priorities
    possible_priorities = list2['Важность (по мнению МЗ)'].unique()

    # Collecting all column names
    for priority in possible_priorities:
        column_names[priority] = list(list2[list2['Важность (по мнению МЗ)']
                                            == priority]['Название поля'])

    # Collecting required column names
    if forceAll:
        data = pd.read_excel(path, index_col='ID (автономер в базе)')
    else:
        # TODO add exception for Unknown priority
        required_priorities = list()
        for priority in priorities:
            required_priorities.extend(column_names[priority])

        # Loading data with staff features
        data = pd.read_excel(path, index_col='ID (автономер в базе)')[required_priorities]
    return data
