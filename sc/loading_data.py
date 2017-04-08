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

    # Collecting required columns
    if forceAll:
        data = pd.read_excel(path)
    else:
        # TODO add exception for Unknown priority
        required_priorities = list()
        required_priorities.append('ID (автономер в базе)')
        for priority in priorities:
            required_priorities.extend(column_names[priority])

        # Loading data with staff features
        data = pd.read_excel(path)[required_priorities]
    return data


# Function for loading RAW target

def load_targets(path='../data/Target2013.xlsx',
                 priorities=['Высокая'],
                 forceAll=False):
    """Load target-data with required columns names"""
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
                                            == priority]['Поле'])
    # Collecting required column names
    if forceAll:
        data = pd.read_excel(path)
    else:
        # TODO add exception for Unknown priority
        required_priorities = list()
        # TODO add RE for extraction same priority together
        required_priorities.extend(['ID (автономер в базе)', 'Явка на смене (Смена)'])
        for priority in priorities:
            required_priorities.extend(column_names[priority])

        # Loading data with staff working-results
        data = pd.read_excel(path)[required_priorities]
    return data


def load_data():
    """Function for loading data and returning data_features and data_target DataFrame"""
    # Loading from original Excel files:
    # TODO try to fix encoding
    # data_features = load_features(priorities=priorities)
    # data_features.to_csv('../data/tmp/F13.csv', encoding='cp1251')
    # data_target = load_targets()
    # data_target.to_csv('../data/tmp/T13.csv', encoding='cp1251')

    # Loading from steady-files:
    data_features = pd.read_csv('../data/tmp/F13.csv', encoding='cp1251',
                                index_col=0)
    data_target = pd.read_csv('../data/tmp/T13.csv', encoding='cp1251',
                              index_col=0)

    data = data_features.merge(data_target,
                               on='ID (автономер в базе)')
    # print('--------------Features--------', '\n', data_features)
    # print('--------------Target--------', '\n', data_target)
    # print(data)
    X = data[list(data_features)]
    Y = data[list(data_target)]
    return X, Y
