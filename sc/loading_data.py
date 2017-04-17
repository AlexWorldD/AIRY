# Special functions for loading data
# AIRY v.0.01

import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_pandas, trange
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


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

# ----------------------------------------- Loading data -----------------------------------------
def load_features(path='../data/Features2013.xlsx',
                  priorities=['Важный'],
                  forceAll=False):
    """Load data with required columns names"""
    # Setting possible priorities for features

    default_columns = list(['ID (автономер в базе)', 'Фамилия', 'Имя', 'Отчество'])
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
        required_priorities.extend(default_columns)
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
        required_priorities.extend(['ID (автономер в базе)', 'Явка на смене (Смена)', 'Тип биллинга'])
        for priority in priorities:
            required_priorities.extend(column_names[priority])

        # Loading data with staff working-results
        data = pd.read_excel(path)[required_priorities]
    return data


def update_csv():
    """Temporary function for updating temporary CSV files"""
    priorities = ['Важный',
                  'Средняя']
    data_features = load_features(priorities=priorities)
    data_features.to_csv('../data/tmp/F13.csv', encoding='cp1251')
    data_target = load_targets()
    data_target.to_csv('../data/tmp/T13.csv', encoding='cp1251')
    print('Update CSVs completed!')


def modify_target(data_target):
    """Special function for modification data from the 2nd (Target) file"""
    # TODO add cleaning from Peaks
    temp_names = list(data_target)
    tqdm.pandas(desc="Calculate QualityRatio for staff")
    # Calculate QualityRatio:
    data_target['QualityRatioTotal'] = data_target.progress_apply(quality_ratio2, axis=1, args=(0.5, 0.9))

    # Additional metrics of Quality.
    # data_target['QualityRatioQSP'] = data_target.progress_apply(quality_ratio2, axis=1, args=(0.5, 0.9), mode='QSP')
    # data_target['QualityRatioQScan'] = data_target.progress_apply(quality_ratio2, axis=1, args=(0.5, 0.9), mode='QScan')

    # Dropping unnecessary columns:
    data_target.drop(temp_names[1:], axis=1, inplace=True)
    temp_names = list(data_target)
    # Delete null-values of QualityRatio.
    data_target.dropna(how='any', subset=temp_names[1:], inplace=True)

    # Draw plots of distribution:
    # plot_hist(data_target['QualityRatioTotal'])
    # plot_hist(data_target['QualityRatioQSP'], 'QSP')
    # plot_hist(data_target['QualityRatioQScan'], 'QScan')
    return data_target


def features_fillna(train_data):
    """Special function for handling missing data"""
    # Transform Birthday column to DataTime format:
    train_data['Дата рождения'] = pd.to_datetime(train_data['Дата рождения'], errors='coerce')

    # --------------------------------- FillNA ---------------------------------
    # TODO add conversion from age column to integer - is it necessary??
    age_mask = (train_data['Возраст'].isnull()) & (train_data['Дата рождения'].notnull())
    train_data['Возраст'][age_mask] = train_data[age_mask].apply(fix_age, axis=1)

    # FillNA for Patronymic
    # TODO add cleaning from rubbish such as '---' and '0---'
    train_data['Отчество'].fillna('Не указано', inplace=True)
    # FillNA for family
    train_data['Семейное положение'].fillna('Не указано', inplace=True)
    # FillNA for Attraction in work: 1 if some text was typed and 0 otherwise.
    # TODO add bag of words - is it necessary??
    train_data['Что привлекает в работе'][train_data['Что привлекает в работе'].notnull()] = \
        train_data['Что привлекает в работе'].notnull().apply(lambda t: 0.5)
    train_data['Что привлекает в работе'].fillna(0, inplace=True)
    # Fill NA for current position
    train_data['Должность'][train_data['Должность'].notnull()] = \
        train_data['Должность'].notnull().apply(lambda t: 0.5)
    train_data['Должность'].fillna(0, inplace=True)

    return train_data.dropna()


def features2vector(train_data):
    """Special function for conversion categorical to binary array"""
    # ------------------------------ CATEGORICAL ------------------------------
    # Splitting BD to day and month + getting zodiac sign
    tqdm.pandas(desc="Splitting BD to day   ")
    train_data['DayOfBirth'] = train_data['Дата рождения'].progress_apply(lambda t: t.day)
    tqdm.pandas(desc="Splitting BD to month ")
    train_data['MonthOfBirth'] = train_data['Дата рождения'].progress_apply(lambda t: t.month)
    tqdm.pandas(desc="Getting zodiac sign   ")
    train_data['Zodiac'] = train_data['Дата рождения'].progress_apply(zodiac)
    # Work with categorical features such as Name
    tqdm.pandas(desc="Work with names:      ")
    train_data['Имя'] = train_data['Имя'].progress_apply(lambda t: t.lower())
    # TODO change to OneHotEncoder for test data transformation. - is it necessary??
    dummies = pd.get_dummies(train_data, columns=['Имя', 'Отчество', 'Пол', 'Дети', 'Семейное положение',
                                                  'Есть основная работа', 'Zodiac'])
    return dummies

def data_preprocessing(data):
    """Drop unnecessary columns and Scaling, Standardization and Normalization"""
    drop_titles = ['ID (автономер в базе)', 'Фамилия', 'Дата рождения']
    data.drop(drop_titles, axis=1, inplace=True)
    print(data['DayOfBirth'])
    titles = list(data)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    rescaledData = pd.DataFrame(scaler.fit_transform(data.values),
                                 index=data.index,
                                 columns=data.columns)
    # print(rescaledData)
    # scaler2 = preprocessing.StandardScaler().fit(rescaledData)
    # standardizeData = scaler2.transform(rescaledData)
    # print(standardizeData)
    print(rescaledData)


def load_data():
    """Function for loading data and returning data_features and data_target DataFrame"""
    # Loading from original Excel files:
    # TODO try to fix encoding
    # data_features = load_features(priorities=priorities)
    # data_features.to_csv('../data/tmp/F13.csv', encoding='cp1251')
    # data_target = load_targets()
    # data_target.to_csv('../data/tmp/T13.csv', encoding='cp1251')

    # Loading from steady-files:
    # update_csv()
    data_features = pd.read_csv('../data/tmp/F13.csv', encoding='cp1251',
                                index_col=0)
    data_target = modify_target(pd.read_csv('../data/tmp/T13.csv', encoding='cp1251',
                                            index_col=0))
    # TODO add cleaning data!!
    # Merge 2 parts of data to one DataFrame
    data = features_fillna(data_features.merge(data_target,
                                               on='ID (автономер в базе)'))
    X = features2vector(data[list(data_features)])
    Y = data[list(data_target)[1:]]

    return X, Y


# ----------------------------------------- Munging data -----------------------------------------
# 'Явка на смене (Смена)', 'Востребована оплата по смене', 'Выработка % от нормы по сканированию (Qscan)',
# 'Выработка % от нормы по ручному пересчету (QSP)', 'QTotalCalcType', 'QTotal', 'Ошибок сканирования (штук)',
# 'Статус смены (Смена)'
# Обращение по индекса не только не ускоряет процесс, но и замедляет его на 20%.
def quality_ratio(row):
    """Special function for calculating QualityRatio for staff"""
    if row['QTotalCalcType'] == 'По ставке':
        if row['Явка на смене (Смена)'] == 'Да':
            return 1
        elif row['Статус смены (Смена)'] == 'Подтвержден':
            return 0
        else:
            return 0.5
    elif row['QTotalCalcType'] == 'По выработке' and row['Явка на смене (Смена)'] == 'Да':
        if row['Выработка % от нормы по сканированию (Qscan)'] >= 0.85:
            return 1
        elif row['Выработка % от нормы по сканированию (Qscan)'] < 0.5:
            return 0
        else:
            value = (row['Выработка % от нормы по сканированию (Qscan)'] - 0.5) / 0.35
            return value
    else:
        return 0


# TODO ask about QTotal or QScan or QSP
# Another if-else structure, looks better:
def quality_ratio2(row, qscan_min=0.5, qscan_max=0.85, mode='QTotal'):
    """Special function for calculating QualityRatio for staff"""
    # Define the work mode:
    if mode == 'QSP':
        required_title = 'Выработка % от нормы по ручному пересчету (QSP)'
    elif mode == 'QScan':
        required_title = 'Выработка % от нормы по сканированию (Qscan)'
    else:
        required_title = 'QTotal'
    # Working:
    if row['Тип биллинга'] == 'Первичный':
        if row['Явка на смене (Смена)'] == 'Да':
            if row['QTotalCalcType'] == 'По ставке':
                return 1
            elif row['QTotalCalcType'] == 'По выработке':
                # Additional statement for skipping HighPeeks:
                if row[required_title] < 15:
                    if row[required_title] >= qscan_max:
                        return 1
                    elif row[required_title] < qscan_min:
                        return 0
                    else:
                        value = (row[required_title] - qscan_min) / (qscan_max - qscan_min)
                        return value
        elif row['Статус смены (Смена)'] == 'Подтвержден':
            return 0
    elif row['Тип биллинга'] == 'Штрафной':
        return 0
    else:
        return 1


def fix_age(row):
    return 2014 - row['Дата рождения'].year


def calendar_info(value):
    """Special function for conversion Date of Birth to Day and Month of Birth"""
    tmp = list()
    tmp.append(value.day)
    tmp.append(value.month)
    return tmp

def zodiac(value):

    zodiacs = [(120, 'Cap'), (218, 'Aqu'), (320, 'Pis'), (420, 'Ari'), (521, 'Tau'),
               (621, 'Gem'), (722, 'Can'), (823, 'Leo'), (923, 'Vir'), (1023, 'Lib'),
               (1122, 'Sco'), (1222, 'Sag'), (1231, 'Cap')]

    date_number = value.month*100+value.day
    for z in zodiacs:
        if date_number <= z[0]:
            return z[1]


def plot_hist(x, mode='QTotal'):
    plt.hist(x, 21, facecolor='g', alpha=0.75)
    plt.xlabel(mode)
    plt.ylabel('Counts')
    plt.title('Histogram of ' + str(mode))
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.grid(True)
    if not os.path.exists('../data/plots'):
        os.makedirs('../data/plots')
    plt.savefig('../data/plots/Column' + str(mode) + '.png')
    plt.show()

def plot_bar(x, title=''):
    plt.bar(x)

def missing_data(data):
    """Analysis data and find missing values"""
    counts = data.describe(include='all').loc[:'count'].T
    print(counts)
    total = len(data)
    missed_data = counts[counts['count'] <= total].apply(lambda tmp:
                                                         (total - tmp) / total)['count']
    print("Количество пропусков: ")
    print(missed_data.sort_values(ascending=False))


# Clear data function:
def clear_txt(txt):
    txt = txt.apply(lambda t: t.lower())
    txt = txt.replace('[^a-z0-9]', ' ', regex=True)
    return txt
