# Some of custom functions
# AIRY v.0.1

import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_pandas, trange
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


# ----------------------------------------- Loading  Features data -----------------------------------------
def load_features(path='../data/Features2013.xlsx',
                  priorities=['Важный'],
                  forceAll=False):
    """Load data with required columns names"""
    # Setting possible priorities for features

    default_columns = list(['ID (автономер в базе)', 'Фамилия', 'Имя', 'Отчество'])
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


# ----------------------------------------- Loading Target data -----------------------------------------
def load_targets(path='../data/Target2013.xlsx',
                 priorities=['Высокая'],
                 forceAll=False):
    """Load target-data with required columns names"""
    # Setting possible priorities for features
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


# ----------------------------------------- Update temporary CSV files for boostUp -------------------------------------
def update_csv(use=''):
    """Temporary function for updating temporary CSV files"""
    priorities = ['Важный',
                  'Средняя']
    priorities.extend(use)
    data_features = load_features(priorities=priorities)
    data_features.to_csv('../data/tmp/F13.csv', encoding='cp1251')
    data_target = load_targets()
    data_target.to_csv('../data/tmp/T13.csv', encoding='cp1251')
    print('Update CSVs completed!')


# ----------------------------------------- Features fillNA ----------------------------------------------------
def features_fillna(data):
    """Special function for handling missing data"""
    # Transform Birthday column to DataTime format:
    data['Дата рождения'] = pd.to_datetime(data['Дата рождения'], errors='coerce')

    # --------------------------------- FillNA ---------------------------------
    # TODO add conversion from age column to integer - is it necessary??
    age_mask = (data['Возраст'].isnull()) & (data['Дата рождения'].notnull())
    data['Возраст'][age_mask] = data[age_mask].apply(fix_age, axis=1)
    # FillNA for Patronymic
    # TODO add cleaning from rubbish such as '---' and '0---'

    data['Отчество'].fillna('Не указано', inplace=True)
    # FillNA for family
    data['Семейное положение'].fillna('Не указано', inplace=True)
    # FillNA for Attraction in work: 1 if some text was typed and 0 otherwise.
    # TODO add bag of words - is it necessary??
    data['Что привлекает в работе'][data['Что привлекает в работе'].notnull()] = \
        data['Что привлекает в работе'].notnull().apply(lambda t: 0.5)
    data['Что привлекает в работе'].fillna(0, inplace=True)
    # Fill NA for current position
    data['Должность'][data['Должность'].notnull()] = \
        data['Должность'].notnull().apply(lambda t: 0.5)
    data['Должность'].fillna(0, inplace=True)

    return data.dropna()


# ----------------------------------------- Get age info from BD column ----------------------------------------
def fix_age(row):
    """Special function for calculating age via Birthday"""
    return 2014 - row['Дата рождения'].year


# ----------------------------------------- Get zodiac sign from BD column -------------------------------------
def zodiac(value):
    """Found zodiac sign from Birthday information"""
    zodiacs = [(120, 'Cap'), (218, 'Aqu'), (320, 'Pis'), (420, 'Ari'), (521, 'Tau'),
               (621, 'Gem'), (722, 'Can'), (823, 'Leo'), (923, 'Vir'), (1023, 'Lib'),
               (1122, 'Sco'), (1222, 'Sag'), (1231, 'Cap')]

    date_number = value.month * 100 + value.day
    for z in zodiacs:
        if date_number <= z[0]:
            return z[1]


# ----------------------------------------- Additional features -------------------------------------
def add_features(data, split_bd=True, zodiac_sign=True):
    """Special function for add features as day and month of birth and zodiac sign"""
    # Splitting BD to day and month + getting zodiac sign
    if split_bd:
        tqdm.pandas(desc="Splitting BD to day   ")
        data['DayOfBirth'] = data['Дата рождения'].progress_apply(lambda t: t.day)
        tqdm.pandas(desc="Splitting BD to month ")
        data['MonthOfBirth'] = data['Дата рождения'].progress_apply(lambda t: t.month)
    if zodiac_sign:
        tqdm.pandas(desc="Getting zodiac sign   ")
        data['Zodiac'] = data['Дата рождения'].progress_apply(zodiac)
    # tqdm.pandas(desc="Binarize work info    ")
    # data['Есть основная работа'] = data['Есть основная работа'].progress_apply(lambda x: 1 if x == 'Да' else 0)
    return data


# ----------------------------------------- Vectorize features -------------------------------------
def vectorize(data, titles=['Имя', 'Отчество', 'Пол', 'Дети', 'Семейное положение']):
    # Work with categorical features such as Name
    # TODO change to OneHotEncoder for test data transformation. - is it necessary??
    dummies = pd.get_dummies(data, columns=titles, sparse=False)
    return dummies


# ----------------------------------------- Get father's names + lowercase() -------------------------------------
def modify_names(data):
    """Special function for cutting father's names"""
    tqdm.pandas(desc="Work with names       ")
    data['Имя'] = data['Имя'].progress_apply(lambda t: t.lower())
    tqdm.pandas(desc="Work with patronymic  ")
    data['Отчество'] = data['Отчество'].progress_apply(patronymic)
    return data


# ----------------------------------------- Modify target data -------------------------------------
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


# ----------------------------------------- Calculate target var for staff -------------------------------------
def binary_quality_ratio(row, alpha=0.7, mode='QTotal'):
    """Special function for calculating BinaryQualityRatio for staff"""
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
                    if row[required_title] >= alpha:
                        return 1
                    else:
                        return 0
        elif row['Статус смены (Смена)'] == 'Подтвержден':
            return 0
    elif row['Тип биллинга'] == 'Штрафной':
        return 0
    else:
        return 1


# ----------------------------------------- Modify target data -------------------------------------


def binarize_target(data_target):
    """Special function for conversion" raw-target to [0, 1] values"""
    # TODO add cleaning from Peaks
    temp_names = list(data_target)
    tqdm.pandas(desc="Calculate QualityRatio for staff")
    # Calculate QualityRatio:
    data_target['QualityRatioTotal'] = data_target.progress_apply(binary_quality_ratio, axis=1)
    # Dropping unnecessary columns:
    data_target.drop(temp_names[1:], axis=1, inplace=True)
    temp_names = list(data_target)
    # Delete null-values of QualityRatio.
    data_target.dropna(how='any', subset=temp_names[1:], inplace=True)
    # Draw plots of distribution:
    # plot_hist(data_target['QualityRatioTotal'])
    return data_target


# ----------------------------------------- Calculate target var for staff -------------------------------------
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


# ----------------------------------------- Load data from files and return X, binary Y --------------------------------
def load_data_bin():
    """Loading data and returning data_features and data_target DataFrames. Return required columns as is"""
    # Loading from steady-files:
    # update_csv(use='A')
    data_features = pd.read_csv('../data/tmp/F13.csv', encoding='cp1251',
                                index_col=0)
    print("Features: ", data_features.shape)
    data_target = binarize_target(pd.read_csv('../data/tmp/T13.csv', encoding='cp1251',
                                              index_col=0))
    print("Target: ", data_target.shape)
    # Merge 2 parts of data to one DataFrame
    data = data_features.merge(data_target,
                               on='ID (автономер в базе)')
    print("Merged: ", data.shape)
    data = features_fillna(data)
    print("FillNA: ", data.shape)

    X = data[list(data_features)]
    Y = data[list(data_target)[1:]]
    return X, Y


# ----------------------------------------- Print statistic of DataFrame -------------------------------------
def missing_data(data, plot=False, title='Features'):
    """Analysis data and find missing values"""
    counts = data.describe(include='all').loc[:'count'].T.sort_values(by='count', ascending=False)
    if plot:
        plt.show(counts.head(15).plot.bar())
    print(counts)
    total = len(data)
    missed_data = counts[counts['count'] <= total].apply(lambda tmp:
                                                         (total - tmp) / total)['count']
    print("Количество пропусков: ")
    miss_sort = missed_data.sort_values(ascending=True)
    print(miss_sort)

    return miss_sort
    # Draw and save plot
    # plt.rcParams.update({'font.size': 22})
    # fig = plt.figure("Data analysis: ", figsize=(16, 12))
    # fig.suptitle('Data analysis', fontweight='bold')
    # ax = fig.add_subplot(111)
    # ax.set_title(title, fontdict={'fontsize': 10})
    # ax.set_ylabel('Score')
    # ax.set_xlabel('log(C)')
    # ax.plot(_range, quality, 'g', linewidth=2)
    # ax.grid(True)
    # if not os.path.exists('../data/plots/DataAnalysis/'):
    #     os.makedirs('../data/plots/DataAnalysis/')
    # plt.savefig('../data/plots/DataAnalysis/' +title+datetime.now().strftime('%Y%m%d_%H%M%S') + '.png')


# -----------------------------------------  Logistic Regression   -------------------------------------
def LR():
    """Testing linear method for train"""
    train_data, train_target = load_data_bin()
    train_data = add_features(train_data)
    # drop_titles = ['ID (автономер в базе)', 'Имя', 'Отчество', 'Фамилия', 'Дата рождения']
    drop_titles = ['ID (автономер в базе)', 'Фамилия', 'Дата рождения']
    train_data.drop(drop_titles, axis=1, inplace=True)
    categorical_titles = list(train_data.select_dtypes(exclude=[np.number]))
    work_titles = list(train_data)
    train_data = vectorize(train_data, titles=categorical_titles)
    # KFold for splitting
    cv = KFold(n_splits=5,
               shuffle=True,
               random_state=241)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    rescaledData = pd.DataFrame(scaler.fit_transform(train_data.values),
                                index=train_data.index,
                                columns=train_data.columns)
    start = timer()
    lr = linear_model.LogisticRegression(C=10,
                                         random_state=241,
                                         n_jobs=-1)
    scores = cross_val_score(lr, rescaledData, train_target['QualityRatioTotal'],
                             cv=cv, scoring='roc_auc',
                             n_jobs=-1)
    score = np.mean(scores)
    tt = timer() - start
    print("C parameter is " + str(10))
    print("Score is ", score)
    print("Time elapsed: ", tt)
    print("""-----------¯\_(ツ)_/¯ -----------""")


# ----------------------------------------- Test Logistic Regression  -------------------------------------
def test_logistic(title=''):
    """Testing linear method for train"""
    train_data, train_target = load_data_bin()
    train_data = add_features(train_data)
    # drop_titles = ['ID (автономер в базе)', 'Имя', 'Отчество', 'Фамилия', 'Дата рождения']
    drop_titles = ['ID (автономер в базе)', 'Фамилия', 'Дата рождения']
    train_data.drop(drop_titles, axis=1, inplace=True)
    train_data = modify_names(train_data)
    categorical_titles = list(train_data.select_dtypes(exclude=[np.number]))
    work_titles = list(train_data)
    train_data = vectorize(train_data, titles=categorical_titles)
    # KFold for splitting
    cv = KFold(n_splits=5,
               shuffle=True,
               random_state=241)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    rescaledData = pd.DataFrame(scaler.fit_transform(train_data.values),
                                index=train_data.index,
                                columns=train_data.columns)
    quality = []
    time = []
    _range = np.arange(-4, 2)
    C = np.power(10.0, _range)
    for c in C:
        start = timer()
        lr = linear_model.LogisticRegression(C=c,
                                             random_state=241,
                                             n_jobs=-1)
        scores = cross_val_score(lr, rescaledData, train_target['QualityRatioTotal'],
                                 cv=cv, scoring='roc_auc',
                                 n_jobs=-1)
        score = np.mean(scores)
        quality.append(score)
        tt = timer() - start
        time.append(tt)
        print("C parameter is " + str(c))
        print("Score is ", score)
        print("Time elapsed: ", tt)
        print("""-----------¯\_(ツ)_/¯ -----------""")

        # Draw it:
    score_best = max(quality)
    idx = quality.index(score_best)
    C_best = C[idx]
    time_best = time[idx]
    print("Наилучший результат достигается при C=" + str(C_best))
    print("Score is ", score_best)
    print("Time elapsed: ", time_best)
    # Draw and save plot
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure("Logistic regression: ", figsize=(16, 12))
    fig.suptitle('Logistic regression  ' + str(score_best), fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_title(str(work_titles), fontdict={'fontsize': 10})
    ax.set_ylabel('Score')
    ax.set_xlabel('log(C)')
    ax.plot(_range, quality, 'g', linewidth=2)
    ax.grid(True)
    if not os.path.exists('../data/plots/Models/LogisticRegression/'):
        os.makedirs('../data/plots/Models/LogisticRegression/')
    plt.savefig('../data/plots/Models/LogisticRegression/' + title + datetime.now().strftime('%Y%m%d_%H%M%S') + '.png')


# ----------------------------------------- Test Neural Network  -------------------------------------
def test_neural():
    """Testing linear method for train"""
    train_data, train_target = load_data_bin()
    train_data = add_features(train_data)
    drop_titles = ['ID (автономер в базе)', 'Фамилия', 'Дата рождения']
    train_data.drop(drop_titles, axis=1, inplace=True)
    train_data = modify_names(train_data)
    categorical_titles = list(train_data.select_dtypes(exclude=[np.number]))
    work_titles = list(train_data)
    train_data = vectorize(train_data, titles=categorical_titles)
    # KFold for splitting
    cv = KFold(n_splits=5,
               shuffle=True,
               random_state=241)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    rescaledData = pd.DataFrame(scaler.fit_transform(train_data.values),
                                index=train_data.index,
                                columns=train_data.columns)
    quality = []
    time = []
    _range = np.arange(-5, 3)
    C = np.power(10.0, _range)
    for c in C:
        start = timer()
        lr = MLPClassifier(alpha=c, hidden_layer_sizes=(10, 10),
                           random_state=241)
        scores = cross_val_score(lr, rescaledData, train_target['QualityRatioTotal'],
                                 cv=cv, scoring='roc_auc',
                                 n_jobs=-1)
        score = np.mean(scores)
        quality.append(score)
        tt = timer() - start
        time.append(tt)
        print("C parameter is " + str(c))
        print("Score is ", score)
        print("Time elapsed: ", tt)
        print("""-----------¯\_(ツ)_/¯ -----------""")

    # Draw it:
    score_best = max(quality)
    idx = quality.index(score_best)
    C_best = C[idx]
    time_best = time[idx]
    print("Наилучший результат достигается при C=" + str(C_best))
    print("Score is ", score_best)
    print("Time elapsed: ", time_best)
    # Draw and save plot
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure("Neural network: ", figsize=(16, 12))
    fig.suptitle('MLPClassifier', fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_title(str(work_titles), fontdict={'fontsize': 10})
    ax.set_ylabel('Score')
    ax.set_xlabel('log(C)')
    ax.plot(_range, quality, 'g', linewidth=2)
    ax.grid(True)
    if not os.path.exists('../data/plots/Models/Neural/'):
        os.makedirs('../data/plots/Models/Neural/')
    plt.savefig('../data/plots/Models/Neural/MLPClassifier_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.png')


# ----------------------------------------- Neural Network Model -------------------------------------
def neural(c=0.1):
    """Build neural network model"""
    train_data, train_target = load_data_bin()
    train_data = add_features(train_data)
    drop_titles = ['ID (автономер в базе)', 'Фамилия', 'Дата рождения']
    train_data.drop(drop_titles, axis=1, inplace=True)
    train_data = modify_names(train_data)
    categorical_titles = list(train_data.select_dtypes(exclude=[np.number]))
    work_titles = list(train_data)
    train_data = vectorize(train_data, titles=categorical_titles)
    # KFold for splitting
    cv = KFold(n_splits=5,
               shuffle=True,
               random_state=241)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    rescaledData = pd.DataFrame(scaler.fit_transform(train_data.values),
                                index=train_data.index,
                                columns=train_data.columns)
    start = timer()

    lr = MLPClassifier(alpha=c, hidden_layer_sizes=(100,),
                       random_state=241)
    scores = cross_val_score(lr, rescaledData, train_target['QualityRatioTotal'],
                             cv=cv, scoring='roc_auc',
                             n_jobs=-1)
    score = np.mean(scores)
    tt = timer() - start
    print("C parameter is " + str(c))
    print("Score is ", score)
    print("Time elapsed: ", tt)
    print("""-----------¯\_(ツ)_/¯ -----------""")


def patronymic(t):
    t = t.lower()
    if t[-4:] in ['ович', 'евич', 'овна', 'евна']:
        return t[:-4]
    elif t[-1:] in ['-', '0', '6', '7']:
        return 'Не указано'
    else:
        return t
