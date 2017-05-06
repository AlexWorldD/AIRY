# Some of custom functions
# AIRY v.0.1

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm, tqdm_pandas, trange
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2
import MultiColumnEncoder

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


# ----------------------------------------- Update temporary CSV files for boostUp -------------------------------------
def get_csv():
    """Temporary function for updating temporary CSV files"""
    data_features = pd.read_excel('../data/FULL/Features.xlsx')
    data_features.to_csv('../data/FULL/F.csv', encoding='cp1251')
    data_target = pd.read_excel('../data/FULL/Targets.xlsx')
    data_target.to_csv('../data/FULL/T.csv', encoding='cp1251')
    print('Update CSVs completed!')


# ----------------------------------------- Features fillNA ----------------------------------------------------
def features_fillna(data, fillna=True):
    """Special function for handling missing data"""
    # Transform Birthday column to DataTime format:
    data['Дата рождения'] = pd.to_datetime(data['Дата рождения'], errors='coerce')
    # TODO add conversion from age column to integer - is it necessary??
    age_mask = (data['Возраст'].isnull()) & (data['Дата рождения'].notnull())
    data['Возраст'][age_mask] = data[age_mask].apply(fix_age, axis=1)
    # FillNA for Attraction in work: 1 if some text was typed and 0 otherwise.
    # TODO add bag of words - is it necessary??
    data['Что привлекает в работе'][data['Что привлекает в работе'].notnull()] = \
        data['Что привлекает в работе'].notnull().apply(lambda t: 0.5)
    # Fill NA for current position
    data['Должность'][data['Должность'].notnull()] = \
        data['Должность'].notnull().apply(lambda t: 0.5)
    data['Должность'].fillna(0, inplace=True)
    if fillna:
        # --------------------------------- FillNA ---------------------------------
        # FillNA for Patronymic
        # TODO add cleaning from rubbish such as '---' and '0---'

        data['Отчество'].fillna('Не указано', inplace=True)
        # FillNA for family
        data['Семейное положение'].fillna('Не указано', inplace=True)
        data['Что привлекает в работе'].fillna(0, inplace=True)
        data['Есть имейл (указан сервис)'].fillna('Не указано', inplace=True)
        data['Первые 4 цифры моб телефона'].fillna(0, inplace=True)
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
def category_encode(data, titles=['Имя', 'Отчество', 'Пол', 'Дети', 'Семейное положение'], mode='OneHot'):
    # Work with categorical features such as Name
    # TODO change to OneHotEncoder for test data transformation. - is it necessary??
    if mode == 'OneHot':
        data = pd.get_dummies(data, columns=titles, sparse=False)
    elif mode == 'LabelsEncode':
        data = MultiColumnEncoder.EncodeCategorical(columns=titles).fit_transform(data)
    return data


# ----------------------------------------- Get father's names + lowercase() -------------------------------------
def modify_names(data, popular=False):
    """Special function for cutting father's names"""
    tqdm.pandas(desc="Work with names       ")
    data['Имя'] = data['Имя'].progress_apply(lambda t: t.lower())
    if popular:
        tqdm.pandas(desc="Work with NAME.v2       ")
        data['Имя'] = data['Имя'].progress_apply(popular_names)
    tqdm.pandas(desc="Work with patronymic  ")
    data['Отчество'] = data['Отчество'].progress_apply(patronymic)
    return data


def popular_names(t):
    popular = ['светлана', 'екатерина', 'александр', 'елена', 'татьяна', 'ирина', 'сергей', 'ольга', 'наталья',
               'алексей', 'анна', 'юлия', 'мария', 'олег', 'оксана', 'марина', 'дмитрий', 'анастасия', 'надежда',
               'александра', 'евгений', 'андрей', 'владимир', 'максим', 'наталия', 'станислав', 'жанна', 'роман',
               'денис', 'людмила', 'виталий', 'иван', 'галина', 'владислав', 'игорь', 'евгения', 'виктория', 'михаил',
               'валентина', 'алена', 'любовь', 'дарья', 'никита', 'антон', 'виолета', 'николай', 'олеся', 'юрий',
               'ксения', 'василий', 'геннадий', 'павел', 'нина', 'эльвира', 'алина', 'лариса', 'артем', 'константин',
               'кристина', 'валерий', 'махамаджан', 'вячеслав', 'виктор', 'илья', 'анатолий', 'кирилл', 'маргарита',
               'вера', 'руслан', 'лилия', 'лидия', 'яна', 'алла', 'ростислав', 'айгуль', 'инна', 'эдуард', 'солохидин',
               'алёна']
    if t not in popular:
        return 'Редкое имя'
    else:
        return t


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


def binarize_target(data_target, with_type=False):
    """Special function for conversion" raw-target to [0, 1] values"""
    # TODO add cleaning from Peaks
    temp_names = list(data_target)
    # Delete QTotalCalcType from title for deleting
    if with_type:
        temp_names.remove('QTotalCalcType')
    temp_names.remove('ID (автономер в базе)')
    tqdm.pandas(desc="Calculate QualityRatio for staff")
    # Calculate QualityRatio:
    data_target['QualityRatioTotal'] = data_target.progress_apply(binary_quality_ratio, axis=1)
    # Dropping unnecessary columns:
    data_target.drop(temp_names, axis=1, inplace=True)
    temp_names = list(data_target)
    # Delete null-values of QualityRatio.
    data_target.dropna(how='any', subset=temp_names, inplace=True)
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


# ----------------------------------------- Loading data v2.0 -------------------------------------
def load_data(transform_category='', drop='', fillna=True):
    """Loading data from steady CSV-files"""
    # Loading from steady-files:
    # update_csv(use='A')
    data_features = pd.read_csv('../data/tmp/F13.csv', encoding='cp1251',
                                index_col=0)
    print("Features: ", data_features.shape)
    data_target = binarize_target(pd.read_csv('../data/tmp/T13.csv', encoding='cp1251',
                                              index_col=0), with_type=False)
    print("Target: ", data_target.shape)
    # Merge 2 parts of data to one DataFrame
    data = data_features.merge(data_target,
                               on='ID (автономер в базе)')
    print("Merged: ", data.shape)
    # missing_data(data, plot=True)
    data = features_fillna(data, fillna=fillna)
    print("FillNA: ", data.shape)

    # Munging data
    data = add_features(data)
    # drop_titles = ['ID (автономер в базе)', 'Имя', 'Отчество', 'Фамилия', 'Дата рождения']
    drop_titles = ['ID (автономер в базе)', 'Фамилия', 'Дата рождения']
    data.drop(drop_titles, axis=1, inplace=True)
    # ---- NAMES ----
    data = modify_names(data)
    # ---- Email ----
    tqdm.pandas(desc="Work with email  ")
    data['Есть имейл (указан сервис)'] = data['Есть имейл (указан сервис)'].progress_apply(email)
    # ---- Add MOBILE ----
    data = get_mobile(data)

    # Drop required columns:
    if not drop == '':
        data.drop(drop, axis=1, inplace=True)
    # ---- Categorical ----
    categorical_titles = list(data.select_dtypes(exclude=[np.number]))
    # print(categorical_titles)
    work_titles = list(data)
    if transform_category in ['OneHot', 'LabelsEncode']:
        data = category_encode(data, titles=categorical_titles, mode=transform_category)
    t_t = list(data)
    t_t.remove('QualityRatioTotal')
    X = data[t_t]
    Y = data[list(data_target)[1:]]
    return X, Y, work_titles


# ----------------------------------------- Load data from Excel files and return X, binary Y -------------------------
def load_data_from_file(path='../data/Target2013_v2.xlsx',
                        use='',
                        forceAll=False):
    """Loading data and returning data_features and data_target DataFrames. Return required columns as is"""
    # Loading from steady-files:
    # update_csv(use='A')
    priorities = ['Важный',
                  'Средняя']
    priorities.extend(use)
    data_features = load_features(priorities=priorities, forceAll=forceAll)
    data_target = load_targets(path=path, forceAll=forceAll)
    print('Download from Excel!')
    print("Features: ", data_features.shape)
    data_target = binarize_target(data_target)

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
        # plt.figure("Data Analysis: ", figsize=(10, 6))
        plt.subplot(counts.head(25).plot.barh())
        plt.tight_layout()
        plt.show()
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
def LR(C=1, drop='', scoring='roc_auc', selectK=''):
    """Testing linear method for train"""
    train_data, train_target, work_titles = load_data(transform_category=True, drop=drop)
    if not selectK == '':
        print(train_data.shape)
        train_data_new = SelectKBest(chi2, k=selectK).fit_transform(train_data, train_target)
        print(train_data_new.shape)
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        rescaledData = pd.DataFrame(scaler.fit_transform(train_data_new),
                                    index=train_data.index)
    else:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        rescaledData = pd.DataFrame(scaler.fit_transform(train_data.values),
                                    index=train_data.index,
                                    columns=train_data.columns)
    # KFold for splitting
    cv = KFold(n_splits=5,
               shuffle=True,
               random_state=241)

    start = timer()

    # Possible scoring function:
    sc = ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples',
          'f1_weighted',
          'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision',
          'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall',
          'recall_macro',
          'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
    if scoring not in sc:
        print('Your scoring finction is not support! Change to default..')
        scoring = 'roc_auc'

    lr = linear_model.LogisticRegression(C=C,
                                         random_state=241,
                                         n_jobs=-1)
    # model = SelectFromModel(lr, prefit=True)
    # X_new = model.transform(rescaledData)
    # print(X_new.shape)
    scores = cross_val_score(lr, rescaledData, train_target['QualityRatioTotal'],
                             cv=cv, scoring=scoring,
                             n_jobs=-1)
    score = np.mean(scores)
    tt = timer() - start
    print("C parameter is " + str(C))
    print("Score is ", score)
    print("Time elapsed: ", tt)
    print("""-----------¯\_(ツ)_/¯ -----------""")
    return score, selectK


# ----------------------------------------- Test Logistic Regression  -------------------------------------
def test_logistic(drop='', scoring='roc_auc', title='', selectK='', fillna=True):
    """Testing linear method for train"""
    # train_data, train_target = load_data_bin()
    # train_data = add_features(train_data)
    # # drop_titles = ['ID (автономер в базе)', 'Имя', 'Отчество', 'Фамилия', 'Дата рождения']
    # drop_titles = ['ID (автономер в базе)', 'Фамилия', 'Дата рождения']
    # train_data.drop(drop_titles, axis=1, inplace=True)
    # # ---- NAMES ----
    # train_data = modify_names(train_data)
    # # ---- Email ----
    # tqdm.pandas(desc="Work with email  ")
    # train_data['Есть имейл (указан сервис)'] = train_data['Есть имейл (указан сервис)'].progress_apply(email)
    # # ---- Add MOBILE ----
    # train_data = get_mobile(train_data)
    # # ---- Categorical ----
    # categorical_titles = list(train_data.select_dtypes(exclude=[np.number]))
    # # print(categorical_titles)
    # work_titles = list(train_data)
    # train_data = vectorize(train_data, titles=categorical_titles)
    train_data, train_target, work_titles = load_data(transform_category=True, drop=drop, fillna=fillna)
    if not selectK == '':
        print(train_data.shape)
        train_data_new = SelectKBest(chi2, k=selectK).fit_transform(train_data, train_target)
        print(train_data_new.shape)
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        rescaledData = pd.DataFrame(scaler.fit_transform(train_data_new),
                                    index=train_data.index)
    else:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        rescaledData = pd.DataFrame(scaler.fit_transform(train_data.values),
                                    index=train_data.index,
                                    columns=train_data.columns)
    # KFold for splitting
    cv = KFold(n_splits=5,
               shuffle=True,
               random_state=241)

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
                                 cv=cv, scoring=scoring,
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
    print("Score is ", score_best, ' with scoring=', scoring)
    print("Time elapsed: ", time_best)
    # Draw and save plot
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure("Logistic regression: ", figsize=(16, 12))
    fig.suptitle('Logistic regression  ' + str(score_best), fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_title(str(work_titles), fontdict={'fontsize': 10})
    ax.set_ylabel('Score')
    ax.set_xlabel('log(C), ' + scoring)
    ax.plot(_range, quality, 'g', linewidth=2)
    ax.grid(True)
    if not os.path.exists('../data/plots/Models/LogisticRegression/'):
        os.makedirs('../data/plots/Models/LogisticRegression/')
    plt.savefig('../data/plots/Models/LogisticRegression/' + title + datetime.now().strftime('%m%d_%H%M') + '.png')


# ----------------------------------------- Test Neural Network  -------------------------------------
def test_neural(train_data, train_target, work_titles, title='', neural_size=(10, 10)):
    """Testing linear method for train"""
    # train_data, train_target = load_data_bin()
    # train_data = add_features(train_data)
    # drop_titles = ['ID (автономер в базе)', 'Фамилия', 'Дата рождения']
    # train_data.drop(drop_titles, axis=1, inplace=True)
    # train_data = modify_names(train_data)
    # categorical_titles = list(train_data.select_dtypes(exclude=[np.number]))
    # work_titles = list(train_data)
    # train_data = vectorize(train_data, titles=categorical_titles)
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
    _range = np.arange(-4, 1)
    C = np.power(10.0, _range)
    for c in C:
        start = timer()
        lr = MLPClassifier(alpha=c, hidden_layer_sizes=neural_size,
                           random_state=241)
        scores = cross_val_score(lr, rescaledData, train_target,
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
    plt.savefig(
        '../data/plots/Models/Neural/MLPClassifier_' + title + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.png')


def test_GBC(estimators=[10, 20, 50, 100], selecK=200, title=''):
    train_data, train_target, work_titles = load_data(transform_category=True)
    print(train_data.shape)
    train_data_new = SelectKBest(chi2, k=selecK).fit_transform(train_data, train_target)
    print(train_data_new.shape)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    rescaledData = pd.DataFrame(scaler.fit_transform(train_data_new),
                                index=train_data.index)
    # KFold for splitting
    cv = KFold(n_splits=5,
               shuffle=True,
               random_state=241)

    # Build model:
    quality = []
    time = []
    # estimators = [2000]
    for est in estimators:
        start = timer()
        gbc = GradientBoostingClassifier(n_estimators=est,
                                         random_state=241)
        scores = cross_val_score(gbc, rescaledData, train_target,
                                 cv=cv, scoring='roc_auc',
                                 n_jobs=-1)
        score = np.mean(scores)
        quality.append(score)
        print("Estimators number is " + str(est))
        print("Score is ", score)
        tt = timer() - start
        time.append(tt)
        print("Time elapsed: ", tt)
        print("""-----------¯\_(ツ)_/¯ -----------""")

    # Draw it:
    score_best = max(quality)
    idx = quality.index(score_best)
    est_best = estimators[idx]
    time_best = time[idx]
    print("Наилучший результат достигается при n_est=" + str(est_best))
    print("Score is ", score_best)
    print("Time elapsed: ", time_best)
    # Draw and save plot
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure("Gradient Boosting: ", figsize=(16, 12))
    fig.suptitle('GBC', fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_title(str(work_titles), fontdict={'fontsize': 10})
    ax.set_ylabel('Score')
    ax.set_xlabel('N estimators')
    ax.plot(estimators, quality, 'g', linewidth=2)
    ax.grid(True)
    if not os.path.exists('../data/plots/Models/GBC/'):
        os.makedirs('../data/plots/Models/GBC/')
    plt.savefig('../data/plots/Models/GBC/Gradient_' + title + '_' + datetime.now().strftime('%m%d_%H%M') + '.png')


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
    train_data = category_encode(train_data, titles=categorical_titles)
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


# ----------------------------------------- Clean patronymic column -------------------------------------
def patronymic(t):
    """Special function for cleaning patronymic"""
    t = t.lower()
    if t[-4:] in ['ович', 'евич', 'овна', 'евна']:
        return t[:-4]
    elif t[-1:] in ['-', '0', '6', '7']:
        return 'Не указано'
    else:
        return t


# ----------------------------------------- Clean e-mail column -------------------------------------
def email(t):
    """Special function for cleaning patronymic"""
    if not t.find('gmail') == -1 or not t.find('jmail'):
        return 'gmail'
    elif not t.find('mail') == -1:
        return 'mail'
    elif not t.find('ya') == -1 or not t.find('dex'):
        return 'yandex'
    elif not t.find('rambl') == -1:
        return 'rambler'
    elif t in ['Не указано', 'list.ru', 'maxipost.ru', 'bk.ru', 'inbox.ru', 'narod.ru', 'icloud.com']:
        return t
    else:
        return 'Other'


# ----------------------------------------- Get mobile carrier -------------------------------------
def mobile(t, mode='Operator'):
    """Special function for getting information about mobile operator"""
    if int(t) // 10000 in [7, 8]:
        code = int(t) % 100
    else:
        code = int(t) // 100
    if mode == 'Operator':
        MTC = [902, 904, 908, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 950, 978, 980, 981, 982, 983, 984, 985,
               986,
               987, 988, 989]
        BEELINE = [900, 902, 903, 904, 905, 906, 908, 909, 950, 951, 953, 960, 961, 962, 963, 964, 965, 966, 967, 968,
                   969]
        MEGAFON = [920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 936, 937, 938, 939, 999]
        RT = [904, 908, 950, 951, 955, 958, 970, 971, 992]
        T2 = [900, 901, 902, 904, 908, 950, 951, 952, 953, 958, 977, 991, 992, 993, 994, 995, 996, 999]
        if code in MTC:
            return 'MTC'
        elif code in BEELINE:
            return 'Beeline'
        elif code in MEGAFON:
            return 'Megafon'
        elif code in RT:
            return 'RT'
        elif code in T2:
            return 'Tele2'
        else:
            if code == 0:
                return 'Не указано'
            else:
                return 'Unknown'
    else:
        return str(code)


# ----------------------------------------- Getting mobile operator() -------------------------------------
def get_mobile(data, mode='Operator'):
    """Special function for cutting father's names"""
    tqdm.pandas(desc="Work with MOBILE       ")
    data['Mobile'] = data['Первые 4 цифры моб телефона'].progress_apply(mobile, mode=mode)
    data.drop('Первые 4 цифры моб телефона', axis=1, inplace=True)
    return data


# ----------------------------------------- Print bar of required column -------------------------------------
def print_bar(data, tmp='Имя', filna=False, head=10, sort=True, vh=False):
    mails = data.groupby(by=tmp).size()
    if filna:
        mails.drop('Не указано', inplace=True)
    if sort:
        mails.sort_values(ascending=False, inplace=True)
    plt.figure("Neural network: ", figsize=(10, 6))
    if vh:
        plt.subplot(mails.head(head).plot.barh())
    else:
        plt.subplot(mails.head(head).plot.bar())
    plt.tight_layout()
    plt.show()


def load_dataset(split_sex=False, split_QType=False, save_categorical=False):
    """Loading required dataset from steady CSV-files"""
    # Loading from steady-files:
    # update_csv(use='A')
    data_features = pd.read_csv('../data/tmp/F13.csv', encoding='cp1251',
                                index_col=0)
    print("Features: ", data_features.shape)
    data_target = binarize_target(pd.read_csv('../data/tmp/T13.csv', encoding='cp1251',
                                              index_col=0), with_type=split_QType)
    print("Target: ", data_target.shape)
    # Merge 2 parts of data to one DataFrame
    data = data_features.merge(data_target,
                               on='ID (автономер в базе)')
    print("Merged: ", data.shape)
    data = features_fillna(data)
    print("FillNA: ", data.shape)

    # Munging data
    data = add_features(data)
    # drop_titles = ['ID (автономер в базе)', 'Имя', 'Отчество', 'Фамилия', 'Дата рождения']
    drop_titles = ['ID (автономер в базе)', 'Фамилия', 'Дата рождения']
    data.drop(drop_titles, axis=1, inplace=True)
    # ---- NAMES ----
    data = modify_names(data)
    # ---- Email ----
    tqdm.pandas(desc="Work with email  ")
    data['Есть имейл (указан сервис)'] = data['Есть имейл (указан сервис)'].progress_apply(email)
    # ---- Add MOBILE ----
    data = get_mobile(data)
    # ---- Categorical ----
    categorical_titles = list(data.select_dtypes(exclude=[np.number]))
    work_titles = list(data)
    # print(categorical_titles)
    if split_sex:
        categorical_titles.remove('Пол')
        work_titles.remove('Пол')
    if split_QType:
        categorical_titles.remove('QTotalCalcType')

    # Convert category to binary-vectors
    if not save_categorical:
        data = category_encode(data, titles=categorical_titles)

    # Splitting datasets with Sex or QType column:
    train_data = list()
    train_target = list()
    t_t = list(data)
    t_t.remove('QualityRatioTotal')
    if split_QType:
        gr = data.groupby(by='QTotalCalcType')
        t_t.remove('QTotalCalcType')
        X_1 = gr.get_group('По ставке')
        X_2 = gr.get_group('По выработке')
        train_data.append(X_1[t_t])
        train_target.append(X_1['QualityRatioTotal'])
        train_data.append(X_2[t_t])
        train_target.append(X_2['QualityRatioTotal'])
    else:
        train_data.append(data[t_t])
        train_target.append(data['QualityRatioTotal'])
    return train_data, train_target, work_titles


def test_RandomForest(estimators=[4, 10, 20, 50, 100], title='RandomForest'):
    train_data, train_target, work_titles = load_data_v2(transform_category='LabelsEncode')
    print(train_data.shape)
    # KFold for splitting
    cv = KFold(n_splits=5,
               shuffle=True,
               random_state=241)

    # Build model:
    quality = []
    time = []
    # estimators = [2000]
    for est in estimators:
        start = timer()
        rf = RandomForestClassifier(n_estimators=est, random_state=241)
        scores = cross_val_score(rf, train_data, train_target['QualityRatioTotal'],
                                 cv=cv, scoring='roc_auc',
                                 n_jobs=-1)
        score = np.mean(scores)
        quality.append(score)
        print("Estimators number is " + str(est))
        print("Score is ", score)
        tt = timer() - start
        time.append(tt)
        print("Time elapsed: ", tt)
        print("""-----------¯\_(ツ)_/¯ -----------""")

    # Draw it:
    score_best = max(quality)
    idx = quality.index(score_best)
    est_best = estimators[idx]
    time_best = time[idx]
    print("Наилучший результат достигается при n_est=" + str(est_best))
    print("Score is ", score_best)
    print("Time elapsed: ", time_best)
    # Draw and save plot
    sns.set(font_scale=2)
    # TODO change all plotting to SEABORN
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure("Random Forest: ", figsize=(16, 12))
    fig.suptitle('RF', fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_title(str(work_titles), fontdict={'fontsize': 10})
    ax.set_ylabel('Score')
    ax.set_xlabel('N estimators')
    ax.plot(estimators, quality, 'g', linewidth=2)
    ax.grid(True)
    if not os.path.exists('../data/plots/Models/RandomForest/'):
        os.makedirs('../data/plots/Models/RandomForest/')
    plt.savefig('../data/plots/Models/RandomForest/RF_' + title + '_' + datetime.now().strftime('%m%d_%H%M') + '.png')


# ----------------------------------------- Features fillNA ----------------------------------------------------
def features_fillna_v2(data, fillna=True):
    """Special function for handling missing data"""
    # Transform Birthday column to DataTime format:
    data['Дата рождения'] = pd.to_datetime(data['Дата рождения'], errors='coerce')
    # TODO add conversion from age column to integer - is it necessary??
    age_mask = (data['Возраст'].isnull()) & (data['Дата рождения'].notnull())
    tqdm.pandas(desc="Work age  ")
    data['Возраст'][age_mask] = data[age_mask].progress_apply(fix_age, axis=1)
    # FillNA for Attraction in work: 1 if some text was typed and 0 otherwise.
    # TODO add bag of words - is it necessary??
    if fillna:
        # --------------------------------- FillNA ---------------------------------
        # FillNA for Patronymic
        # TODO add cleaning from rubbish such as '---' and '0---'
        data['Отчество'].fillna('Не указано', inplace=True)
        data['E-mail'].fillna('Не указано', inplace=True)
        data['Мобильный телефон'].fillna(0, inplace=True)
        data['Субъект федерации'].fillna('Не указано', inplace=True)
    return data.dropna()


def load_data_v2(transform_category='', drop='', fillna=True):
    """Loading data from steady CSV-files"""
    # Loading from steady-files:
    types = {'ID (автономер в базе)': np.int64, 'QTotal': np.float64}
    def_drop = ['Сотрудник', 'Субъект федерации']
    if not drop == '':
        def_drop.extend(drop)
    titles = ['ID (автономер в базе)', 'Сотрудник', 'Фамилия', 'Имя', 'Отчество', 'Пол', 'Город', 'Дата рождения',
              'Возраст', 'Субъект федерации', 'E-mail', 'Гражданство', 'Алкоголь', 'Аллергические реакции',
              'Мобильный телефон']
    # 'Мобильный телефон',]
    # ['ID (автономер в базе)', 'Сотрудник', 'Фамилия', 'Имя', 'Отчество', 'Пол', 'Город', 'Дата рождения', 'Возраст',
    # 'Мобильный телефон', 'Серия паспорта', 'Номер паспорта', 'Рабочий статус сотрудника (нов)',
    # 'Уровень квалификации', 'Квалификация', 'Город.1', 'Страна', 'Субъект федерации', 'E-mail',
    # 'Адрес прописки совпадает с адресом проживания', 'Алкоголь', 'Аллергические реакции', 'Ближ.метро', 'Выдан',
    # 'Гражданство', 'Дата выдачи', 'Дети', 'Есть основная работа', 'Источник', 'Курение',
    # 'Опыт проведения инвентаризаций', 'Предпочтительны смены', 'Причина состояния', 'ПСС', 'ПСС Сдан (Да/Нет)',
    # 'Семейное положение', 'Состояние', 'Хронические заболевания', 'QualityRatioTotal']
    data_features = pd.read_csv('../data/FULL/Features.csv', delimiter=';')[titles]
    # print(data_features.info())
    print("Features: ", data_features.shape)
    # missing_data(data_features)
    data_target = binarize_target(pd.read_csv('../data/FULL/Targets.csv', delimiter=';', dtype=types))

    # print(data_target.info())
    print("Target: ", data_target.shape)
    # Merge 2 parts of data to one DataFrame
    data = data_features.merge(data_target,
                               on='ID (автономер в базе)')
    print(list(data))
    print("Merged: ", data.shape)
    # missing_data(data, plot=True)
    data = features_fillna_v2(data, fillna=fillna)
    print("FillNA: ", data.shape)
    #
    # Munging data
    data = add_features(data)
    # drop_titles = ['ID (автономер в базе)', 'Имя', 'Отчество', 'Фамилия', 'Дата рождения']
    drop_titles = ['ID (автономер в базе)', 'Фамилия', 'Дата рождения']
    data.drop(drop_titles, axis=1, inplace=True)
    # ---- NAMES ----
    data = modify_names(data)
    # ---- Email ----
    tqdm.pandas(desc="Work with email  ")
    data['E-mail'] = data['E-mail'].progress_apply(email)
    # ---- Add MOBILE ----
    tqdm.pandas(desc="Work with MOBILE       ")
    data['Mobile'] = data['Мобильный телефон'].progress_apply(mobile)
    data.drop('Мобильный телефон', axis=1, inplace=True)
    #
    # Drop required columns:
    if not def_drop == '':
        data.drop(def_drop, axis=1, inplace=True)
    print(data)
    # ---- Categorical ----
    categorical_titles = list(data.select_dtypes(exclude=[np.number]))
    print(categorical_titles)
    work_titles = list(data)
    if transform_category in ['OneHot', 'LabelsEncode']:
        data = category_encode(data, titles=categorical_titles, mode=transform_category)
    t_t = list(data)
    t_t.remove('QualityRatioTotal')
    X = data[t_t]
    Y = data[list(data_target)[1:]]
    return X, Y, work_titles


# ----------------------------------------- Test Logistic Regression  -------------------------------------
def test_logistic_v2(drop='', scoring='roc_auc', title='', selectK='', fillna=True):
    """Testing linear method for train"""
    train_data, train_target, work_titles = load_data_v2(transform_category='OneHot', drop=drop, fillna=fillna)
    # if not selectK == '':
    #     print(train_data.shape)
    #     train_data_new = SelectKBest(chi2, k=selectK).fit_transform(train_data, train_target)
    #     print(train_data_new.shape)
    #     scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    #     rescaledData = pd.DataFrame(scaler.fit_transform(train_data_new),
    #                                 index=train_data.index)
    # else:
    #     scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    #     rescaledData = pd.DataFrame(scaler.fit_transform(train_data.values),
    #                                 index=train_data.index,
    #                                 columns=train_data.columns)
    # KFold for splitting
    cv = KFold(n_splits=5,
               shuffle=True,
               random_state=241)

    quality = []
    time = []
    _range = np.arange(-4, 2)
    C = np.power(10.0, _range)
    for c in C:
        start = timer()
        lr = linear_model.LogisticRegression(C=c,
                                             random_state=241,
                                             n_jobs=-1)
        scores = cross_val_score(lr, train_data, train_target['QualityRatioTotal'],
                                 cv=cv, scoring=scoring,
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
    print("Score is ", score_best, ' with scoring=', scoring)
    print("Time elapsed: ", time_best)
    # Draw and save plot
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure("Logistic regression: ", figsize=(16, 12))
    fig.suptitle('Logistic regression  ' + str(score_best), fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_title(str(work_titles), fontdict={'fontsize': 10})
    ax.set_ylabel('Score')
    ax.set_xlabel('log(C), ' + scoring)
    ax.plot(_range, quality, 'g', linewidth=2)
    ax.grid(True)
    if not os.path.exists('../data/plots/Models/LogisticRegression/'):
        os.makedirs('../data/plots/Models/LogisticRegression/')
    plt.savefig('../data/plots/Models/LogisticRegression/' + title + datetime.now().strftime('%m%d_%H%M') + '.png')
