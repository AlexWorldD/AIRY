# AIRY v.0.01

import numpy as np
import pandas as pd
import os
from tqdm import tqdm, tqdm_pandas
from loading_data import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from timeit import default_timer as timer
from sklearn.feature_extraction import DictVectorizer

# As CONST_VAR for linking features and target
ID = 'ID (автономер в базе)'

if __name__ == '__main__':
    start = timer()
    # Setting required priorities for features
    priorities = ['Важный',
                  'Средняя']
    # update_csv()
    # TODO DROP the ID-column
    train_data, train_target = load_data()
    # KFold for splitting
    cv = KFold(n_splits=5,
               shuffle=True,
               random_state=241)

    drop_titles = ['ID (автономер в базе)', 'Фамилия', 'Дата рождения']
    train_data.drop(drop_titles, axis=1, inplace=True)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    rescaledData = pd.DataFrame(scaler.fit_transform(train_data.values),
                                index=train_data.index,
                                columns=train_data.columns)

    # lr = linear_model.Ridge(random_state=42, alpha=0.1)
    # lrCV = linear_model.RidgeCV(cv=cv)
    # lrCV.fit(rescaledData, train_target['QualityRatioTotal'])

    quality = []
    time = []
    C = np.power(10.0, np.arange(-3, 2))
    for c in C:
        start1 = timer()
        lr = linear_model.Ridge(random_state=42, alpha=c)
        scores = cross_val_score(lr, rescaledData, train_target['QualityRatioTotal'],
                                 cv=cv, n_jobs=-1)
        score = np.mean(scores)
        quality.append(score)
        tt = timer() - start1
        time.append(tt)
        print("C parameter is " + str(c))
        print("Score is ", score)
        print("Time elapsed: ", tt)
        print("""-----------¯\_(ツ)_/¯ -----------""")

            # Find optimal
    # lr.fit(rescaledData, train_target['QualityRatioTotal'])
    # score = cross_val_score(lr, train_data, train_target['QualityRatioTotal'],
    #                         cv=cv, n_jobs=-1)
    # Build model


    # tqdm.pandas(desc="Work with names: ")
    # train_data['Имя'] = train_data['Имя'].progress_apply(lambda t: t.lower())
    # enc = DictVectorizer()
    # dummies = pd.get_dummies(train_data, columns=['Имя', 'Отчество', 'Пол', 'Дети', 'Семейное положение'])
    # print(dummies.dropna())

    # Convert BD to datetime format
    # 'Явка на смене (Смена)', 'Востребована оплата по смене', 'Выработка % от нормы по сканированию (Qscan)',
    # 'Выработка % от нормы по ручному пересчету (QSP)', 'QTotalCalcType', 'QTotal', 'Ошибок сканирования (штук)',
    # 'Статус смены (Смена)'

    # data_target = pd.read_csv('../data/tmp/T13.csv', encoding='cp1251',
    #                           index_col=0)
    # print(list(data_target))
    # missing_data(data_target)

    # train_data = pd.read_csv('../data/tmp/F13.csv', encoding='cp1251',
    #                            index_col=0)



    print('Elapsed time:', timer() - start)
