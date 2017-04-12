# AIRY v.0.01

import numpy as np
import pandas as pd
import os
from tqdm import tqdm, tqdm_pandas
from loading_data import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from timeit import default_timer as timer

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

    # Convert BD to datetime format
    # train_data['Дата рождения'] = pd.to_datetime(train_data['Дата рождения'])
    # missing_data(train_data)
    # age_mask = (train_data['Возраст'].isnull()) & (train_data['Дата рождения'].notnull())
    #
    # train_data['Возраст'][age_mask] = train_data[age_mask].apply(fix_age, axis=1)
    #
    # print(train_data[(train_data['Возраст'].notnull()) & (train_data['Дата рождения'].isnull())])
    # missing_data(train_target)
    # tt = load_features(forceAll=True)
    # missing_data(tt)
    # 'Явка на смене (Смена)', 'Востребована оплата по смене', 'Выработка % от нормы по сканированию (Qscan)',
    # 'Выработка % от нормы по ручному пересчету (QSP)', 'QTotalCalcType', 'QTotal', 'Ошибок сканирования (штук)',
    # 'Статус смены (Смена)'

    # data_target = pd.read_csv('../data/tmp/T13.csv', encoding='cp1251',
    #                           index_col=0)
    # print(list(data_target))
    # missing_data(data_target)

    # train_data = pd.read_csv('../data/tmp/F13.csv', encoding='cp1251',
    #                            index_col=0)
    missing_data(train_data)

    print('Elapsed time:', timer() - start)
