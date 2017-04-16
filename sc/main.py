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
    print(train_data)
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
