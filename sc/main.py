# AIRY v.0.01

import numpy as np
import pandas as pd
import os
from tqdm import tqdm, tqdm_pandas
from custom_functions import *
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
    # train_data, train_target = load_data_bin()

    # train_data = vectorize(train_data)
    # drop_titles = ['ID (автономер в базе)', 'Фамилия', 'Дата рождения']
    # train_data.drop(drop_titles, axis=1, inplace=True)
    # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # rescaledData = pd.DataFrame(scaler.fit_transform(train_data.values),
    #                             index=train_data.index,
    #                             columns=train_data.columns)
    # scaler2 = preprocessing.StandardScaler()
    # rescaledData = pd.DataFrame(scaler2.fit_transform(rescaledData.values),
    #                             index=rescaledData.index,
    #                             columns=rescaledData.columns)
    # print(rescaledData)
    # LR()
    # KFold for splitting
    cv = KFold(n_splits=5,
               shuffle=True,
               random_state=241)
    test_logistic()


    print('Elapsed time:', timer() - start)
