# AIRY v.0.01

import numpy as np
import pandas as pd
import os
from loading_data import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# As CONST_VAR for linking features and target
ID = 'ID (автономер в базе)'

if __name__ == '__main__':
    # Loading data from Excel file

    start = timer()
    # Setting required priorities for features
    priorities = ['Важный',
                  'Средняя']

    # Loading from original files:
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

    print(data_features)
    # data = data_features.merge(data_target,
    #                            on='ID (автономер в базе)')
    # A quick look to target-data:
    # print(data(10))
    print('Elapsed time:', timer() - start)
