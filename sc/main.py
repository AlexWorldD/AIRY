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
import matplotlib.pyplot as plt

# As CONST_VAR for linking features and target
ID = 'ID (автономер в базе)'


# TODO change indexing from name to idx.
def quality_ratio(row):
    """Special function for calculating QualityRatio for staff"""
    if row['QTotalCalcType'] == 'По ставке':
        if row['Явка на смене (Смена)'] == 'Да':
            return 1
        elif row['Статус смены (Смена)'] == 'Подтвержден':
            return 0
        else:
            return 0.5
    elif row['QTotalCalcType'] == 'По выработке':
        if row['Выработка % от нормы по ручному пересчету (QSP)'] >= 0.85:
            return 1
        elif row['Выработка % от нормы по ручному пересчету (QSP)'] < 0.5:
            return 0
        else:
            return 0.5


if __name__ == '__main__':
    start = timer()
    # Setting required priorities for features
    priorities = ['Важный',
                  'Средняя']
    # TODO DROP the ID-column
    train_data, train_target = load_data()
    # 'Явка на смене (Смена)', 'Востребована оплата по смене', 'Выработка % от нормы по сканированию (Qscan)',
    # 'Выработка % от нормы по ручному пересчету (QSP)', 'QTotalCalcType', 'QTotal', 'Ошибок сканирования (штук)',
    # 'Статус смены (Смена)'
    tqdm.pandas(desc="Calculate QualityRatio for staff")
    train_target['Quality Ratio'] = train_target.progress_apply(quality_ratio, axis=1)
    print(train_target)
    print('Elapsed time:', timer() - start)
