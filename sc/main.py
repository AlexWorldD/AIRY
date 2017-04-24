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
    # update_csv(use=['A', 'B', 'C'])
    # TODO DROP the ID-column
    # train_data, train_target = load_data_bin()
    # train_data = get_mobile(train_data, mode='Numbers')
    # print(train_data['Mobile'])
    # mobiles = train_data.groupby(by='Mobile').size()
    # mobiles.sort_values(ascending=False, inplace=True)
    # print(mobiles)

    # tqdm.pandas(desc="Work with NAME.v2  ")
    # # train_data[tmp] = train_data[tmp].progress_apply(email)
    # train_data'Имя'] = train_data['Имя'].progress_apply(lambda t: t.lower())
    # mails = train_data.groupby(by=tmp).size()
    # mails.sort_values(ascending=False, inplace=True)
    # print(list(pd.DataFrame(mails[mails>10]).T))
    # print_bar(train_data, tmp=tmp, filna=True)

    # print(rescaledData)
    # LR()

    # KFold for splitting
    cv = KFold(n_splits=5,
               shuffle=True,
               random_state=241)
    # neural()
    # test_logistic(title='MobileOperator')
    # data = load_features(forceAll=True)
    # counts = data.describe(include='all').loc[:'count'].T.sort_values(by='count', ascending=False)
    # plt.show(counts.head(25).plot.bar())
    # print(counts.head(25))

    # data_features = pd.read_csv('../data/tmp/F13.csv', encoding='cp1251',
    #                             index_col=0)
    # print("Features: ", data_features.shape)
    # data_target = binarize_target(pd.read_csv('../data/tmp/T13.csv', encoding='cp1251',
    #                                           index_col=0), with_type=True)
    # print("Target: ", data_target.shape)
    # # Merge 2 parts of data to one DataFrame
    # data = data_features.merge(data_target,
    #                            on='ID (автономер в базе)')
    # print("Merged: ", data.shape)
    # data = features_fillna(data)
    # print("FillNA: ", data.shape)
    #
    # train_data = data[list(data_features)]
    # train_target = data['QualityRatioTotal']
    # gr = data.groupby(by='QTotalCalcType').get_group('По выработке')
    # print(gr)
    # print(train_data, train_target)
    # train_data, train_target, work_titles = load_dataset(split_QType=False, save_categorical=True)
    # train_data, train_target, work_titles = load_data(transform_category=False)
    # print(list(train_data))
    test_logistic(title='TestTT', scoring='accuracy')
    # LR()
    # print(train_target.groupby(by='QualityRatioTotal').size())
    # test_neural(train_data[0], train_target[0], work_titles, neural_size=(10, 10), title='10-10')
    # t_d, t_t = load_data_from_file(use=['A', 'B', 'C'])
    # print(t_d)
    # test_logistic(train_data[1], train_target[1], work_titles, title='WithQTYPESplit_2_')
    print('Elapsed time:', timer() - start)
