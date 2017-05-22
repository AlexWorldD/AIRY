# AIRY v.0.01

import numpy as np
import pandas as pd
import os
from tqdm import tqdm, tqdm_pandas
from custom_functions import *
from models import *
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from timeit import default_timer as timer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from tqdm import tnrange, tqdm_notebook

# As CONST_VAR for linking features and target
ID = 'ID (автономер в базе)'


def modify_prediction(t, min=0.4, max=0.5):
    m1 = t > max
    m2 = t < min
    m3 = (t <= max) & (t >= min)
    t[m1] = 1
    t[m2] = 0
    t[m3] = 0.5
    return t


def proba(t, a=0.6):
    m = t > a
    m2 = t <= a
    t[m] = 1
    t[m2] = 0
    return t


def acc(y, p):
    print(p)
    print(y)
    res = (p == 1).astype(np.int16)
    res2 = (y == 1).astype(np.int16)
    return np.sum(res) / np.sum(res2)
    # return np.sum(res)/res.size


if __name__ == '__main__':
    start = timer()
    # update_csv(use=['A', 'B', 'C'])
    # TODO DROP the ID-column
    # df = pd.DataFrame({'A': np.random.rand(2) - 1, 'B': np.random.rand(2)}, index=['val1', 'val2'])
    # ax = df.plot(kind='bar', color=['r', 'b'])
    # x_offset = -0.03
    # y_offset = 0.02
    # for p in ax.patches:
    #     b = p.get_bbox()
    #     val = "{:+.2f}".format(b.y1 + b.y0)
    #     ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset))
    # plt.show()
    # test_zero()
    # train_data, train_target, w_t = load_data_v2(f='FeaturesBIN2', t='Targets2016', all=True)
    # tmp=train_data.groupby(by='Mobile').size()
    # tmp.sort_values(ascending=False, inplace=True)
    # print(tmp)
    # print(train_data)
    # print(train_data)
    # data_analysis(transform_category='LabelsEncode')
    # test_logistic_v2(selectK='', fea='Email', title='64bit', t='Targets2016', drop=['Субъект федерации'])

    # test_LR(scoring='f1', title='f1')
    titles = ['E-mail', 'Гражданство',
              'Mobile', 'Zodiac', 'DayOfBirth', 'MonthOfBirth', 'DayOfWeek', 'Имя', 'Отчество', 'Город']

    # Cutting patronymic:
    # res = []
    # for name in trange(1, 6, desc='Name'):
    #     for pat in trange(2, 10, 2, desc='Patronymic'):
    #         res.append(Neural_v2(fea='ToWord', title='Complex', C=1, cut=[name, pat], hidden=(100,)))
    # best = max(res)
    # idx = res.index(best)
    # np.save('Results/Name_Patr_100.npy', res)
    # print(idx)
    # print('Name: ', int(idx)//len(range(2, 10, 2))+1)
    # print('Patronymic: ', int(idx) % len(range(2, 10, 2)))
    # print(max(res))
    # res = np.load('Results/Name_Patr_100.npy')
    # print(res)

    print(Neural_v2(fea='ToWord', title='bestNamePat', C=1, cut=[2, 2], hidden=(50, 50), plot_auc=True, make_pretty=3000))
    # Neural_v2(fea='ToWord', title='Complex', C=1, cut=[4,8], hidden=(50,50), save=True)
    # LR_v2(title='NewVersionBEST', cut=[4,8], selectK='best', save=True)
    # find_alpha()
    # find_bestNeural(fea='GRID',  cut=[2, 2], title='alpha_1_')
    # find_bestLR(fea='GRID', cut=True, drop=['Субъект федерации'], title='Full_withKSelect', selectK=550)
    # find_alpha()
    # test_LR(title='LR_C', selectK=540)
    # LR_v2(fea='ToWord', title='l2_norm', selectK='best', C=10, plot_auc=True, plot_pr=True, cut=[10,4])
    # predicted1, y1 = LR_v2(title='NewVersionBEST', cut=True, selectK='best', no_plot=True, final=True)
    # predicted1 = modify_prediction(predicted1)
    # y1 = np.array(y1)
    # print(acc(y1, modify_prediction(predicted1)))
    # predicted2, y2 = RF(selectK='best', title='5kTrees', no_plot=True)
    # np.save('Results/predicted1.npy', predicted1)
    # np.save('Results/predicted2.npy', predicted2)
    # np.save('Results/y1.npy', y1)
    # np.save('Results/y2.npy', y2)
    # t_t = list(train_data_new)
    # t_t.remove('QualityRatioTotal')
    # grouped = train_data_new.groupby(t_t, as_index=False)
    # data_target = grouped.agg({'QualityRatioTotal': np.sum})
    # data_target['QualityRatioTotal'] = data_target['QualityRatioTotal'].apply(np.sign)
    # data_target = data_target.drop(data_target[data_target['QualityRatioTotal'] == 0].index)
    # data_target.reset_index(level=0, inplace=True)
    # plot_results(fea='Zodiac', cv='SVC')
    print('Elapsed time:', timer() - start)
