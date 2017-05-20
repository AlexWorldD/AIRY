# AIRY v.0.01

import numpy as np
import pandas as pd
import os
from tqdm import tqdm, tqdm_pandas
from custom_functions import *
from models import *
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

# As CONST_VAR for linking features and target
ID = 'ID (автономер в базе)'

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
    X = load_data_v3(all=True, no_split=True)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    train_data_new = pd.DataFrame(scaler.fit_transform(X.values),
                                  index=X.index,
                                  columns=X.columns)

    train_data_new, X_test = train_test_split(X, test_size=.3,
                                              random_state=241)
    print(split_data(train_data_new))
    # t_t = list(train_data_new)
    # t_t.remove('QualityRatioTotal')
    # grouped = train_data_new.groupby(t_t, as_index=False)
    # data_target = grouped.agg({'QualityRatioTotal': np.sum})
    # data_target['QualityRatioTotal'] = data_target['QualityRatioTotal'].apply(np.sign)
    # data_target = data_target.drop(data_target[data_target['QualityRatioTotal'] == 0].index)
    # data_target.reset_index(level=0, inplace=True)
    # plot_results(fea='Zodiac', cv='SVC')
    print('Elapsed time:', timer() - start)
