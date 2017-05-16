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
    data_analysis()
    # print_bar(train_data, tmp='Имя', head=50)
    # test_RandomForest()
    # test_logistic_v2(selectK=250, fea='V3', title='fixTarget', t='Targets2016', drop=['Субъект федерации'])


    print('Elapsed time:', timer() - start)
