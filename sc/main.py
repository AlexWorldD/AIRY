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

    start = timer()
    # Setting required priorities for features
    priorities = ['Важный',
                  'Средняя']
    # TODO DROP the ID-column
    train_data, train_target = load_data()
    print(train_data)
    print('Elapsed time:', timer() - start)
