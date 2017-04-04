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

if __name__ == '__main__':
    # Loading data from Excel file

    # Setting required priorities for features
    priorities = ['Важный',
                  'Средняя']

    data = load_features(priorities=priorities)
    print(data.head())
