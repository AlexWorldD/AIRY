from custom_functions import *
from sklearn import svm
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


# ----------------------------------------- Test Models via Data Modifications  -------------------------------------
def load_data_v3(transform_category='OneHot', t='Targets2016', f='FeaturesBIN3', drop='',
                 fillna=True,
                 add_f=True,
                 name_modification=True,
                 emails=True,
                 mob=True,
                 all=False, use_private=False, clean_target=False, required_titles='', no_split=False):
    """Loading data from steady CSV-files"""
    # Loading from steady-files:
    types = {'ID (автономер в базе)': np.int64, 'QTotal': np.float64, 'BIN': np.int64, 'Мобильный телефон': np.int64}
    def_drop = ['Фамилия']
    use = ['Пол', 'Возраст', 'QualityRatioTotal']
    if not required_titles == '':
        use.extend(required_titles)
    base = ['ID (автономер в базе)', 'Пол', 'Дата рождения', 'Возраст', 'QualityRatioTotal']
    if not drop == '':
        def_drop.extend(drop)
    titles = ['ID (автономер в базе)', 'Фамилия', 'Имя', 'Отчество', 'Пол', 'Город', 'Дата рождения',
              'Возраст', 'Субъект федерации', 'E-mail', 'Гражданство',
              'Мобильный телефон']
    t_titles = ['ID (автономер в базе)', 'QTotal', 'QTotalCalcType', 'Статус смены (Смена)', 'Явка на смене (Смена)',
                'Тип биллинга']
    private = ['Серия паспорта', 'Номер паспорта', 'Дата выдачи', 'BIN']
    if use_private:
        titles.extend(private)

    if all:
        fillna = True
        add_f = True
        name_modification = True
        emails = True
        mob = True

    data_features = pd.read_csv('../data/FULL/' + f + '.csv', delimiter=';', dtype=types)[titles]
    # print(data_features.info())
    print("Features: ", data_features.shape)
    data_target = binarize_target(pd.read_csv('../data/FULL/' + t + '.csv', delimiter=';', dtype=types)[t_titles])

    if clean_target:
        grouped = data_target.groupby(by='ID (автономер в базе)')
        data_target = grouped.aggregate(np.sum)
        data_target['Size'] = data_target['QualityRatioTotal'].apply(np.absolute)
        data_target['QualityRatioTotal'] = data_target['QualityRatioTotal'].apply(np.sign)
        data_target = data_target.drop(data_target[data_target['QualityRatioTotal'] == 0].index)
        base.append('Size')
        data_target.reset_index(level=0, inplace=True)
    # print(data_target.info())
    print("Target: ", data_target.shape)
    # Merge 2 parts of data to one DataFrame
    data = data_features.merge(data_target,
                               on='ID (автономер в базе)')
    print("Merged: ", data.shape)

    # ----------------------------------- Munging Data ------------------------
    data = features_fillna_v2(data, fillna=fillna)
    # Without Email:
    print("FillNA: ", data.shape)

    if add_f:
        data = add_features(data)
        # split_bd = vectorize(add_features(data[base], zodiac_sign=False))
        # q, t= test_models(split_bd)
        # zodiac_s = vectorize(add_features(data[base]))
        # q2, t2 = test_models(zodiac_s)
        # range = np.arange(-4, 2)
        # plot_results(q, q2, range, fea='Zodiac', cv='LR')

    # ---- NAMES ----
    # print(data)
    # if name_modification:
    # data = modify_names(data)
    # ---- Email ----
    if emails:
        data = get_email(data)
    # ---- Add MOBILE ----
    if mob:
        data = get_mobile(data, mode='Operator')
    # data.to_excel('FinalDataSet.xlsx')
    # Drop required columns:
    if not def_drop == '':
        data.drop(def_drop, axis=1, inplace=True)
    if not required_titles == '':
        data = data[use]
        print('Final shape:', data.shape)
    # Transofrm category
    data = vectorize(data, mode=transform_category, thin_data=False)
    if no_split:
        return data
    t_t = list(data)
    t_t.remove('QualityRatioTotal')
    X = data[t_t]
    # Y = data[list(data_target)[1:2]]
    Y = data['QualityRatioTotal']
    # data_stat(X)
    return X, Y


def split_data(data):
    data.drop('ID (автономер в базе)', axis=1, inplace=True)
    t_t = list(data)
    t_t.remove('QualityRatioTotal')
    X = data[t_t]
    # Y = data[list(data_target)[1:2]]
    Y = data['QualityRatioTotal']
    return X, Y


def prepare_test_set(data):
    t_t = list(data)
    t_t.remove('QualityRatioTotal')
    grouped = data.groupby(t_t, as_index=False)
    data_target = grouped.agg({'QualityRatioTotal': np.sum})
    data_target['QualityRatioTotal'] = data_target['QualityRatioTotal'].apply(np.sign)
    return data_target
# ----------------------------------------- Test Logistic Regression  -------------------------------------
def test_LR(drop='', fea='', scoring='roc_auc', title='', selectK='', fillna=True, f='FeaturesBIN3',
            t='Targets2016'):
    """Testing linear method for train"""
    train_data, train_target = load_data_v3(transform_category='OneHot', t=t, f=f, drop=drop, all=True)
    fnd = False
    if selectK == 'best':
        fnd = True
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        train_data = pd.DataFrame(scaler.fit_transform(train_data),
                                  index=train_data.index)
        print('Features from model..')
    elif not selectK == '':
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        train_data = pd.DataFrame(scaler.fit_transform(train_data),
                                  index=train_data.index)
        print(train_data.shape)
        train_data_new = SelectKBest(chi2, k=selectK).fit_transform(train_data, train_target)
        print(train_data_new.shape)
    else:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        train_data_new = pd.DataFrame(scaler.fit_transform(train_data.values),
                                      index=train_data.index,
                                      columns=train_data.columns)
    print("Scaling complete!")
    # KFold for splitting
    cv = KFold(n_splits=10,
               shuffle=True,
               random_state=241)

    quality = []
    time = []
    _range = np.arange(-4, 2)
    C = np.power(10.0, _range)
    for c in C:
        start = timer()
        lr = LogisticRegression(C=c,
                                random_state=241,
                                n_jobs=-1)
        if fnd:
            print(train_data.shape)
            train_data_new = SelectFromModel(lr).fit_transform(train_data, train_target)
            print(train_data_new.shape)
        scores = cross_val_score(lr, train_data_new, train_target,
                                 cv=cv, scoring=scoring,
                                 n_jobs=-1)
        score = np.mean(scores)
        quality.append(score)
        tt = timer() - start
        time.append(tt)
        print("C parameter is " + str(c))
        print("Score is ", score)
        print("Time elapsed: ", tt)
        print("""-----------¯\_(ツ)_/¯ -----------""")

        # Draw it:
    score_best = max(quality)
    idx = quality.index(score_best)
    C_best = C[idx]
    time_best = time[idx]
    print("Наилучший результат достигается при C=" + str(C_best))
    print("Score is ", score_best, ' with scoring=', scoring)
    print("Time elapsed: ", time_best)
    # Draw and save plot
    sns.set(font_scale=2)
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure("Logistic regression: ", figsize=(16, 12))
    fig.suptitle('Logistic regression  ' + str(score_best), fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_ylabel('AUC mean', fontsize=20, labelpad=10)
    ax.set_xlabel('log(C)', labelpad=10)
    ax.plot(_range, quality, 'b', linewidth=2)
    ax.grid(True)
    if not os.path.exists('../data/Results/LogisticRegression/' + fea + '/'):
        os.makedirs('../data/Results/LogisticRegression/' + fea + '/')
    plt.savefig('../data/Results/LogisticRegression/' + fea + '/' + title + '_' + datetime.now().strftime(
        '%d_%H%M') + '.png')


# ----------------------------------------- Test Logistic Regression  -------------------------------------
def LR(drop='', C=1, fea='', scoring='roc_auc', title='', selectK='', fillna=True, f='FeaturesBIN3',
       t='Targets2016', required=''):
    """Testing linear method for train"""
    train_data, train_target = load_data_v3(transform_category='OneHot', t=t, f=f, drop=drop, all=True,
                                            required_titles=required)
    # shuffle and split training and test sets

    fnd = False
    if selectK == 'best':
        fnd = True
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        train_data = pd.DataFrame(scaler.fit_transform(train_data),
                                  index=train_data.index)
        print('Features from model..')
    elif not selectK == '':
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        train_data = pd.DataFrame(scaler.fit_transform(train_data),
                                  index=train_data.index)
        print(train_data.shape)
        train_data_new = SelectKBest(chi2, k=selectK).fit_transform(train_data, train_target)
        print(train_data_new.shape)
    else:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        train_data_new = pd.DataFrame(scaler.fit_transform(train_data.values),
                                      index=train_data.index,
                                      columns=train_data.columns)
    print("Scaling complete!")
    train_data_new, X_test, train_target, y_test = train_test_split(train_data_new, train_target, test_size=.3,
                                                                    random_state=241)
    # KFold for splitting
    cv = KFold(n_splits=10,
               shuffle=True,
               random_state=241)

    start = timer()
    lr = LogisticRegression(C=C,
                            random_state=241,
                            n_jobs=-1)
    if fnd:
        print(train_data.shape)
        train_data_new = SelectFromModel(lr).fit_transform(train_data, train_target)
        print(train_data_new.shape)
    y_predict = lr.fit(train_data_new, train_target).decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_predict)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic +' + title)
    plt.legend(loc="lower right")
    if not os.path.exists('../data/Results/LogisticRegression/' + fea + '/'):
        os.makedirs('../data/Results/LogisticRegression/' + fea + '/')
    plt.savefig('../data/Results/LogisticRegression/' + fea + '/' + title + '_' + datetime.now().strftime(
        '%d_%H%M') + '.png', bbox_inches='tight', dpi=300)
    tt = timer() - start
    return auc(fpr, tpr)

    print("C parameter is " + str(C))
    print("Score is ", scores.mean())
    print("Time elapsed: ", tt)
    print("""-----------¯\_(ツ)_/¯ -----------""")

    # Draw it:
    # Draw and save plot
    sns.set(font_scale=2)
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure("Logistic regression: ", figsize=(16, 12))
    fig.suptitle('Logistic regression  ' + str(scores.mean()), fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_ylabel('AUC mean', fontsize=20, labelpad=10)
    ax.set_xlabel('log(C)', labelpad=10)
    ax.plot(_range, quality, 'b', linewidth=2)
    ax.grid(True)
    if not os.path.exists('../data/Results/LogisticRegression/' + fea + '/'):
        os.makedirs('../data/Results/LogisticRegression/' + fea + '/')
    plt.savefig('../data/Results/LogisticRegression/' + fea + '/' + title + '_' + datetime.now().strftime(
        '%d_%H%M') + '.png')


# ----------------------------------------- Transform all categorical features to vectors  -----------------------------
def vectorize(data, mode='OneHot', thin_data=False):
    """Special function for data vectorization"""
    # ---- Categorical ----
    if thin_data:
        data = thin(data)
    categorical_titles = list(data.select_dtypes(exclude=[np.number]))
    print("Found categorical features: ", categorical_titles)
    for it in categorical_titles:
        data[it] = data[it].astype('category')
    work_titles = list(data)
    if mode in ['OneHot', 'LabelsEncode', 'Hash']:
        data = category_encode(data, titles=categorical_titles, mode=mode)
    print('Vectorized: ', data.shape)
    return data


# ----------------------------------------- Test Logistic Regression  -------------------------------------
def test_models(data, scoring='roc_auc', title='', selectK=''):
    """Testing different models"""

    t_t = list(data)
    t_t.remove('QualityRatioTotal')
    X = data[t_t]
    Y = data['QualityRatioTotal']
    fnd = False
    if selectK == 'best':
        fnd = True
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        train_data = pd.DataFrame(scaler.fit_transform(X),
                                  index=X.index)
        print('Features from model..')
    elif not selectK == '':
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        train_data = pd.DataFrame(scaler.fit_transform(X),
                                  index=X.index)
        print(train_data.shape)
        train_data_new = SelectKBest(chi2, k=selectK).fit_transform(train_data, Y)
        print('After features selection: ', train_data_new.shape)
    else:
        print('Without feature selection!')
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        train_data_new = pd.DataFrame(scaler.fit_transform(X.values),
                                      index=X.index,
                                      columns=X.columns)
    # KFold for splitting
    cv = KFold(n_splits=5,
               shuffle=True,
               random_state=241)

    quality = []
    time = []
    _range = np.arange(-4, 2)
    C = np.power(10.0, _range)
    for c in C:
        start = timer()
        lr = linear_model.LogisticRegression(C=c,
                                             random_state=241,
                                             n_jobs=-1)
        sv = svm.SVC(C=c, random_state=241)
        if fnd:
            print(train_data.shape)
            train_data_new = SelectFromModel(lr).fit_transform(train_data, Y)
            print('After features selection: ', train_data_new.shape)
        scores = cross_val_score(lr, train_data_new, Y,
                                 cv=cv, scoring=scoring,
                                 n_jobs=-1)
        score = np.mean(scores)
        quality.append(score)
        tt = timer() - start
        time.append(tt)
    # Draw it:
    score_best = max(quality)
    idx = quality.index(score_best)
    C_best = C[idx]
    time_best = time[idx]
    print("Наилучший результат достигается при C=" + str(C_best))
    print("Score is ", score_best, ' with scoring=', scoring)
    print("Time elapsed: ", time_best)
    return quality, time


def plot_results(q, q2, _range, fea, cv='Logistic'):
    # Draw and save plot
    sns.set(font_scale=2)
    # range = np.arange(-4, 2)
    # q = [0.66275214378133218, 0.66292229907915701, 0.66292089779027208, 0.66577675719793505, 0.66638800680018262, 0.66461884864827081]
    # q2 = [0.65863830491423769, 0.66417026270216351, 0.66437268238419844, 0.66582407260394583, 0.66845376454228123, 0.67134223982168684]
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(cv, figsize=(16, 12))
    fig.suptitle(cv + '  ' + str(max(q)), fontweight='bold')
    ax = fig.add_subplot(111)
    # ax.set_title(str(work_titles), fontdict={'fontsize': 10})
    ax.set_ylabel('AUC mean', fontsize=20, labelpad=10)
    ax.set_xlabel('log(C), ', labelpad=10)
    ax.plot(_range, q, 'b', linewidth=2)
    ax.plot(_range, q2, 'g', linewidth=2)
    ax.grid(True)
    if not os.path.exists('../data/plots/Models/LogisticRegression/' + fea + '/'):
        os.makedirs('../data/plots/Models/LogisticRegression/' + fea + '/')
    plt.savefig('../data/plots/Models/LogisticRegression/' + fea + '/' + cv + datetime.now().strftime(
        '%m%d_%H%M') + '.png', bbox_inches='tight', dpi=300)


def different_features():
    titles = ['E-mail', 'Гражданство',
              'Mobile', 'Zodiac', 'DayOfBirth', 'MonthOfBirth', 'DayOfWeek', 'Имя', 'Отчество', 'Город']
    required_t = []
    res = []
    for title in titles:
        print('Added: ---------' + title + '------------')
        required_t.append(title)
        res.append(LR(required=required_t, title=title, fea='Features'))

    fig, ax = plt.subplots()
    lw = 2
    plt.rcParams.update({'font.size': 22})
    plt.bar(np.arange(len(titles)), res, 0.3, color='darkorange')
    plt.ylim([0.0, 1.05])
    ax.set_xticks(np.arange(len(titles)))
    ax.set_xticklabels(titles)
    plt.xlabel('Features', fontsize=20, labelpad=10)
    plt.ylabel('AUC ROC', fontsize=20, labelpad=10)
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if not os.path.exists('../data/Results/LogisticRegression/'):
        os.makedirs('../data/Results/LogisticRegression/')
    plt.savefig('../data/Results/LogisticRegression/Comparing_' + datetime.now().strftime(
        '%d_%H%M') + '.png', bbox_inches='tight', dpi=300)
