import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import argparse
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, auc
from sklearn.feature_selection import RFECV
import mlflow
import shap
import json

from tqdm import tqdm

fdir_raw = Path("data/raw/")
fdir_processed = Path("data/interim")
fdir_traintest = Path("data/processed") / 'sex'
fdir_external = Path("data/external")
ml_models_fdir = Path("models")


use_CV = True

model_type = 'catboost'
model_type = 'xgboost'

# sex = 'chrXY'
# sex = 'autosome'

feature_importance_method = 'native'
feature_importance_method = 'SHAP'

n_threads = 6

value_to_predict = 'Sex'
# value_to_predict = 'Experimental_Factor:_population (exp)'

organ = ["BLOOD1", 'BRAIN0', "HEART", "BRAIN1", 'None'][1]

for sex_chromosome in ['chrXY', 'autosome', 'chrX', 'chrY']:
    # for sex_chromosome in ['chrXY']:

    with open(f'models/{model_type}.json', 'r') as file:
        model_params = json.load(file)

    # print(model_params)
    data = pd.read_hdf(fdir_traintest / f'geuvadis.preprocessed.sex.h5', key=sex_chromosome)
    data_header = pd.read_hdf(fdir_processed / 'geuvadis.preprocessed.h5', key="header")

    feature_importance_df = pd.read_hdf(fdir_processed / f'feature_importance.{model_type}.{value_to_predict}.h5',
                                        key=f'{sex_chromosome}',)

    features = feature_importance_df[feature_importance_method]

    if organ != "None":
        fname = next((fdir_external / organ / 'reg').glob("*processed.h5"))
        fname = fname.name

        data_eval = pd.read_hdf(fdir_external / organ / 'reg' / fname, index_col=0)
        features = features.loc[features.index.intersection(data_eval.columns)]

    features = features.sort_values(ascending=False).index

    roc_array_total = {}
    accuracy_array_total = {}
    f1_array_total = {}
    precision_array_total = {}
    recall_array_total = {}

    max_n_features = 100

    for i in tqdm(range(1, max_n_features)):
        n_features = i

        data_shrinked = data[features[:n_features]]
        data_header_shrinked = data_header.loc[data.index]

        X = data_shrinked.values
        y = data_header_shrinked['Sex']

        label_encoder = LabelEncoder().fit(y)
        y = label_encoder.transform(y)

        roc_array = []

        accuracy_array = []
        f1_array = []
        precision_array = []
        recall_array = []

        # cv = StratifiedKFold(n_splits=5)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=test_size, random_state=random_state, shuffle=True)

        for train, val in cv.split(X, y):
            X_train = X[train]
            y_train = y[train]
            X_test = X[val]
            y_test = y[val]

            # test_size = 0.2
            # random_state = 42

            train_scaler = StandardScaler().fit(X_train)
            test_scaler = StandardScaler().fit(X_test)

            X_train = train_scaler.transform(X_train)
            X_test = test_scaler.transform(X_test)

            # label_encoder = LabelEncoder().fit(y_train)
            # y_train = label_encoder.transform(y_train)
            # y_test = label_encoder.transform(y_test)

            # X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train)

            # X_train_ = X_train[train]
            # y_train_ = y_train[train]
            # X_val = X_train[val]
            # y_val = y_train[val]

            X_train_ = X_train
            y_train_ = y_train
            X_val = X_test
            y_val = y_test

            if model_type == 'xgboost':
                model = xgb.XGBClassifier(**model_params)
                model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], verbose=False)

            if model_type == 'catboost':
                model = CatBoostClassifier(**model_params)
                model.fit(X_train_, y_train_,
                          eval_set=(X_val, y_val),
                          verbose=False,
                          use_best_model=True,
                          plot=False,
                          early_stopping_rounds=20)

            # if not (ml_models_fdir / 'xgboost').is_dir():
            #     (ml_models_fdir / 'xgboost').mkdir()

            # saved_model_filename = f"geuvadis_{sex}.json"
            # model.save_model(fname=ml_models_fdir / 'xgboost' / saved_model_filename)

            roc_array.append(roc_auc_score(y_test, model.predict(X_test)))

            accuracy_array.append(accuracy_score(y_test, model.predict(X_test)))
            f1_array.append(f1_score(y_test, model.predict(X_test)))
            precision_array.append(precision_score(y_test, model.predict(X_test)))
            recall_array.append(recall_score(y_test, model.predict(X_test)))

        roc_array_total[i] = roc_array

        accuracy_array_total[i] = accuracy_array
        f1_array_total[i] = f1_array
        precision_array_total[i] = precision_array
        recall_array_total[i] = recall_array

        # plt.figure()
        # plt.plot(np.arange(1, len(roc_array) + 1), roc_array)
        # plt.show()

    roc_array_df = pd.DataFrame.from_dict(roc_array_total)
    accuracy_array_df = pd.DataFrame.from_dict(accuracy_array_total)
    f1_array_df = pd.DataFrame.from_dict(f1_array_total)
    precision_array_df = pd.DataFrame.from_dict(precision_array_total)
    recall_array_df = pd.DataFrame.from_dict(recall_array_total)

    plt.figure()
    plt.errorbar(np.arange(1, max_n_features), roc_array_df.mean(), yerr=roc_array_df.std(), label='roc auc')
    plt.errorbar(np.arange(1, max_n_features), accuracy_array_df.mean(), yerr=accuracy_array_df.std(), label='accuracy')
    plt.errorbar(np.arange(1, max_n_features), f1_array_df.mean(), yerr=f1_array_df.std(), label='f1')
    plt.errorbar(np.arange(1, max_n_features), precision_array_df.mean(), yerr=precision_array_df.std(), label='precision')
    plt.errorbar(np.arange(1, max_n_features), recall_array_df.mean(), yerr=recall_array_df.std(), label='recall')
    plt.title(sex_chromosome + ", organ: " + organ)
    plt.show()
