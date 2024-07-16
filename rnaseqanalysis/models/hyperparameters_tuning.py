import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, auc

import optuna
import shap


import mlflow
from mlflow.models import infer_signature

from tqdm import tqdm

mlflow.set_tracking_uri(uri="http://localhost:8080")


fdir_raw = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/raw/")
fdir_processed = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/interim")
fdir_traintest = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/processed") / 'sex'

use_CV = True

model_type = 'catboost'
model_type = 'xgboost'

sex = 'chrXY'
sex = 'autosome'

n_threads = 6
params_xgb = {
    # "early_stopping_rounds": 20,
    "n_jobs": n_threads,
    "objective": 'binary:logistic',
    "n_estimators": 500,
    'device': 'cuda',
    'eta': 0.05,
    "gamma": 1e-6,
    # 'verbosity': 0
}

params_catboost = {
    "loss_function": 'Logloss',  # MultiClass
    "od_pval": 0.05,
    "thread_count": n_threads,
    "task_type": "GPU",
    "iterations": 500,
    "learning_rate": 0.03
    #  devices='0'
}

if model_type == 'xgboost':
    model_params = params_xgb
if model_type == 'catboost':
    model_params = params_catboost

# optuna.logging.set_verbosity

n_features = 30


data = pd.read_csv(fdir_traintest / f'geuvadis.preprocessed.{sex}.csv', index_col=0)
data_header = pd.read_csv(fdir_raw / 'Geuvadis.SraRunTable.txt', index_col=0)
data_header = data_header[[
    'Sex',
]]


features = pd.read_csv(fdir_processed / f'feature_importance.{model_type}.{sex}.csv', index_col=0)
data = data[features.iloc[:n_features].index]

data_header = data_header.loc[data.index]

X = data.values
y = data_header['Sex']

label_encoder = LabelEncoder().fit(y)
y = label_encoder.transform(y)

cv = StratifiedKFold(n_splits=5)


def mlflow_callback(study: optuna.Study, trial: optuna.Trial):
    mlflow.set_experiment(f"{sex}_{model_type}")

    f1 = trial.value if trial.value is not None else float("nan")

    with mlflow.start_run():
        mlflow.log_params(trial.params)
        # mlflow.log_metric("auc", mean_auc)
        # mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        # mlflow.log_metric("precision", precision)
        # mlflow.log_metric("recall", recall)


def objective(trial: optuna.Trial):
    with mlflow.start_run(nested=True):
        params = model_params
        # params['alpha'] = trial.suggest_float('alpha', 1e-8, 1.0, log=True)
        # params['gamma'] = trial.suggest_float('gamma', 1e-8, 1e-6, log=True)
        params["max_depth"] = trial.suggest_int("max_depth", 3, 10, log=True)

        f1_array = []

        cv_inner = StratifiedKFold(n_splits=5)

        for train, val in tqdm(cv_inner.split(X_train, y_train)):

            if model_type == 'xgboost':
                model = xgb.XGBClassifier(**params)
            if model_type == 'catboost':
                model = CatBoostClassifier(*params)

            X_train_ = X_train[train]
            y_train_ = y_train[train]
            X_val = X_train[val]
            y_val = y_train[val]

            # model.set_params(**params)
            model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], verbose=False)
            pred = model.predict(X_test)

            # accuracy = accuracy_score(y_test, pred)
            f1_array.append(f1_score(y_test, pred))
            # precision = precision_score(y_test, pred)
            # recall = recall_score(y_test, pred)
            # return_values = accuracy, f1, precision, recall
        f1 = np.mean(f1_array)
        return f1


for train, val in tqdm(cv.split(X, y)):

    X_train = X[train]
    y_train = y[train]
    X_test = X[val]
    y_test = y[val]

    # train_scaler = StandardScaler().fit(X_train)
    # test_scaler = StandardScaler().fit(X_test)

    # X_train = train_scaler.transform(X_train)
    # X_test = test_scaler.transform(X_test)

    # y_train = label_encoder.transform(y_train)
    # y_test = label_encoder.transform(y_test)

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
    study.optimize(objective,
                   n_trials=10 - 3,
                   callbacks=[mlflow_callback])

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
