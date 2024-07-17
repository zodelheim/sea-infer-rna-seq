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

from tqdm import tqdm


fdir_raw = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/raw/")
fdir_processed = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/interim")
fdir_traintest = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/processed") / 'sex'
fdir_external = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/external")
ml_models_fdir = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/models")

use_CV = True

model_type = 'catboost'
model_type = 'xgboost'

sex = 'chrXY'
# sex = 'chrX'
# sex = 'chrY'
# sex = 'autosome'

print('```')
print("*" * 20)
print(model_type)
print(sex)
print("*" * 20)

n_threads = 6
params_xgb = {
    # "early_stopping_rounds": 20,
    "n_jobs": n_threads,
    "objective": 'binary:logistic',
    "n_estimators": 500,
    'device': 'cuda',
    'eta': 0.05,
    'max_depth': 3,
    "gamma": 1e-6,
    # 'verbosity': 0
    # "booster": "dart",
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
    model = xgb.XGBClassifier(**params_xgb)
    # model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], verbose=False)

if model_type == 'catboost':
    model = CatBoostClassifier(**params_catboost)
    # model.fit(X_train_, y_train_,
    #           eval_set=(X_val, y_val),
    #           verbose=False,
    #           use_best_model=True,
    #           plot=False,
    #           early_stopping_rounds=20)


fname = Path("heart.merged.TPM.preprocessed.csv")

data_heart = pd.read_csv(fdir_external / 'HEART' / 'reg' / fname, index_col=0)
# print(data_heart.head())

data_heart_header = pd.read_csv(fdir_external / 'HEART' / 'reg' / 'SraRunTable.txt', sep=',')
# print(data_heart_header)
print('ground true: ', (data_heart_header['sex'].values == 'male').astype(int))

# data_heart[data_heart < -12] = pd.NA


features = pd.read_csv(fdir_processed / f'feature_importance.{model_type}.{sex}.csv', index_col=0)
features = features.loc[features.index.intersection(data_heart.columns)]
features = features.sort_values(ascending=False, by="0")
n_features = 50
# print(features.iloc[:n_features].index)
data_heart = data_heart[features.iloc[:n_features].index]
data_heart.to_csv(fdir_external / 'HEART' / 'reg' / "heart.merged.TMP.preprocessed.important_features.csv")
# print(model)

X = data_heart.values

train_scaler = StandardScaler().fit(X)
X = StandardScaler().fit_transform(X)
proba = np.zeros(shape=(X.shape[0], 2))
pred = np.zeros(shape=(X.shape[0]))

for i in range(5):
    saved_model_filename = f"geuvadis_fold{i}_{sex}.json"
    model.load_model(fname=ml_models_fdir / model_type / saved_model_filename)

    proba += model.predict_proba(X)
    pred += model.predict(X)


proba = proba / 5
# pred = pred / 5
print('predicted:   ', (proba[:, 1] > 0.5).astype(int))
# print(pred.astype(int))

print('```')
