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

#! SHOLD BE THE SAME AS IN train_model.py
feature_importance_method = 'native'
# feature_importance_method = 'SHAP'

# sex = 'chrXY'
# sex = 'chrX'
# sex = 'chrY'
# sex = 'autosome'
n_features = 50

for sex in ['chrXY', 'chrX', 'chrY', 'autosome']:

    print('```')
    print("*" * 20)
    print(model_type)
    print(sex)
    print("*" * 20)

    with open(f'models/{model_type}.json', 'r') as file:
        model_params = json.load(file)

    if model_type == 'xgboost':
        model = xgb.XGBClassifier(**model_params)

    if model_type == 'catboost':
        model = CatBoostClassifier(**model_params)

    fname = Path("heart.merged.TPM.processed.h5")

    data_heart = pd.read_hdf(fdir_external / 'HEART' / 'reg' / fname, index_col=0)

    data_heart_header = pd.read_csv(fdir_external / 'HEART' / 'reg' / 'SraRunTable.txt', sep=',')
    print('ground true: ', (data_heart_header['sex'].values == 'male').astype(int))

    # data_heart[data_heart < -12] = pd.NA

    features = pd.read_hdf(fdir_processed / f'feature_importance.{model_type}.sex.h5', key=sex)

    features = features.loc[features.index.intersection(data_heart.columns), feature_importance_method]
    features = features.sort_values(ascending=False)
    # print(features.iloc[:n_features].index)
    data_heart = data_heart[features.iloc[:n_features].index]

    X = data_heart.values
    y = data_heart_header['sex'].values

    label_encoder = LabelEncoder().fit(y)
    print(label_encoder.classes_, "[0, 1]")

    y = label_encoder.transform(y)

    train_scaler = StandardScaler().fit(X)
    X = StandardScaler().fit_transform(X)
    proba = np.zeros(shape=(X.shape[0], 2))
    pred = np.zeros(shape=(X.shape[0]))

    accuracies = []
    f1 = []
    precisions = []
    recalls = []

    for i in range(5):
        saved_model_filename = f"geuvadis_fold{i}_{sex}.json"
        model.load_model(fname=ml_models_fdir / model_type / saved_model_filename)

        proba += model.predict_proba(X)

        pred_ = model.predict(X)
        if sex == 'autosome':
            pred_ = np.abs(pred_ - 1)
            # print(pred_)
        pred += pred_

        accuracies.append(accuracy_score(y, pred_))
        f1.append(f1_score(y, pred_))
        precisions.append(precision_score(y, pred_))
        recalls.append(recall_score(y, pred_))

    proba = proba / 5
    # pred = pred / 5
    if sex == 'autosome':
        print('predicted:   ', (proba[:, 0] > 0.5).astype(int))
    else:
        print('predicted:   ', (proba[:, 1] > 0.5).astype(int))
    # print(pred.astype(int))

    mean_accuracy = np.mean(accuracies)
    mean_f1 = np.mean(f1)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    print(f"{mean_accuracy=}")
    print(f"{mean_f1=}")
    print(f"{mean_precision=}")
    print(f"{mean_recall=}")
    print('```')
