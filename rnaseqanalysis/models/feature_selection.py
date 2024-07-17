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

from tqdm import tqdm

fdir_raw = Path("data/raw/")
fdir_processed = Path("data/interim")
fdir_traintest = Path("data/processed") / 'sex'

use_CV = True

model_type = 'catboost'
model_type = 'xgboost'

# sex = 'chrXY'
# sex = 'autosome'

for sex in ['chrXY', 'autosome', 'chrX', 'chrY']:

    feature_importance_method = 'native'
    feature_importance_method = 'shap'

    n_threads = 6
    params_xgb = {
        # "early_stopping_rounds": 20,
        "n_jobs": n_threads,
        "objective": 'binary:logistic',
        "n_estimators": 500,
        'device': 'cuda',
        'eta': 0.05,
        "gamma": 1e-6,
        'max_depth': 3,
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

    data = pd.read_hdf(fdir_traintest / f'geuvadis.preprocessed.sex.h5', key=sex)

    # data = pd.read_csv(fdir_traintest / f'geuvadis.preprocessed.10_features.{sex}.csv', index_col=0)

    data_header = pd.read_hdf(fdir_processed / 'geuvadis.preprocessed.h5', key="header")
    # print(data_header)
    # exit()
    # data_header = data_header[[
    #     'Sex',
    #     # 'Experimental_Factor:_population (exp)'
    # ]]
    # data_header = data_header.loc[data.index]

    # data[data < -12] = pd.NA
    # --------------------------------------------------------------------------------
    X = data.values
    y = data_header['Sex']

    label_encoder = LabelEncoder().fit(y)
    y = label_encoder.transform(y)

    feature_importance_dict = {}
    n_features_is_subset = 100
    n_features_to_print = 30

    # cv = StratifiedKFold(n_splits=5)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
    # X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, shuffle=True)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    accuracies = []
    f1 = []
    precisions = []
    recalls = []

    fig, ax = plt.subplots(figsize=(6, 6))
    for train, val in tqdm(cv.split(X, y)):
        X_train = X[train]
        y_train = y[train]
        X_test = X[val]
        y_test = y[val]

        train_scaler = StandardScaler().fit(X_train)
        test_scaler = StandardScaler().fit(X_test)

        X_train = train_scaler.transform(X_train)
        X_test = test_scaler.transform(X_test)

        # y_train = label_encoder.transform(y_train)
        # y_test = label_encoder.transform(y_test)

        X_train_ = X_train
        y_train_ = y_train
        X_val = X_test
        y_val = y_test

        if model_type == 'xgboost':
            model = xgb.XGBClassifier(**params_xgb)
            model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], verbose=False)

        if model_type == 'catboost':
            model = CatBoostClassifier(**params_catboost)
            model.fit(X_train_, y_train_,
                      eval_set=(X_val, y_val),
                      verbose=False,
                      use_best_model=True,
                      plot=False,
                      early_stopping_rounds=20)

        pred = model.predict(X_test)
        pred_prob = model.predict_proba(X_test)

        # ConfusionMatrixDisplay(
        #     confusion_matrix(y_test,
        #                      pred)
        # )  # .plot()

        # viz = RocCurveDisplay.from_predictions(
        #     y_test, pred_prob[:, 1]
        # )

        if feature_importance_method == 'native':
            importances = model.feature_importances_
        if feature_importance_method == 'shap':
            # importances = model.feature_importances_
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            importances = np.abs(shap_values).mean(axis=0)

        feature_importance_df = pd.DataFrame({
            'Feature': data.columns,
            'Importance': importances
            # 'Importance': XGB.feature_importances_
        })
        feature_importance_df = feature_importance_df.sort_values(
            by='Importance', ascending=False)
        # plt.show()

        features = feature_importance_df['Feature'].iloc[:n_features_is_subset].values
        for feature in features:
            if feature not in feature_importance_dict.keys():
                feature_importance_dict[feature] = 0

            feature_importance_dict[feature] += 1

            # print(feature_importance_df.iloc[:20])

            # sns.barplot(feature_importance_df.iloc[:30],
            #             x='Importance', y="Feature")
            # plt.show()

            # n_features = 10
            # data = data[feature_importance_df["Feature"].iloc[:n_features]]

            # data.to_csv(fdir_traintest / f'geuvadis.preprocessed.{n_features}_features.{sex}.csv')

        viz = RocCurveDisplay.from_predictions(
            y_test, pred_prob[:, 1],
            ax=ax,
        )

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0
        tprs.append(interp_tpr)

        accuracies.append(accuracy_score(y_test, pred))
        f1.append(f1_score(y_test, pred))
        precisions.append(precision_score(y_test, pred))
        recalls.append(recall_score(y_test, pred))

    # plt.show()

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    mean_accuracy = np.mean(accuracies)
    mean_f1 = np.mean(f1)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)

    print(sex)
    print("-" * 20)
    print(f"{mean_auc=}")
    print(f"{mean_accuracy=}")
    print(f"{mean_f1=}")
    print(f"{mean_precision=}")
    print(f"{mean_recall=}")
    print("-" * 20)

    feature_importance_df = pd.Series(feature_importance_dict)
    feature_importance_df = feature_importance_df.sort_values(ascending=False)

    print(feature_importance_df.iloc[:n_features_to_print])

    feature_importance_df.to_csv(fdir_processed / f'feature_importance.{model_type}.{sex}.csv')

    # exit()

    feature_importance_df = pd.read_csv(fdir_processed / f'feature_importance.{model_type}.{sex}.csv', index_col=0)
    # features = feature_importance_df['Feature']
    features = feature_importance_df.index

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
                model = xgb.XGBClassifier(**params_xgb)
                model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], verbose=False)

            if model_type == 'catboost':
                model = CatBoostClassifier(**params_catboost)
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
    plt.show()
