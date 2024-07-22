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
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, KFold
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, auc
from sklearn.decomposition import PCA
import umap
import json

from sklearn.neighbors import KNeighborsClassifier


from tqdm import tqdm

# mlflow.set_tracking_uri(uri="http://localhost:8080")


fdir_raw = Path("data/raw/")
fdir_processed = Path("data/interim")
fdir_traintest = Path("data/processed") / 'sex'
fdir_external = Path("data/external")
ml_models_fdir = Path("models")

use_CV = True

model_type = 'catboost'
model_type = 'xgboost'
# model_type = 'knn'

feature_importance_method = 'native'
feature_importance_method = 'SHAP'

n_features = 0

value_to_predict = 'Sex'

organ = ["BLOOD1", 'BRAIN0', "HEART", "BRAIN1", 'None'][2]

for sex in ['chrXY', 'chrX', 'chrY', 'autosome']:
    # for sex in ['chrY']:

    print("*" * 20)
    print(model_type)
    print(sex)
    print("*" * 20)

    with open(f'models/{model_type}.json', 'r') as file:
        model_params = json.load(file)

    data = pd.read_hdf(fdir_traintest / f'geuvadis.preprocessed.sex.h5', key=sex)

    features = pd.read_hdf(
        fdir_processed / f'feature_importance.{"xgboost"}.{value_to_predict}.h5',
        key=f'{sex}',
    )

    features = features[feature_importance_method]
    features = features.sort_values(ascending=False)

    if organ != "None":
        fname = next((fdir_external / organ / 'reg').glob("*processed.h5"))
        fname = fname.name

        data_eval = pd.read_hdf(fdir_external / organ / 'reg' / fname, index_col=0)
        features = features.loc[features.index.intersection(data_eval.columns)]
        if n_features != 0:
            features = features.sort_values(ascending=False)
            print(features.iloc[:n_features])
        else:
            print(features.shape)

    if n_features != 0:
        features_list = features.iloc[:n_features]
    else:
        features_list = features

    features_fname = f"geuvadis_features_{sex}_calibration_{organ}.csv"
    features_list.to_csv(ml_models_fdir / model_type / features_fname)

    data = data[features_list.index]

    data_header = pd.read_hdf(fdir_processed / 'geuvadis.preprocessed.h5', key='header')

    X = data.values
    y = data_header['Sex']

    # X_comps = PCA(2).fit_transform(X)
    # sns.scatterplot(x=X_comps[:, 0], y=X_comps[:, 1], hue=y)
    # plt.show()

    # reducer = umap.UMAP()
    # embedding = reducer.fit_transform(X)
    # sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=y)
    # plt.show()
    # exit()

    test_size = 0.2
    random_state = 42

    label_encoder = LabelEncoder().fit(y)
    print(label_encoder.classes_, "[0, 1]")

    # print(dir(label_encoder))
    # exit()
    y = label_encoder.transform(y)

    # ----------------------------------------------------------------------------------------
    if use_CV:
        cv = StratifiedKFold(n_splits=5)
    else:
        X_train_, X_val, y_train_, y_val = train_test_split(X, y)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    accuracies = []
    f1 = []
    precisions = []
    recalls = []

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (train, val) in tqdm(enumerate(cv.split(X, y))):
        X_train = X[train]
        y_train = y[train]
        X_test = X[val]
        y_test = y[val]

        train_scaler = StandardScaler().fit(X_train)
        test_scaler = StandardScaler().fit(X_test)

        X_train = train_scaler.transform(X_train)
        X_test = test_scaler.transform(X_test)

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

        if model_type == 'knn':
            model = KNeighborsClassifier(**model_params)
            model.fit(X_train_, y_train_)

        if not (ml_models_fdir / model_type).is_dir():
            (ml_models_fdir / model_type).mkdir()

        pred = model.predict(X_test)
        pred_prob = model.predict_proba(X_test)

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

        saved_model_filename = f"geuvadis_fold{i}_{sex}_calibration_{organ}.json"

        if model_type is not 'knn':
            model.save_model(fname=ml_models_fdir / model_type / saved_model_filename)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    mean_accuracy = np.mean(accuracies)
    mean_f1 = np.mean(f1)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)

    print("-" * 20)
    print(f"{mean_auc=},")
    print(f"{mean_accuracy=},")
    print(f"{mean_f1=},")
    print(f"{mean_precision=},")
    print(f"{mean_recall=},")
    print("-" * 20)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        lw=2,
        alpha=0.8,
    )
    plt.show()

    # exit()
    # with mlflow.start_run():
    #     mlflow.log_params(params_xgb)

    #     mlflow.log_metric("auc", mean_auc)
    #     mlflow.log_metric("accuracy", mean_accuracy)
    #     mlflow.log_metric("f1", mean_f1)
    #     mlflow.log_metric("precision", mean_precision)
    #     mlflow.log_metric("recall", mean_recall)

    #     mlflow.set_tag("Training Info", "Basic LR model for iris data")

    #     signature = infer_signature(X_train, model.predict(X_train))

    #     model_info = mlflow.xgboost.log_model(
    #         xgb_model=model,
    #         artifact_path=f"{sex}_data",
    #         signature=signature,
    #         input_example=X_train,
    #         # registered_model_name="tracking-quickstart",
    #     )
    #     mlflow.shap.log_explanation(shap.Explainer(model).shap_values, X_train)
