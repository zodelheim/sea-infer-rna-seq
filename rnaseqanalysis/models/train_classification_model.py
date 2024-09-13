import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
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
import cupy

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

value_to_predict = 'Sex'
# value_to_predict = 'population'

result_dict = {}

n_featues_dict = {
    'BRAIN0': {
        'chrXY': 10,
        'chrX': 9,
        'chrY': 80,
        'autosome': 91,
    },
    'BRAIN1': {
        'chrXY': 7,
        'chrX': 5,
        'chrY': 5,
        'autosome': 59,
    },
    'HEART': {
        'chrXY': 9,
        'chrX': 8,
        'chrY': 93,
        'autosome': 82,
    },
    'None': {
        'chrXY': 10,
        'chrX': 10,
        'chrY': 3,
        'autosome': 82,
    }
}


features_shapsumm_threshold = 45

save_results = True
save_features = False

for organ in ['BRAIN0', "HEART", "BRAIN1", 'None']:
    result_dict[organ] = {}
    for sex in ['chrXY', 'chrX', 'chrY', 'autosome']:
        # for sex in ['chrY']:
        result_dict[organ][sex] = {}

        print("*" * 20)
        print(organ)
        print(model_type)
        print(sex)
        print("*" * 20)

        # n_features = 0
        n_features = n_featues_dict[organ][sex]

        with open(f'models/{model_type}.json', 'r') as file:
            model_params = json.load(file)
        model_params = model_params[value_to_predict]

        data = pd.read_hdf(fdir_traintest / f'geuvadis.preprocessed.sex.h5', key=sex)

        features = pd.read_hdf(
            fdir_processed / f'feature_importance.{model_type}.{value_to_predict}.organ_{organ}.h5',
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
                # print(features.iloc[:n_features])
            else:
                print(features.shape)

        if n_features != 0:
            features_list = features.iloc[:n_features]
        else:
            features_list = features.loc[features >= features_shapsumm_threshold]
            n_features = len(features_list)

        if save_features:
            features_fname = f"geuvadis_train_features_{sex}_calibration_{organ}.csv"
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

        preds = np.zeros(shape=y.shape)
        preds_proba = np.zeros(shape=y.shape)

        fig, ax = plt.subplots(figsize=(6, 6))
        for i, (train, val) in tqdm(enumerate(cv.split(X, y))):
            X_train = X[train]
            y_train = y[train]
            X_test = X[val]
            y_test = y[val]

            # train_scaler = StandardScaler().fit(X_train)
            # test_scaler = StandardScaler().fit(X_test)

            train_scaler = RobustScaler().fit(X_train)
            test_scaler = RobustScaler().fit(X_test)

            X_train = train_scaler.transform(X_train)
            X_test = test_scaler.transform(X_test)

            X_train_ = X_train
            y_train_ = y_train
            X_val = X_test
            y_val = y_test

            if model_type == 'xgboost':
                model = xgb.XGBClassifier(**model_params)
                model.fit(cupy.array(X_train_), y_train_, eval_set=[(X_val, y_val)], verbose=False)

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

            pred = model.predict(cupy.array(X_test))
            pred_prob = model.predict_proba(cupy.array(X_test))

            preds[val] = pred
            preds_proba[val] = pred_prob[:, 1]

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

            if save_results:
                saved_model_filename = f"geuvadis_fold{i}_{sex}_calibration_{organ}.json"
                if model_type != 'knn':
                    model.save_model(fname=ml_models_fdir / model_type / saved_model_filename)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0

        mean_auc = auc(mean_fpr, mean_tpr)
        mean_accuracy = np.mean(accuracies)
        mean_f1 = np.mean(f1)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)

        print("-" * 20)
        print("-" * 20)
        print(f"{mean_auc=},")
        print(f"{mean_accuracy=},")
        print(f"{mean_f1=},")
        print(f"{mean_precision=},")
        print(f"{mean_recall=},")
        print("-" * 20)

        total_auc = roc_auc_score(y, preds_proba)
        total_accuracy = accuracy_score(y, preds)
        total_f1 = f1_score(y, preds)
        total_precision = precision_score(y, preds)
        total_recall = recall_score(y, preds)

        print(f"{total_auc=},")
        print(f"{total_accuracy=},")
        print(f"{total_f1=},")
        print(f"{total_precision=},")
        print(f"{total_recall=},")
        print("-" * 20)

        result_dict[organ][sex]['mean_auc'] = total_auc
        result_dict[organ][sex]['mean_accuracy'] = total_accuracy
        result_dict[organ][sex]['mean_f1'] = total_f1
        result_dict[organ][sex]['mean_precision'] = total_precision
        result_dict[organ][sex]['mean_recall'] = total_recall
        result_dict[organ][sex]['n_features'] = n_features

        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            lw=2,
            alpha=0.8,
        )
        plt.title(f"{model_type}, {sex}, {organ}")
        if save_results:
            plt.savefig(f'reports/figures/geuvadis_{sex}_organ_{organ}.png', dpi=300)
            plt.close()
        else:
            plt.show()

        # if save_results:
        _ = ConfusionMatrixDisplay.from_predictions(y, preds, display_labels=['F', "M"])
        plt.title(f"{model_type}, {sex}, {organ}")
        if save_results:
            plt.savefig(f'reports/figures/geuvadis_cm_{sex}_organ_{organ}.png', dpi=300)
            plt.close()
        else:
            plt.show()


if save_results:
    with open(f'reports/train_result_{value_to_predict}_{model_type}.json', 'w') as file:
        json.dump(result_dict, file)
