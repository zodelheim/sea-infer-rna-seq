import pandas as pd
import numpy as np
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
import argparse
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from pathlib import Path
import json
import cupy
import anndata as ad

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from config import FDIR_EXTERNAL, FDIR_INTEMEDIATE, FDIR_PROCESSED, FDIR_RAW

from tqdm import tqdm

fdir_raw = FDIR_RAW
fdir_intermediate = FDIR_INTEMEDIATE
fdir_processed = FDIR_PROCESSED / "sex"
fdir_external = FDIR_EXTERNAL
ml_models_fdir = Path("models")


use_CV = True

model_type = "catboost"
model_type = "xgboost"
# model_type = "random_forest"

Scaler = RobustScaler
# Scaler = StandardScaler

# sex = 'chrXY'
# sex = 'autosome'

feature_importance_method = "native"
feature_importance_method = "SHAP"


organ_names = {
    "BRAIN0": "BRAIN0",
    "HEART": "HEART",
    "BRAIN1": "BRAIN1",
    "None": "BLOOD",
    "CAGE.HEART": "CAGE.HEART",
}

filename_prefixes = {
    "None": "geuvadis",
    "CAGE.HEART": "CAGE.HEART",
    "BRAIN0": "BRAIN0",
    "HEART": "HEART",
    "BRAIN1": "BRAIN1",
}


n_threads = 6

value_to_predict = "sex"
# value_to_predict = 'Experimental_Factor:_population (exp)'

drop_duplicates = True
drop_duplicates = False

for organ in ["BRAIN0", "HEART", "BRAIN1", "None"]:
    # for organ in ["BRAIN1"]:
    # for organ in ["CAGE.HEART"]:
    # for sex_chromosome in ['chr_aXY']:
    for sex_chromosome in ["chr_aXY", "autosomes", "chr_aX", "chr_aY"]:
        with open(f"models/model_params.json", "r") as file:
            model_params = json.load(file)[model_type]
        model_params = model_params[value_to_predict]

        # print(model_params)
        adata = ad.read_h5ad(
            fdir_processed
            / f"{filename_prefixes[organ].upper()}.preprocessed.{value_to_predict}.h5ad"
        )
        adata = adata[:, adata.varm[sex_chromosome]]

        if drop_duplicates:
            adata = adata[:, adata.varm["unique"]]

        feature_importance_df = pd.read_hdf(
            fdir_intermediate
            / f"feature_importance.{model_type}.{value_to_predict}.organ_{organ}.h5",
            key=f"{sex_chromosome}",
        )

        features = feature_importance_df[feature_importance_method]

        if organ not in ["None", "CAGE.HEART"]:
            fname = next((fdir_external / organ / "reg").glob("*processed.h5"))
            fname = fname.name

            data_eval = pd.read_hdf(fdir_external / organ / "reg" / fname, index_col=0)
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

            data_shrinked = adata[:, features[:n_features]]

            X = data_shrinked.X
            y = data_shrinked.obs[value_to_predict]

            label_encoder = LabelEncoder().fit(y)
            y = label_encoder.transform(y)

            roc_array = []

            accuracy_array = []
            f1_array = []
            precision_array = []
            recall_array = []

            # cv = StratifiedKFold(n_splits=5, shuffle=True)
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

                X_train = Scaler().fit_transform(X_train)
                X_test = Scaler().fit_transform(X_test)

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

                # with Live("models/log") as live:
                if model_type == "xgboost":
                    # model_params["callbacks"] = [DVCLiveCallback()]
                    model = xgb.XGBClassifier(**model_params)
                    model.fit(
                        cupy.array(X_train_), y_train_, eval_set=[(X_val, y_val)], verbose=False
                    )

                    X_test_c = cupy.array(X_test)

                if model_type == "catboost":
                    model = CatBoostClassifier(**model_params)
                    model.fit(
                        X_train_,
                        y_train_,
                        eval_set=(X_val, y_val),
                        verbose=False,
                        use_best_model=True,
                        plot=False,
                        early_stopping_rounds=20,
                    )

                if model_type == "random_forest":
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)
                    X_test_c = X_test

                    # live.log_metric("summary_metric", 1.0, plot=False)

                y_pred = model.predict(X_test_c)

                roc_array.append(roc_auc_score(y_test, y_pred))

                accuracy_array.append(accuracy_score(y_test, y_pred))
                f1_array.append(f1_score(y_test, y_pred))
                precision_array.append(precision_score(y_test, y_pred))
                recall_array.append(recall_score(y_test, y_pred))

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
        plt.errorbar(
            np.arange(1, max_n_features),
            roc_array_df.mean(),
            yerr=roc_array_df.std(),
            label="roc auc",
        )
        plt.errorbar(
            np.arange(1, max_n_features),
            accuracy_array_df.mean(),
            yerr=accuracy_array_df.std(),
            label="accuracy",
        )
        plt.errorbar(
            np.arange(1, max_n_features), f1_array_df.mean(), yerr=f1_array_df.std(), label="f1"
        )
        plt.errorbar(
            np.arange(1, max_n_features),
            precision_array_df.mean(),
            yerr=precision_array_df.std(),
            label="precision",
        )
        plt.errorbar(
            np.arange(1, max_n_features),
            recall_array_df.mean(),
            yerr=recall_array_df.std(),
            label="recall",
        )
        plt.title(sex_chromosome + ", organ: " + organ_names[organ])
        plt.ylim((0.4, 1.0))
        plt.ylabel("score value")
        plt.xlabel("# transcripts")
        plt.legend()
        plt.savefig(
            f"reports/figures/nfeatures/{filename_prefixes[organ]}_{sex_chromosome}_organ_{organ}.png",
            dpi=300,
        )
        plt.close()
        # plt.show()
        # exit()
