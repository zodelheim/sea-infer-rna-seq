import argparse
import json
from pathlib import Path

# import cupy
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from catboost import CatBoostClassifier
from config.loader import Config, load_yaml
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from utils.errors import *


try:
    import cupy as cp

    _USE_GPU = True

except ModuleNotFoundError:
    _USE_GPU = False

models_lookup = {"xgboost": xgb.XGBClassifier, "catboost": CatBoostClassifier}


logger.remove()
logger.add(
    RichHandler(markup=True, rich_tracebacks=True),
    level="INFO",
    format="{message}",
)

console = Console()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

cfg = Config(**load_yaml(args.config))

value_to_predict = cfg.models.value_to_predict
model_type = cfg.model_params.model_type
feature_importance_method = cfg.model_params.feature_importance_method

use_CV = cfg.model_params.use_CV
drop_duplicates = False
Scaler = RobustScaler
KFoldType = RepeatedStratifiedKFold

max_n_features = 100

with open(cfg.model_params.config_file, "r") as file:
    config = json.load(file)
    if cfg.model_params.model_type not in config:
        logger.warning(
            f"No params for model '{cfg.model_params.model_type}' found in '{cfg.model_params.config_file.absolute()}'. \nUsing default params."
        )
        model_params = {}
    else:
        model_params = config[cfg.model_params.model_type]

model = models_lookup[cfg.model_params.model_type](**model_params)

logger.info(f"Scaler: {Scaler}")
logger.info(f"KFoldType: {KFoldType}")
logger.info(f"Model: {models_lookup[cfg.model_params.model_type]}")
logger.info(f"Config: {model_params}")

logger.info(f"Read train dataset: {cfg.models.train_dataset}")
train_adata = ad.read_h5ad(
    cfg.paths.processed_path
    / f"{cfg.models.train_dataset.upper()}.preprocessed.{cfg.models.value_to_predict}.h5ad"
)

for name in cfg.datasets:
    logger.info(f"Selecting features in TRAIN:{cfg.models.train_dataset} and TEST:{name} datasets")
    for sex_chromosome in ["chr_aXY", "autosomes", "chr_aX", "chr_aY"]:
        adata = train_adata[:, train_adata.varm[sex_chromosome]].copy()

        if drop_duplicates:
            adata = adata[:, adata.varm["unique"]]

        fi_fname = f"feature_importance.{cfg.model_params.model_type}.{value_to_predict}.{cfg.models.train_dataset.upper()}.{name}.h5"
        feature_importance_df = pd.read_hdf(
            cfg.paths.interim_path / fi_fname,
            key=f"{sex_chromosome}",
        )
        features = feature_importance_df[feature_importance_method]

        data_eval = ad.read_h5ad(
            cfg.paths.processed_path / f"{name.upper()}.preprocessed.{value_to_predict}.h5ad"
        ).to_df()

        features = features.loc[features.index.intersection(data_eval.columns)]
        features = features.sort_values(ascending=False).index

        roc_array_total = {}
        accuracy_array_total = {}
        f1_array_total = {}
        precision_array_total = {}
        recall_array_total = {}

        for i in track(
            range(1, max_n_features),
            description=f"Processing {name}:{sex_chromosome}...",
        ):
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

            cv = KFoldType(n_splits=5, n_repeats=10)
            for train, val in cv.split(X, y):
                X_train = X[train]
                y_train = y[train]
                X_test = X[val]
                y_test = y[val]

                train_scaler = Scaler().fit(X_train)
                X_train = train_scaler.transform(X_train)
                X_test = train_scaler.transform(X_test)

                X_train_ = X_train
                y_train_ = y_train
                y_val = y_test

                if _USE_GPU:
                    X_train_ = cp.array(X_train_)
                    y_train_ = cp.array(y_train_)
                    X_test = cp.array(X_test)
                    X_val = X_test

                else:
                    X_train_ = np.array(X_train_)
                    y_train_ = np.array(y_train_)
                    X_test = np.array(X_test)
                    X_val = X_test

                if cfg.model_params.model_type == "xgboost":
                    model.fit(
                        X_train_,
                        y_train_,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                    )
                    X_test_c = X_test

                else:
                    raise NotImplementedError()

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
        plt.title(f"{sex_chromosome}.{cfg.models.train_dataset}.{name}")
        plt.ylim((0.4, 1.0))
        plt.ylabel("score value")
        plt.xlabel("# transcripts")
        plt.legend()
        plt.savefig(
            cfg.paths.figures / f"{sex_chromosome}.{cfg.models.train_dataset}.{name}.png",
            dpi=300,
        )
        plt.close()
