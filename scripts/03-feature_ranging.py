import argparse
import json

import anndata as ad
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
Scaler = RobustScaler
KFoldType = RepeatedStratifiedKFold

drop_duplicates = False

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
    logger.info(
        f"Compute importances for features in TRAIN:{cfg.models.train_dataset} and TEST:{name} datasets"
    )

    ofname = f"feature_importance.{cfg.model_params.model_type}.{value_to_predict}.{cfg.models.train_dataset.upper()}.{name}.h5"

    if (cfg.paths.interim_path / ofname).is_file():
        logger.warning(f"File {cfg.paths.interim_path / ofname} exists! Skipping...")
        continue

    for sex_chromosome in ["chr_aXY", "autosomes", "chr_aX", "chr_aY"]:
        models_statistics = []
        adata = train_adata[:, train_adata.varm[sex_chromosome]].copy()

        if drop_duplicates:
            adata = adata[:, adata.varm["unique"]]

        data_eval = ad.read_h5ad(
            cfg.paths.processed_path / f"{name.upper()}.preprocessed.{value_to_predict}.h5ad"
        ).to_df()
        adata = adata[:, adata.var_names.intersection(data_eval.columns)]

        X = np.asarray(adata.X)
        y = np.asarray(adata.obs[value_to_predict])

        label_encoder = LabelEncoder().fit(y)
        y = label_encoder.transform(y)

        class_names = label_encoder.classes_

        feature_importance_df = pd.DataFrame(
            np.zeros(shape=(adata.n_vars, 3), dtype=int), columns=["Feature", "native", "SHAP"]
        )
        feature_importance_df["Feature"] = adata.var_names
        feature_importance_df.set_index("Feature", inplace=True)

        n_features_is_subset = 100
        n_features_to_print = 30

        cv = KFoldType(n_splits=5, n_repeats=10)
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        accuracies = []
        f1 = []
        precisions = []
        recalls = []

        for train, val in track(
            cv.split(X, y),
            total=cv.get_n_splits(X, y),
            description=f"Processing {name}:{sex_chromosome}...",
        ):
            X_train = X[train]
            y_train = y[train]
            X_test = X[val]
            y_test = y[val]

            train_scaler = Scaler().fit(X_train)
            X_train = train_scaler.transform(X_train)
            X_test = train_scaler.transform(X_test)

            X_train_ = X_train
            y_train_ = y_train

            if _USE_GPU:
                X_train_ = cp.array(X_train_)
                y_train_ = cp.array(y_train_)
                X_test = cp.array(X_test)
                y_test = cp.array(y_test)
                X_val = X_test
                y_val = y_test

            else:
                X_train_ = np.array(X_train_)
                y_train_ = np.array(y_train_)
                X_test = np.array(X_test)
                y_test = np.array(y_test)
                X_val = X_test
                y_val = y_test

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
            pred = model.predict(X_test_c)
            pred_prob = model.predict_proba(X_test_c)
            models_statistics.append(model.evals_result()["validation_0"]["logloss"])
            importances_native = model.feature_importances_

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            importances_shap = np.abs(shap_values).mean(axis=0)

            if len(importances_shap.shape) > 1:
                importances_dict = {
                    "Feature": adata.var_names,
                    "SHAP": importances_shap.sum(axis=1),
                    "native": importances_native,
                }
                for idx, value in enumerate(class_names):
                    importances_dict[f"SHAP_{value}"] = importances_shap[:, idx]
            else:
                importances_dict = {
                    "Feature": adata.var_names,
                    "SHAP": importances_shap,
                    "native": importances_native,
                }

            feature_importance_ = pd.DataFrame(importances_dict)

            for fe in ["SHAP", "native"]:
                feature_importance_ = feature_importance_.sort_values(by=fe, ascending=False)

                features = feature_importance_["Feature"].iloc[:n_features_is_subset].values

                for feature in features:
                    feature_importance_df.loc[feature, fe] += 1

            if len(class_names) == 1:
                viz = RocCurveDisplay.from_predictions(
                    y_test,
                    pred_prob[:, 1],
                    ax=ax,
                )

                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0
                tprs.append(interp_tpr)

                accuracies.append(accuracy_score(y_test, pred))
                f1.append(f1_score(y_test, pred))
                precisions.append(precision_score(y_test, pred))
                recalls.append(recall_score(y_test, pred))

        if len(class_names) == 1:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0

            mean_auc = auc(mean_fpr, mean_tpr)
            mean_accuracy = np.mean(accuracies)
            mean_f1 = np.mean(f1)
            mean_precision = np.mean(precisions)
            mean_recall = np.mean(recalls)

            logger.info(sex_chromosome)
            logger.info("-" * 20)
            logger.info(f"{mean_auc=}")
            logger.info(f"{mean_accuracy=}")
            logger.info(f"{mean_f1=}")
            logger.info(f"{mean_precision=}")
            logger.info(f"{mean_recall=}")
            logger.info("-" * 20)

        feature_importance_df = feature_importance_df.sort_values(by="SHAP", ascending=False)
        logger.info(f"Top {n_features_to_print} features by SHAP")
        logger.info(feature_importance_df.iloc[:n_features_to_print])

        feature_importance_df = feature_importance_df.sort_values(by="native", ascending=False)
        logger.info(f"Top {n_features_to_print} features by model.feature_importances_")
        logger.info(feature_importance_df.iloc[:n_features_to_print])

        # feature_importance_df.to_csv(fdir_processed / f'feature_importance.{model_type}.{sex}.csv')
        feature_importance_df.to_hdf(
            cfg.paths.interim_path / ofname,
            key=f"{sex_chromosome}",
            format="f",
        )

        logger.info(f"Saved to {cfg.paths.interim_path / ofname}")
        logger.info(f"Key={sex_chromosome}")

        with open(
            cfg.paths.logs / f"{cfg.models.train_dataset.upper()}{sex_chromosome}_model_log.json",
            "w",
        ) as json_file:
            json.dump(models_statistics, json_file, indent=4)

    logger.info(f"Saved to {cfg.paths.interim_path / ofname}")
