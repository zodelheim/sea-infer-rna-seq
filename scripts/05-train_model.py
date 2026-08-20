import argparse
import json
import anndata as ad
import cupy
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from config.loader import Config, load_yaml, FeatureSelectionMethodsEnum
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from sklearn.decomposition import PCA
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, RobustScaler

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
drop_duplicates = False
test_size = 0.2
random_state = 42
use_CV = cfg.model_params.use_CV

Scaler = RobustScaler
KFoldType = RepeatedStratifiedKFold

if cfg.model_params.feature_selection_method == FeatureSelectionMethodsEnum.top50:
    n_features = 50


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
logger.info(f"Model: {cfg.model_params.model_type} - {models_lookup[cfg.model_params.model_type]}")
logger.info(f"KFoldType: {KFoldType}")
logger.info(f"Config: {model_params}")

logger.info(f"Read train dataset: {cfg.models.train_dataset}")
train_adata = ad.read_h5ad(
    cfg.paths.processed_path
    / f"{cfg.models.train_dataset.upper()}.preprocessed.{value_to_predict}.h5ad"
)


result_dict = {}

for name in cfg.datasets:
    logger.info(
        f"Training on features from TRAIN:{cfg.models.train_dataset} and TEST:{name} datasets"
    )
    result_dict[name] = {}
    for sex_chromosome in ["chr_aXY", "autosomes", "chr_aX", "chr_aY"]:
        console.rule(f"Processing {name}: {sex_chromosome}")

        result_dict[name][sex_chromosome] = {}
        adata = train_adata[:, train_adata.varm[sex_chromosome]].copy()
        if drop_duplicates:
            adata = adata[:, adata.varm["unique"]]

        fi_fname = f"feature_importance.{cfg.model_params.model_type}.{value_to_predict}.{cfg.models.train_dataset.upper()}.{name}.h5"
        features = pd.read_hdf(
            cfg.paths.interim_path / fi_fname,
            key=f"{sex_chromosome}",
        )
        logger.info(f"Read Feature Importance: {cfg.paths.interim_path / fi_fname}")
        logger.info(f"Key={sex_chromosome}")

        features = features[cfg.model_params.feature_importance_method]
        features = features.sort_values(ascending=False)

        data_eval = ad.read_h5ad(
            cfg.paths.processed_path / f"{name.upper()}.preprocessed.{value_to_predict}.h5ad"
        ).to_df()
        features = features.loc[features.index.intersection(data_eval.columns)]

        if n_features != 0:
            features = features.sort_values(ascending=False)
        features_list = features.iloc[:n_features]

        logger.info(f"Number of features: {len(features_list)}")
        logger.info(f"{len(features_list)}")

        features_fname = (
            f"train_features.{sex_chromosome}.{cfg.models.train_dataset.upper()}.{name}.csv"
        )
        features_list.to_csv(cfg.paths.models / cfg.model_params.model_type / features_fname)

        adata = adata[:, features_list.index]

        X = adata.X
        y = adata.obs[value_to_predict]

        label_encoder = LabelEncoder().fit(y)
        logger.info(f"Map {label_encoder.classes_} to [0, 1]")

        y = label_encoder.transform(y)

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

        for i, (train, val) in track(enumerate(cv.split(X, y))):
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

            pred = model.predict(cupy.array(X_test))
            pred_prob = model.predict_proba(cupy.array(X_test))

            preds[val] = pred
            preds_proba[val] = pred_prob[:, 1]

            viz = RocCurveDisplay.from_predictions(
                y_test,
                pred_prob[:, 1],
            )

            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0
            tprs.append(interp_tpr)

            accuracies.append(accuracy_score(y_test, pred))
            f1.append(f1_score(y_test, pred))
            precisions.append(precision_score(y_test, pred))
            recalls.append(recall_score(y_test, pred))

            (cfg.paths.models / cfg.model_params.model_type).mkdir(exist_ok=True)

            saved_model_filename = f"model_weights.fold{i}.{sex_chromosome}.{cfg.models.train_dataset.upper()}.{name}.json"
            if cfg.model_params.model_type != "knn":
                model.save_model(
                    fname=cfg.paths.models / cfg.model_params.model_type / saved_model_filename
                )

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0

        mean_auc = auc(mean_fpr, mean_tpr)
        mean_accuracy = np.mean(accuracies)
        mean_f1 = np.mean(f1)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)

        logger.info(f"{mean_auc=},")
        logger.info(f"{mean_accuracy=},")
        logger.info(f"{mean_f1=},")
        logger.info(f"{mean_precision=},")
        logger.info(f"{mean_recall=},")

        total_auc = roc_auc_score(y, preds_proba)
        total_accuracy = accuracy_score(y, preds)
        total_f1 = f1_score(y, preds)
        total_precision = precision_score(y, preds)
        total_recall = recall_score(y, preds)

        logger.info(f"{total_auc=},")
        logger.info(f"{total_accuracy=},")
        logger.info(f"{total_f1=},")
        logger.info(f"{total_precision=},")
        logger.info(f"{total_recall=},")

        result_dict[name][sex_chromosome]["mean_auc"] = total_auc
        result_dict[name][sex_chromosome]["mean_accuracy"] = total_accuracy
        result_dict[name][sex_chromosome]["mean_f1"] = total_f1
        result_dict[name][sex_chromosome]["mean_precision"] = total_precision
        result_dict[name][sex_chromosome]["mean_recall"] = total_recall
        result_dict[name][sex_chromosome]["n_features"] = n_features

with open(
    cfg.paths.results
    / f"train_results.{cfg.model_params.model_type}.{value_to_predict}.{cfg.models.train_dataset.upper()}.json",
    "w",
) as file:
    json.dump(result_dict, file)
