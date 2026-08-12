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
    r2_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, RobustScaler
from scipy.stats import spearmanr


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

result_dict = {}
value_to_predict = cfg.models.value_to_predict
drop_duplicates = False

if cfg.model_params.feature_selection_method == FeatureSelectionMethodsEnum.top50:
    n_features = 50

logger.info(f"Read train dataset: {cfg.models.train_dataset}")
train_adata = ad.read_h5ad(
    cfg.paths.processed_path
    / f"{cfg.models.train_dataset.upper()}.preprocessed.{value_to_predict}.h5ad"
)

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


for name in cfg.datasets:
    logger.info(f"Evaluation model on EVAL:{name} dataset")
    result_dict[name] = {}
    for sex_chromosome in ["chr_aXY", "autosomes", "chr_aX", "chr_aY"]:
        console.rule(f"Predicting {name}: {sex_chromosome}, model: {cfg.model_params.model_type}")
        result_dict[name][sex_chromosome] = {}
        adata = train_adata[:, train_adata.varm[sex_chromosome]].copy()
        if drop_duplicates:
            adata = adata[:, adata.varm["unique"]]

        data_eval = ad.read_h5ad(
            cfg.paths.processed_path / f"{name.upper()}.preprocessed.{value_to_predict}.h5ad"
        )

        features_fname = (
            f"train_features.{sex_chromosome}.{cfg.models.train_dataset.upper()}.{name}.csv"
        )
        features_list = pd.read_csv(
            cfg.paths.models / cfg.model_params.model_type / features_fname, index_col=0
        )

        adata = adata[:, features_list.index]
        data_eval = data_eval[:, features_list.index]

        logger.info(f"Number of features: {len(features_list)}")
        logger.info(f"Features: {list(features_list.index)}")

        X_train = adata.X
        y_train = adata.obs[value_to_predict]

        X = data_eval.X
        y = data_eval.obs[value_to_predict]

        label_encoder = LabelEncoder().fit(y)
        logger.info(f"Map {label_encoder.classes_} to [0, 1]")

        y = label_encoder.transform(y)

        train_scaler = RobustScaler().fit(X_train)
        X = train_scaler.transform(X)

        proba = np.zeros(shape=(X.shape[0], 2))
        pred = np.zeros(shape=(X.shape[0]))

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        accuracies = []
        f1 = []
        precisions = []
        recalls = []
        r2_scores = []
        spearman_r = []
        spearman_p = []
        tot_auc = 0

        for i in range(5):  #! TODO: magic number -> move outside?
            saved_model_filename = f"model_weights.fold{i}.{sex_chromosome}.{cfg.models.train_dataset.upper()}.{name}.json"
            model.load_model(
                fname=cfg.paths.models / cfg.model_params.model_type / saved_model_filename
            )
            proba += model.predict_proba(X)
            pred_ = model.predict(X)
            pred += pred_
            accuracies.append(accuracy_score(y, pred_))
            f1.append(f1_score(y, pred_))
            precisions.append(precision_score(y, pred_))
            recalls.append(recall_score(y, pred_))
            r2_scores.append(r2_score(y, pred_))
            spearman_r.append(spearmanr(y, pred_).statistic)
            spearman_p.append(spearmanr(y, pred_).pvalue)

            viz = RocCurveDisplay.from_predictions(
                y,
                model.predict_proba(X)[:, 1],
                # ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0
            tprs.append(interp_tpr)

            tot_auc += roc_auc_score(y, model.predict_proba(X)[:, 1])

        tot_auc = tot_auc / 5
        proba = proba / 5
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0

        mean_auc = auc(mean_fpr, mean_tpr)
        mean_accuracy = np.mean(accuracies)
        mean_f1 = np.mean(f1)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_r2 = np.mean(r2_scores)
        mean_spearman = np.mean(spearman_r)

        result_dict[name][sex_chromosome]["mean_auc"] = mean_auc
        result_dict[name][sex_chromosome]["mean_accuracy"] = mean_accuracy
        result_dict[name][sex_chromosome]["mean_f1"] = mean_f1
        result_dict[name][sex_chromosome]["mean_precision"] = mean_precision
        result_dict[name][sex_chromosome]["mean_recall"] = mean_recall
        result_dict[name][sex_chromosome]["mean_r2"] = mean_r2
        result_dict[name][sex_chromosome]["mean_spearmanr"] = mean_spearman

        logger.info(f"{mean_auc=},")
        logger.info(f"{mean_accuracy=},")
        logger.info(f"{mean_f1=},")
        logger.info(f"{mean_precision=},")
        logger.info(f"{mean_recall=},")
        logger.info(f"{mean_r2=},")
        logger.info(f"{mean_spearman=},")

        proba_thresh = 0.5
        tot_accuracy = accuracy_score(y, proba[:, 1] > proba_thresh)
        tot_f1 = f1_score(y, proba[:, 1] > proba_thresh)
        tot_precision = precision_score(y, proba[:, 1] > proba_thresh)
        tot_recall = recall_score(y, proba[:, 1] > proba_thresh)
        tot_r2 = r2_score(y, proba[:, 1] > proba_thresh)
        tot_spearman = spearmanr(y, proba[:, 1] > proba_thresh)

        result_dict[name][sex_chromosome]["tot_auc"] = tot_auc
        result_dict[name][sex_chromosome]["tot_accuracy"] = tot_accuracy
        result_dict[name][sex_chromosome]["tot_f1"] = tot_f1
        result_dict[name][sex_chromosome]["tot_precision"] = tot_precision
        result_dict[name][sex_chromosome]["tot_recall"] = tot_recall
        result_dict[name][sex_chromosome]["tot_r2"] = tot_r2
        result_dict[name][sex_chromosome]["tot_spearman_r"] = tot_spearman.statistic
        result_dict[name][sex_chromosome]["tot_spearman_p"] = tot_spearman.pvalue

        logger.info(f"{tot_auc=},")
        logger.info(f"{tot_accuracy=},")
        logger.info(f"{tot_f1=},")
        logger.info(f"{tot_precision=},")
        logger.info(f"{tot_recall=},")
        logger.info(f"{tot_r2=},")
        logger.info(f"{tot_spearman.statistic=},")

    with open(
        cfg.paths.results
        / f"eval_results.{cfg.model_params.model_type}.{value_to_predict}.{cfg.models.train_dataset.upper()}.{name}.json",
        "w",
    ) as file:
        json.dump(result_dict, file)
