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
from tqdm import tqdm
import anndata as ad
from loguru import logger

from sklearn.neighbors import KNeighborsClassifier
from config import FDIR_EXTERNAL, FDIR_INTEMEDIATE, FDIR_PROCESSED, FDIR_RAW


# mlflow.set_tracking_uri(uri="http://localhost:8080")


fdir_raw = FDIR_RAW
fdir_intermediate = FDIR_INTEMEDIATE
fdir_processed = FDIR_PROCESSED / "sex"
fdir_external = FDIR_EXTERNAL
ml_models_fdir = Path("models")

use_CV = True

model_type = "catboost"
model_type = "xgboost"
# model_type = 'random_forest'
# model_type = 'knn'


feature_importance_method = "native"
feature_importance_method = "SHAP"

organ_names = {
    "BRAIN0": "BRAIN0",
    "HEART": "HEART",
    "BRAIN1": "BRAIN1",
    "None": "BLOOD",
    "CAGE.HEART": "CAGE.HEART",
}

value_to_predict = "sex"
# value_to_predict = 'population'

result_dict = {}

n_featues_dict = {
    "BRAIN0": {
        "chr_aXY": 2,  # 1
        "chr_aX": 1,  # 1
        "chr_aY": 1,
        "autosomes": 46,  # 40
    },
    "HEART": {
        "chr_aXY": 3,  # 2
        "chr_aX": 2,
        "chr_aY": 1,
        "autosomes": 22,  # 33
    },
    "BRAIN1": {
        "chr_aXY": 2,  # 2
        "chr_aX": 1,  # 2
        "chr_aY": 1,
        "autosomes": 29,  # 12
    },
    "None": {
        "chr_aXY": 1,  # 2, #3
        "chr_aX": 1,
        "chr_aY": 1,
        "autosomes": 40,  # 18
    },
    "CAGE.HEART": {
        "chr_aXY": 6,  # 10
        "chr_aX": 9,
        "chr_aY": 9,
        "autosomes": 20,
    },
}

filename_prefixes = {
    "None": "geuvadis",
    "CAGE.HEART": "CAGE.HEART",
    "BRAIN0": "BRAIN0",
    "HEART": "HEART",
    "BRAIN1": "BRAIN1",
}


features_shapsumm_threshold = 45

save_features = True
save_results = True

drop_duplicates = False

with open(f"models/model_params.json", "r") as file:
    model_params = json.load(file)[model_type]
model_params = model_params[value_to_predict]
# model_params['max_depth'] = 7

if model_type == "xgboost":
    model = xgb.XGBClassifier(**model_params)

for organ in ["None", "BRAIN0", "BRAIN1", "HEART"]:
    # for organ in ["CAGE.HEART"]:
    # for organ in ["BRAIN1"]:
    if organ in ["CAGE.HEART"]:
        Scaler = StandardScaler
    else:
        Scaler = RobustScaler

    result_dict[organ] = {}

    fig_cm, axs_cm = plt.subplots(2, 2)
    cbar_ax1 = fig_cm.add_axes([0.84, 0.25, 0.015, 0.6], in_layout=True)
    cbar_ax2 = fig_cm.add_axes([0.92, 0.25, 0.015, 0.6], in_layout=True)

    for sex_chromosome, ax_cm in zip(["chr_aXY", "chr_aX", "chr_aY", "autosomes"], axs_cm.flat):
        # for sex_chromosome in ["chr_aX"]:
        result_dict[organ][sex_chromosome] = {}

        logger.info("*" * 20)
        logger.info(organ)
        logger.info(model_type)
        logger.info(sex_chromosome)
        logger.info("*" * 20)

        # n_features = 0
        # n_features = n_featues_dict[organ][sex_chromosome]
        n_features = 50

        if organ in ["CAGE.HEART"]:
            adata = ad.read_h5ad(
                fdir_processed / f"CAGE.HEART.preprocessed.{value_to_predict}.h5ad"
            )
        else:
            adata = ad.read_h5ad(fdir_processed / f"GEUVADIS.preprocessed.{value_to_predict}.h5ad")

        adata = adata[:, adata.varm[sex_chromosome]]

        if drop_duplicates:
            adata = adata[:, adata.varm["unique"]]

        features = pd.read_hdf(
            fdir_intermediate
            / f"feature_importance.{model_type}.{value_to_predict}.organ_{organ}.h5",
            key=f"{sex_chromosome}",
        )

        features = features[feature_importance_method]
        features = features.sort_values(ascending=False)

        if organ not in ["None", "CAGE.HEART"]:
            #! RAW DATA
            # fname = next((fdir_external / organ / "reg").glob("*processed.h5"))
            # fname = fname.name
            # data_eval = pd.read_hdf(fdir_external / organ / "reg" / fname, index_col=0)
            #! PREPROCESSED DATA
            data_eval = ad.read_h5ad(
                fdir_processed / f"{filename_prefixes[organ]}.preprocessed.{value_to_predict}.h5ad"
            ).to_df()

            features = features.loc[features.index.intersection(data_eval.columns)]

            if n_features != 0:
                features = features.sort_values(ascending=False)
                # logger.info(features.iloc[:n_features])
            else:
                logger.info(features.shape)

        features_fname = (
            f"{filename_prefixes[organ]}_train_features_{sex_chromosome}_calibration_{organ}.csv"
        )

        if n_features != 0:
            if save_features:
                features_list = features.iloc[:n_features]
            else:
                features_list = pd.read_csv(
                    ml_models_fdir / model_type / features_fname, dtype=object
                )
                features_list.set_index("Feature", inplace=True)
        else:
            features_list = features.loc[features >= features_shapsumm_threshold]

            #! CHECK: use by threshold
            # n_features = len(features_list)
            #! CHECK: OR use max of!!!
            n_features = max(len(features_list), n_featues_dict[organ][sex_chromosome])

            features_list = features.iloc[:n_features]
        logger.info(f"{n_features=}")

        if save_features:
            features_list.to_csv(ml_models_fdir / model_type / features_fname)

        adata = adata[:, features_list.index]

        X = adata.X
        y = adata.obs[value_to_predict]

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
        logger.info(f"{label_encoder.classes_}, [0, 1]")

        # logger.info(dir(label_encoder))
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

            train_scaler = Scaler().fit(X_train)
            X_train = train_scaler.transform(X_train)
            X_test = train_scaler.transform(X_test)

            X_train_ = X_train
            y_train_ = y_train
            X_val = X_test
            y_val = y_test

            if model_type == "xgboost":
                # model = xgb.XGBClassifier(**model_params)
                model.fit(cupy.array(X_train_), y_train_, eval_set=[(X_val, y_val)], verbose=False)

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

            if model_type == "knn":
                model = KNeighborsClassifier(**model_params)
                model.fit(X_train_, y_train_)

            if not (ml_models_fdir / model_type).is_dir():
                (ml_models_fdir / model_type).mkdir()

            pred = model.predict(cupy.array(X_test))
            pred_prob = model.predict_proba(cupy.array(X_test))

            preds[val] = pred
            preds_proba[val] = pred_prob[:, 1]

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

            if save_results:
                saved_model_filename = (
                    f"{filename_prefixes[organ]}_fold{i}_{sex_chromosome}_calibration_{organ}.json"
                )
                if model_type != "knn":
                    model.save_model(fname=ml_models_fdir / model_type / saved_model_filename)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0

        mean_auc = auc(mean_fpr, mean_tpr)
        mean_accuracy = np.mean(accuracies)
        mean_f1 = np.mean(f1)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)

        logger.info("-" * 20)
        logger.info("-" * 20)
        logger.info(f"{mean_auc=},")
        logger.info(f"{mean_accuracy=},")
        logger.info(f"{mean_f1=},")
        logger.info(f"{mean_precision=},")
        logger.info(f"{mean_recall=},")
        logger.info("-" * 20)

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
        logger.info("-" * 20)

        result_dict[organ][sex_chromosome]["mean_auc"] = total_auc
        result_dict[organ][sex_chromosome]["mean_accuracy"] = total_accuracy
        result_dict[organ][sex_chromosome]["mean_f1"] = total_f1
        result_dict[organ][sex_chromosome]["mean_precision"] = total_precision
        result_dict[organ][sex_chromosome]["mean_recall"] = total_recall
        result_dict[organ][sex_chromosome]["n_features"] = n_features

        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            lw=2,
            alpha=0.8,
        )
        # plt.title(f"{model_type}, {sex_chromosome}, {organ}")
        ax.set_title(f"{model_type}, {sex_chromosome}, {organ}")
        if save_results:
            fig.savefig(
                f"reports/figures/{filename_prefixes[organ]}_{sex_chromosome}_organ_{organ}_ROC.png",
                dpi=300,
            )
            plt.close()
        else:
            fig.show()

        # if save_results:
        # _ = ConfusionMatrixDisplay.from_predictions(y, preds, display_labels=['F', "M"])
        # plt.title(f"{model_type}, {sex}, {organ}")

        mask_diag = np.eye(2, 2, dtype=bool)
        cm = confusion_matrix(y, preds)

        true_total_1 = np.sum(cm[0])
        true_total_2 = np.sum(cm[1])
        cm_ = cm.copy().astype(np.float32)

        cm_[0] = cm_[0] / true_total_1.item() * 100
        cm_[1] = cm_[1] / true_total_2.item() * 100

        cm_anno = [[], []]

        cm_anno[0] = [
            f"{cm[0, 0]} \n ({round(cm_[0, 0])}%)",
            f"{cm[0, 1]} \n ({round(cm_[0, 1])}%)",
        ]
        cm_anno[1] = [
            f"{cm[1, 0]} \n ({round(cm_[1, 0])}%)",
            f"{cm[1, 1]} \n ({round(cm_[1, 1])}%)",
        ]

        sns.heatmap(
            cm,
            annot=cm_anno,
            cmap="Blues",
            ax=ax_cm,
            square=True,
            vmin=0,
            vmax=y.shape[0],
            fmt="",
            mask=~mask_diag,
            cbar_kws={"orientation": "vertical", "format": "%1i"},
            cbar_ax=cbar_ax1 if sex_chromosome == "chr_aXY" else None,
            cbar=sex_chromosome == "chr_aXY",
            annot_kws=dict(ha="center"),
        )
        sns.heatmap(
            cm,
            annot=cm_anno,
            cmap="Reds",
            ax=ax_cm,
            square=True,
            vmin=0,
            vmax=y.shape[0],
            fmt="",
            mask=mask_diag,
            cbar_kws={"orientation": "vertical", "format": "%1i"},
            cbar_ax=cbar_ax2 if sex_chromosome == "chr_aXY" else None,
            cbar=sex_chromosome == "chr_aXY",
            annot_kws=dict(ha="center"),
        )
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("True label")
        ax_cm.xaxis.set_ticklabels(
            ["Female", "Male"],
        )
        ax_cm.yaxis.set_ticklabels(["Female", "Male"], rotation=0)
        ax_cm.set_title(f"{sex_chromosome}")

    fig_cm.tight_layout(rect=(0, 0, 0.87, 1))
    if save_results:
        fig_cm.savefig(f"reports/figures/geuvadis_cm_{organ}.png", dpi=300)
        plt.close()
    else:
        fig_cm.show()


if save_results:
    with open(f"reports/train_result_{value_to_predict}_{model_type}.json", "w") as file:
        json.dump(result_dict, file)
else:
    print("RESULT:")
    print(result_dict)
