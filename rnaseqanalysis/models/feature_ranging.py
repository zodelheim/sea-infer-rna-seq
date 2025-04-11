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
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
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
import json
import cupy
import anndata as ad
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from config import FDIR_EXTERNAL, FDIR_RAW, FDIR_PROCESSED, FDIR_INTEMEDIATE


from tqdm import tqdm

fdir_raw = FDIR_RAW
fdir_intermediate = FDIR_INTEMEDIATE
fdir_processed = FDIR_PROCESSED / "sex"
fdir_external = FDIR_EXTERNAL


use_CV = True

model_type = "catboost"
model_type = "xgboost"
# model_type = 'random_forest'

Scaler = RobustScaler
# Scaler = StandardScaler

# sex = 'chrXY'
# sex = 'autosome'

feature_importance_method = "native"
feature_importance_method = "SHAP"

n_threads = 6

value_to_predict = "sex"

filename_prefixes = {
    "None": "geuvadis",
    "CAGE.HEART": "CAGE.HEART",
    "BRAIN0": "BRAIN0",
    "HEART": "HEART",
    "BRAIN1": "BRAIN1",
}

drop_duplicates = False
# drop_duplicates = True

# value_to_predict = 'Experimental_Factor:_population (exp)'

# for organ in ['BRAIN1']:
for organ in ["BRAIN0", "HEART", "BRAIN1", "None"]:
    # for organ in ["CAGE.HEART"]:
    # for organ in ['None']:
    # for sex_chromosome in ['chrXY']:
    for sex_chromosome in ["chr_aXY", "autosomes", "chr_aX", "chr_aY"]:
        with open(f"models/model_params.json", "r") as file:
            model_params = json.load(file)[model_type]
        model_params = model_params[value_to_predict]

        adata = ad.read_h5ad(
            fdir_processed
            / f"{filename_prefixes[organ].upper()}.preprocessed.{value_to_predict}.h5ad"
        )
        adata = adata[:, adata.varm[sex_chromosome]]

        if drop_duplicates:
            adata = adata[:, adata.varm["unique"]]

        if organ not in ["None", "CAGE.HEART"]:
            fname = next((fdir_external / organ / "reg").glob("*processed.h5"))
            fname = fname.name

            data_eval = pd.read_hdf(fdir_external / organ / "reg" / fname, index_col=0)
            adata = adata[:, adata.var_names.intersection(data_eval.columns)]
        # --------------------------------------------------------------------------------
        X = adata.X
        y = adata.obs[value_to_predict]

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

        # cv = StratifiedKFold(n_splits=5)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

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

            train_scaler = Scaler().fit(X_train)
            test_scaler = Scaler().fit(X_test)

            X_train = train_scaler.transform(X_train)
            X_test = test_scaler.transform(X_test)

            X_train_ = X_train
            y_train_ = y_train
            X_val = X_test
            y_val = y_test

            if model_type == "xgboost":
                if y.max() > 1:
                    model_params["objective"] = "multi:softmax"
                    # model_params['num_class'] = y.max() + 1

                model = xgb.XGBClassifier(**model_params)
                model.fit(cupy.array(X_train_), y_train_, eval_set=[(X_val, y_val)], verbose=False)

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

            pred = model.predict(X_test_c)
            pred_prob = model.predict_proba(X_test_c)

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

            print(sex_chromosome)
            print("-" * 20)
            print(f"{mean_auc=}")
            print(f"{mean_accuracy=}")
            print(f"{mean_f1=}")
            print(f"{mean_precision=}")
            print(f"{mean_recall=}")
            print("-" * 20)

        feature_importance_df = feature_importance_df.sort_values(by="SHAP", ascending=False)
        print("features by SHAP")
        print(feature_importance_df.iloc[:n_features_to_print])

        feature_importance_df = feature_importance_df.sort_values(by="native", ascending=False)
        print("features by model.feature_importances_")
        print(feature_importance_df.iloc[:n_features_to_print])

        # feature_importance_df.to_csv(fdir_processed / f'feature_importance.{model_type}.{sex}.csv')
        feature_importance_df.to_hdf(
            fdir_intermediate
            / f"feature_importance.{model_type}.{value_to_predict}.organ_{organ}.h5",
            key=f"{sex_chromosome}",
            format="f",
        )

        print("\n")
        print(model_type)
        print(model_params)
        print("\n")
