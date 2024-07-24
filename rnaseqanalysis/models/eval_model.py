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
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, auc
import json
from tqdm import tqdm


fdir_raw = Path("data/raw/")
fdir_processed = Path("data/interim")
fdir_traintest = Path("data/processed") / 'sex'
fdir_external = Path("data/external")
ml_models_fdir = Path("models")

use_CV = True

model_type = 'catboost'
model_type = 'xgboost'


#! SHOLD BE THE SAME AS IN train_model.py
# feature_importance_method = 'native'
feature_importance_method = 'SHAP'

# sex = 'chrXY'
# sex = 'chrX'
# sex = 'chrY'
# sex = 'autosome'

value_to_predict = 'Sex'

result_dict = {}

for organ in ['BRAIN0', "HEART", "BRAIN1"]:
    result_dict[organ] = {}
    for sex in ['chrXY', 'chrX', 'chrY', 'autosome']:
        result_dict[organ][sex] = {}
        print('```')
        print("*" * 20)
        print(organ)
        print(model_type)
        print(sex)
        print("*" * 20)

        with open(f'models/{model_type}.json', 'r') as file:
            model_params = json.load(file)

        if model_type == 'xgboost':
            model = xgb.XGBClassifier(**model_params)

        if model_type == 'catboost':
            model = CatBoostClassifier(**model_params)

        fname = next((fdir_external / organ / 'reg').glob("*processed.h5"))
        fname = fname.name

        data_eval = pd.read_hdf(fdir_external / organ / 'reg' / fname, index_col=0)
        data_eval_header = pd.read_csv(fdir_external / organ / 'reg' / 'SraRunTable.txt', sep=',')
        # data_eval_header = data_eval_header.loc(data_eval.index)
        # print(data_eval_header.columns)
        if organ == 'BRAIN1':
            print('ground true: ', (data_eval_header['gender'].values == 'male').astype(int))
        else:
            print('ground true: ', (data_eval_header['sex'].values == 'male').astype(int))

        features_fname = f"geuvadis_train_features_{sex}_calibration_{organ}.csv"
        features_list = pd.read_csv(ml_models_fdir / model_type / features_fname, index_col=0)

        data_eval = data_eval[features_list.index]

        X = data_eval.values
        if organ == 'BRAIN1':
            y = data_eval_header['gender'].values
        else:
            y = data_eval_header['sex'].values

        label_encoder = LabelEncoder().fit(y)
        print(label_encoder.classes_, "[0, 1]")

        y = label_encoder.transform(y)

        # X = StandardScaler().fit_transform(X)
        X = RobustScaler().fit_transform(X)

        proba = np.zeros(shape=(X.shape[0], 2))
        pred = np.zeros(shape=(X.shape[0]))

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        accuracies = []
        f1 = []
        precisions = []
        recalls = []

        for i in range(5):
            saved_model_filename = f"geuvadis_fold{i}_{sex}_calibration_{organ}.json"
            model.load_model(fname=ml_models_fdir / model_type / saved_model_filename)

            proba += model.predict_proba(X)
            pred_ = model.predict(X)
            # if sex == 'autosome':
            #     pred_ = np.abs(pred_ - 1)
            pred += pred_

            accuracies.append(accuracy_score(y, pred_))
            f1.append(f1_score(y, pred_))
            precisions.append(precision_score(y, pred_))
            recalls.append(recall_score(y, pred_))

            viz = RocCurveDisplay.from_predictions(
                y, model.predict_proba(X)[:, 1],
                # ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0
            tprs.append(interp_tpr)

        proba = proba / 5
        # pred = pred / 5
        if sex == 'autosome':
            print('predicted:   ', (proba[:, 1] > 0.5).astype(int))
        else:
            print('predicted:   ', (proba[:, 1] > 0.5).astype(int))
        # print(pred.astype(int))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0

        mean_auc = auc(mean_fpr, mean_tpr)
        mean_accuracy = np.mean(accuracies)
        mean_f1 = np.mean(f1)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)

        result_dict[organ][sex]["mean_auc"] = mean_auc
        result_dict[organ][sex]["mean_accuracy"] = mean_accuracy
        result_dict[organ][sex]["mean_f1"] = mean_f1
        result_dict[organ][sex]["mean_precision"] = mean_precision
        result_dict[organ][sex]["mean_recall"] = mean_recall

        print("-" * 20)
        print(f"{mean_auc=},")
        print(f"{mean_accuracy=},")
        print(f"{mean_f1=},")
        print(f"{mean_precision=},")
        print(f"{mean_recall=},")
        print("-" * 20)
        print('```')

with open(f'reports/eval_result_{value_to_predict}_{model_type}.json', 'w') as file:
    json.dump(result_dict, file)
