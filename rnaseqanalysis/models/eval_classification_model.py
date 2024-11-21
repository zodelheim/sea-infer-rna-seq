import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, auc
import json
from tqdm import tqdm
import anndata as ad


from config import FDIR_EXTERNAL, FDIR_INTEMEDIATE, FDIR_PROCESSED, FDIR_RAW

fdir_raw = FDIR_RAW
fdir_intermediate = FDIR_INTEMEDIATE
fdir_processed = FDIR_PROCESSED / 'sex'
fdir_external = FDIR_EXTERNAL
ml_models_fdir = Path("models")

organ_names = {
    'BRAIN0': "BRAIN0",
    "HEART": "HEART",
    "BRAIN1": "BRAIN1",
    'None': "BLOOD"
}

use_CV = True

model_type = 'catboost'
model_type = 'xgboost'
# model_type = 'random_forest'

save_results = True

tissue_specific = False
tissue_name = 'Nac'

#! SHOLD BE THE SAME AS IN train_model.py
# feature_importance_method = 'native'
feature_importance_method = 'SHAP'

# sex = 'chrXY'
# sex = 'chrX'
# sex = 'chrY'
# sex = 'autosome'

value_to_predict = 'sex'

result_dict = {}

for organ in ["BRAIN1"]:
    # for organ in ['BRAIN0', "HEART", "BRAIN1"]:
    result_dict[organ] = {}
    fig, axs = plt.subplots(2, 2)
    # plt.subplots_adjust(bottom=0.2)
    cbar_ax1 = fig.add_axes([.84, .25, .015, .6], in_layout=True)
    cbar_ax2 = fig.add_axes([.92, .25, .015, .6], in_layout=True)

    for sex_chromosomes, ax in zip(['chr_aXY', 'chr_aX', 'chr_aY', 'autosomes'], axs.flat):
        # for sex in ['chrXY']:
        result_dict[organ][sex_chromosomes] = {}
        print('```')
        print("*" * 20)
        print(organ)
        print(model_type)
        print(sex_chromosomes)
        print("*" * 20)

        with open(f'models/model_params.json', 'r') as file:
            model_params = json.load(file)[model_type]

        if model_type == 'xgboost':
            model = xgb.XGBClassifier(**model_params)

        if model_type == 'catboost':
            model = CatBoostClassifier(**model_params)

        fname = next((fdir_external / organ / 'reg').glob("*processed.h5"))
        fname = fname.name

        data_eval = pd.read_hdf(fdir_external / organ / 'reg' / fname, index_col=0)
        data_eval_header = pd.read_csv(fdir_external / organ / 'reg' / 'SraRunTable.txt', sep=',')

        if tissue_specific:
            data_eval_header.set_index('Run', inplace=True)
            tissue_index = data_eval_header.loc[data_eval_header['tissue'] != tissue_name].index
            data_eval = data_eval.loc[tissue_index]
            data_eval_header = data_eval_header.loc[tissue_index]

        # if organ == 'BRAIN1':
        #     print('ground true: ', (data_eval_header['gender'].values == 'male').astype(int))
        # else:
        #     print('ground true: ', (data_eval_header['sex'].values == 'male').astype(int))

        features_fname = f"geuvadis_train_features_{sex_chromosomes}_calibration_{organ}.csv"
        features_list = pd.read_csv(ml_models_fdir / model_type / features_fname, index_col=0)
        print(f"{len(features_list)=}")
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

        tot_auc = 0

        for i in range(5):
            saved_model_filename = f"geuvadis_fold{i}_{sex_chromosomes}_calibration_{organ}.json"
            model.load_model(fname=ml_models_fdir / model_type / saved_model_filename)

            proba += model.predict_proba(X)
            pred_ = model.predict(X)
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
            plt.close()

            tot_auc += roc_auc_score(y, model.predict_proba(X)[:, 1])

        tot_auc = tot_auc / 5
        proba = proba / 5
        # pred = pred / 5
        # print('predicted_pr:  ', (proba[:, 1] > proba_thresh).astype(int))
        # print('predicted:     ', (pred > 5 / 2).astype(int))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0

        mean_auc = auc(mean_fpr, mean_tpr)
        mean_accuracy = np.mean(accuracies)
        mean_f1 = np.mean(f1)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)

        result_dict[organ][sex_chromosomes]["mean_auc"] = mean_auc
        result_dict[organ][sex_chromosomes]["mean_accuracy"] = mean_accuracy
        result_dict[organ][sex_chromosomes]["mean_f1"] = mean_f1
        result_dict[organ][sex_chromosomes]["mean_precision"] = mean_precision
        result_dict[organ][sex_chromosomes]["mean_recall"] = mean_recall

        print("-" * 20)
        print(f"{mean_auc=},")
        print(f"{mean_accuracy=},")
        print(f"{mean_f1=},")
        print(f"{mean_precision=},")
        print(f"{mean_recall=},")
        print("-" * 20)
        print('```')

        proba_thresh = 0.5

        tot_accuracy = (accuracy_score(y, proba[:, 1] > proba_thresh))
        tot_f1 = (f1_score(y, proba[:, 1] > proba_thresh))
        tot_precision = (precision_score(y, proba[:, 1] > proba_thresh))
        tot_recall = (recall_score(y, proba[:, 1] > proba_thresh))

        # result_dict[organ][sex]["tot_auc"] = tot_auc
        # result_dict[organ][sex]["tot_accuracy"] = tot_accuracy
        # result_dict[organ][sex]["tot_f1"] = tot_f1
        # result_dict[organ][sex]["tot_precision"] = tot_precision
        # result_dict[organ][sex]["tot_recall"] = tot_recall

        # print("-" * 20)
        # print(f"{tot_auc=},")
        # print(f"{tot_accuracy=},")
        # print(f"{tot_f1=},")
        # print(f"{tot_precision=},")
        # print(f"{tot_recall=},")
        # print("-" * 20)
        # print('```')

        mask_diag = np.eye(2, 2, dtype=bool)
        cm = confusion_matrix(y, proba[:, 1] > proba_thresh)
        true_total_1 = np.sum(cm[0])
        true_total_2 = np.sum(cm[1])
        cm_ = cm.copy().astype(np.float32)

        cm_[0] = cm_[0] / true_total_1.item() * 100
        cm_[1] = cm_[1] / true_total_2.item() * 100

        cm_anno = [[], []]

        cm_anno[0] = [f"{cm[0, 0]} \n ({round(cm_[0, 0])}%)", f"{cm[0, 1]} \n ({round(cm_[0, 1])}%)"]
        cm_anno[1] = [f"{cm[1, 0]} \n ({round(cm_[1, 0])}%)", f"{cm[1, 1]} \n ({round(cm_[1, 1])}%)"]

        # cm_anno = np.array(cm_anno)

        sns.heatmap(cm, annot=cm_anno, cmap='Blues', ax=ax, square=True,
                    vmin=0, vmax=len(y), fmt='',
                    mask=~mask_diag,
                    cbar_kws={'orientation': 'vertical', 'format': "%1i"},
                    cbar_ax=cbar_ax1 if sex_chromosomes == 'chrXY' else None,
                    cbar=sex_chromosomes == 'chrXY',
                    annot_kws=dict(ha='center'),
                    )
        sns.heatmap(cm, annot=cm_anno, cmap='Reds', ax=ax, square=True,
                    vmin=0, vmax=len(y), fmt='',
                    mask=mask_diag,
                    cbar_kws={'orientation': 'vertical', 'format': "%1i"},
                    cbar_ax=cbar_ax2 if sex_chromosomes == 'chrXY' else None,
                    cbar=sex_chromosomes == 'chrXY',
                    annot_kws=dict(ha='center'),
                    )
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.xaxis.set_ticklabels(['Female', 'Male'], )
        ax.yaxis.set_ticklabels(['Female', 'Male'], rotation=0)
        ax.set_title(f"{sex_chromosomes}")

        # plt.title(f"{model_type}, {sex}, {organ}")
        # plt.tight_layout()

    fig.tight_layout(rect=(0, 0, .87, 1))
    if save_results:
        plt.savefig(f'reports/figures/cm_{organ}.png', dpi=300)
        plt.close()
    else:
        plt.show()
    # exit()
    # plt.show()

if save_results:
    with open(f'reports/eval_result_{value_to_predict}_{model_type}.json', 'w') as file:
        json.dump(result_dict, file)
