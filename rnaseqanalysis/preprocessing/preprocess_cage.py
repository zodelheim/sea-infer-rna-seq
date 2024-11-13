""" Alternative to make_train_dataset.py but for CAGE dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import anndata as ann

from tqdm import tqdm
# from make_eval_dataset import logarithmization, filter_zero_median
from make_train_dataset import (logarithmization,
                                filter_zero_median,
                                filter_cv_threshold,
                                filter_median_q34,
                                filter_sex_correlated,
                                filter_cv_q34,
                                remove_sex_transcripts)

from rnaseqanalysis.config import FDIR_EXTERNAL


def parse_promoter_enchancer(data: pd.DataFrame, genes_annot: pd.DataFrame) -> pd.DataFrame:

    transcript_class = genes_annot['class']

    new_column_names = []
    for i in range(len(data.columns)):
        new_column_names.append(f"{data.columns[i]}_{transcript_class.iloc[i]}")
    data.columns = new_column_names

    return data


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:

    duplicated_tr = data.columns[data.columns.duplicated()].unique()

    data_mean = data.mean()
    highly_expressed = []

    for tr in tqdm(duplicated_tr):
        highly_expressed.append(data[tr].iloc[:, data_mean[tr].argmax()].copy())
    data.drop(columns=duplicated_tr, inplace=True)
    highly_expressed = pd.concat(highly_expressed)
    data = pd.concat([data, highly_expressed])

    return data


fdir_external = FDIR_EXTERNAL

for organ in ["HEART"]:
    fdir = fdir_external / organ / 'CAGE'

data_raw = pd.read_csv((fdir / "TPM batch corrected PLS ELS.txt"), sep='\t').T
samples_annot = pd.read_excel(fdir / 'Metadata_ERytkin Edits10072024 age request.xlsx',
                              parse_dates=False,)
samples_annot.set_index('samples', inplace=True)
samples_annot['donor'] = samples_annot['donor'].astype(str)

genes_annot = pd.read_csv(fdir / 'ANNOT.csv')


samples_names = data_raw.index.intersection(samples_annot.index)
data_raw = data_raw.loc[samples_names]
samples_annot = samples_annot.loc[samples_names]

data_raw.columns = genes_annot['transcriptId']
genes_annot.set_index('transcriptId', inplace=True)

data = filter_zero_median(data_raw)

duplicated_columns = data.columns[data.columns.duplicated()]

data = parse_promoter_enchancer(data, genes_annot)
data = drop_duplicates(data)

data = filter_sex_correlated(data, samples_annot['sex'].loc[data.index])
data = logarithmization(data)
data = filter_cv_threshold(data, 0.7)

data = filter_median_q34(data)
data = filter_cv_q34(data)
data = data.astype(np.float32)

data_XY, data_X, data_Y, data_autosomes = remove_sex_transcripts(data, genes_annot)

gtf_data = gtf_data.loc[data.columns]
data_header = data_header.loc[data.index]


adata = ann.AnnData(X=data, obs=samples_annot, var=genes_annot)

adata.write(fdir / 'data.processed.h5ad')
