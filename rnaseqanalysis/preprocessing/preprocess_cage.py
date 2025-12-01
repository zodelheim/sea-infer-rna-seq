"""Alternative to make_train_dataset.py but for CAGE dataset"""

import pandas as pd
import numpy as np
from pathlib import Path
import anndata as ad

from tqdm import tqdm

# from make_eval_dataset import logarithmization, filter_zero_median
from make_train_dataset import (
    logarithmization,
    filter_zero_median,
    filter_cv_threshold,
    filter_median_q34,
    filter_correlated,
    filter_cv_q34,
    split_by_sex_transcripts,
)

from config import FDIR_EXTERNAL, FDIR_INTEMEDIATE, FDIR_PROCESSED


def parse_promoter_enchancer(adata: ad.AnnData) -> ad.AnnData:
    transcript_class = adata.var["class"]

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
    highly_expressed = pd.concat(highly_expressed, axis=1)
    data = pd.concat([data, highly_expressed], axis=1)

    return data


def label_unique(data: ad.AnnData) -> ad.AnnData:
    columns_wclass = adata.var["transcript_id"].astype(str) + "_" + adata.var["class"].astype(str)
    duplicated = columns_wclass[columns_wclass.duplicated()]
    truely_unique = (~columns_wclass.duplicated()).values

    adata_mean = adata.X.mean(axis=0)
    for dd in tqdm(duplicated):
        dupl_ids = np.argwhere(columns_wclass == dd)
        largest_mean_id = np.argmax(adata_mean[dupl_ids.ravel()])
        truely_unique[dupl_ids[largest_mean_id].item()] = True

    adata.varm["unique"] = truely_unique

    return adata


for organ in ["HEART"]:
    fdir = FDIR_EXTERNAL / organ / "CAGE"

adata = ad.read_h5ad(FDIR_INTEMEDIATE / f"CAGE.{organ}.raw.h5ad")

#! if no filtering
adata.layers["raw"] = adata.X.copy()
adata.X = logarithmization(adata.to_df()).values


adata.var["seqname"] = adata.var["seqnames"]
adata.var["transcript_id"] = adata.var.index

adata = label_unique(adata)

columns_new = [(col, str(i).zfill(5)) for i, col in enumerate(adata.var_names)]
adata.var_names = [str(i).zfill(5) for i, col in enumerate(adata.var_names)]
# genes_annot.index = [str(i).zfill(5) for i, col in enumerate(adata.var_names)]

adata = split_by_sex_transcripts(adata, drop_duplicates=False)


#! if filtering
"""
genes_annot['seqname'] = genes_annot['seqnames']
genes_annot['transcript_id'] = genes_annot.index

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

# data_XY, data_X, data_Y, data_autosomes = remove_sex_transcripts(data, genes_annot)

# genes_annot = genes_annot.loc[data.columns]
# samples_annot = samples_annot.loc[data.index]
"""

adata.write(FDIR_PROCESSED / "sex" / "CAGE.HEART.preprocessed.sex.h5ad")

# adata = ad.AnnData(X=data, obs=samples_annot, var=genes_annot)
# adata.write(FDIR_EXTERNAL / organ / 'CAGE' / 'data.processed.h5ad')

# data.to_hdf(FDIR_INTEMEDIATE / f'CAGE.heart.preprocessed.h5', key="data", format='f')
# samples_annot.to_hdf(FDIR_INTEMEDIATE / 'CAGE.heart.preprocessed.h5', key="header", format='f')
# genes_annot.to_hdf(FDIR_INTEMEDIATE / 'CAGE.heart.preprocessed.h5', key="gtf", format='table')

# data_Y.to_hdf(FDIR_PROCESSED / "sex" / 'CAGE.heart.preprocessed.sex.h5', key='chrY', format='f')
# data_X.to_hdf(FDIR_PROCESSED / "sex" / 'CAGE.heart.preprocessed.sex.h5', key='chrX', format='f')
# data_autosomes.to_hdf(FDIR_PROCESSED / "sex" / 'CAGE.heart.preprocessed.sex.h5', key='autosome', format='f')
# data_XY.to_hdf(FDIR_PROCESSED / "sex" / 'CAGE.heart.preprocessed.sex.h5', key='chrXY', format='f')
