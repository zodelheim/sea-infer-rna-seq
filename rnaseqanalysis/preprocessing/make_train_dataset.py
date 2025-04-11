import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from scipy.stats import pointbiserialr, pearsonr, spearmanr

from tqdm import tqdm
from gtfparse import read_gtf
from prefect import flow, task
import anndata as ad
from loguru import logger

from config import FDIR_EXTERNAL, FDIR_RAW, FDIR_PROCESSED, FDIR_INTEMEDIATE


@task(log_prints=True, description="drop zero median transcripts")
def filter_zero_median(df: pd.DataFrame) -> pd.DataFrame:
    df_median = df.median()
    if (df_median == 0).any():
        cols_to_drop = df.columns[df_median == 0]
        # print(len(cols_to_drop),
        #       " features will be removed, due to a zero median value")
        df = df.drop(columns=cols_to_drop)
        # print("Current dataset size: ", df.shape)
        print("Dataset shape: ", df.shape)
        return df

    # print("Zero median columns aren't found")
    print("Dataset shape: ", df.shape)

    return df


@task(log_prints=True, description="drop correlated transcripts")
def filter_correlated(X: pd.DataFrame, y: pd.DataFrame | pd.Series, threshold=0.8) -> pd.DataFrame:
    X_corr = X
    y_corr = y
    y_encoded = LabelEncoder().fit_transform(y_corr.values)

    if len(np.unique(y_encoded)) == 2:
        corr_function = pointbiserialr
    else:
        corr_function = spearmanr

    columns_to_drop = []
    for c in tqdm(X_corr.columns):
        corr, pvalue = corr_function(X_corr[c], y_encoded)
        if np.abs(corr) > threshold:
            columns_to_drop.append(c)

    X = X.drop(columns=columns_to_drop)
    print("Dataset shape: ", X.shape)
    return X


@task(log_prints=True, description="logarithmization (log2(x+1))")
def logarithmization(df: pd.DataFrame):
    df = np.log2(df + 1)
    return df


@task(log_prints=True, description="drop transcripts with cv < threshold")
def filter_cv_threshold(df: pd.DataFrame, threshold: float):
    cv = df.std() / df.mean()
    low_cv_cols = cv[cv < threshold].index

    if len(low_cv_cols) > 0:
        # print(f"{len(low_cv_cols)} features have coefficient of variation below {threshold} and will be removed.")
        df = df.drop(columns=low_cv_cols)
    # else:
    #     print("No features found with coefficient of variation below the threshold.")
    # print(f"Current amount of features is {len(df.columns)}")

    print("Dataset shape: ", df.shape)
    return df


@task(log_prints=True, description="filter transcripts with mean < median")
def filter_median_q34(data: pd.DataFrame):
    mean = data.mean(axis=0)
    median = mean.median()
    data = data.loc[:, mean > median]
    print("Dataset shape: ", data.shape)
    return data


@task(log_prints=True, description="filter transcripts with cv mean < cv median")
def filter_cv_q34(data: pd.DataFrame):
    cv = data.std() / data.mean()
    median_cv = cv.median()
    data = data.loc[:, cv > median_cv]
    print("Dataset shape: ", data.shape)
    return data


@task(log_prints=True, description="extract Sex (chrX, chrY) transcripts from gtf")
def locate_sex_transcripts(gtf_data: pd.DataFrame, drop_duplicates) -> tuple[pd.Series, pd.Series]:
    # from https://www.ensembl.org/info/genome/genebuild/human_PARS.html

    pseudoautosoms_Y1 = [10001, 2781479]
    pseudoautosoms_X1 = [10001, 2781479]
    pseudoautosoms_Y2 = [56887903, 57217415]
    pseudoautosoms_X2 = [155701383, 156030895]

    transcripts_x = gtf_data.loc[gtf_data["seqname"] == "chrX"]
    transcripts_y = gtf_data.loc[gtf_data["seqname"] == "chrY"]

    true_transcripts_x = transcripts_x.loc[
        (
            (transcripts_x["end"] < pseudoautosoms_X1[0])
            | (
                (transcripts_x["start"] > pseudoautosoms_X1[1])
                & (transcripts_x["end"] < pseudoautosoms_X2[0])
            )
            | (transcripts_x["start"] > pseudoautosoms_X2[1])
        )
    ]

    true_transcripts_y = transcripts_y.loc[
        (
            (transcripts_y["end"] < pseudoautosoms_Y1[0])
            | (
                (transcripts_y["start"] > pseudoautosoms_Y1[1])
                & (transcripts_y["end"] < pseudoautosoms_Y2[0])
            )
            | (transcripts_y["start"] > pseudoautosoms_Y2[1])
        )
    ]

    # transcripts_x = transcripts_x['transcript_id'].unique()
    # transcripts_y = transcripts_y['transcript_id'].unique()
    if drop_duplicates:
        true_transcripts_x = true_transcripts_x["transcript_id"].unique()
        true_transcripts_y = true_transcripts_y["transcript_id"].unique()
    else:
        true_transcripts_x = true_transcripts_x.index
        true_transcripts_y = true_transcripts_y.index

    return true_transcripts_x, true_transcripts_y


@task(log_prints=True, description="drop sex (chrX, chrY) transcripts from data")
def split_by_sex_transcripts(adata: ad.AnnData, drop_duplicates=True) -> ad.AnnData:
    transcripts_x, transcripts_y = locate_sex_transcripts(adata.var, drop_duplicates)

    transcripts_x = transcripts_x.tolist()
    transcripts_y = transcripts_y.tolist()

    transcripts_x = adata.var_names.intersection(transcripts_x)
    transcripts_y = adata.var_names.intersection(transcripts_y)

    transcripts_autosomes = adata.var[
        (adata.var["seqname"] != "chrX") & (adata.var["seqname"] != "chrY")
    ].index

    data_aXY = pd.Series(np.zeros(adata.n_vars, dtype=bool), index=adata.var_names)
    data_aX = pd.Series(np.zeros(adata.n_vars, dtype=bool), index=adata.var_names)
    data_aY = pd.Series(np.zeros(adata.n_vars, dtype=bool), index=adata.var_names)
    data_autosomes = pd.Series(np.zeros(adata.n_vars, dtype=bool), index=adata.var_names)

    data_aXY[:] = True
    data_aX[transcripts_x.union(transcripts_autosomes)] = True
    data_aY[transcripts_y.union(transcripts_autosomes)] = True
    data_autosomes[transcripts_autosomes] = True

    adata.varm["chr_aXY"] = data_aXY.values
    adata.varm["chr_aX"] = data_aX.values
    adata.varm["chr_aY"] = data_aY.values
    adata.varm["autosomes"] = data_autosomes.values

    print("dataXY shape: ", adata.varm["chr_aXY"].shape)
    print("dataX shape: ", adata.varm["chr_aX"].shape)
    print("dataY shape: ", adata.varm["chr_aY"].shape)
    print("data_autosome shape: ", adata.varm["autosomes"].shape)

    return adata

    # # ----------------------------------- Remove sex chr transcripts -------------------------------------
    # # data = pd.read_csv(fdir_processed / 'geuvadis.preprocessed.csv', index_col=0)
    # transcripts_x = pd.read_csv(fdir_processed / "all_transcripts.chrX.csv",
    #                             index_col=0).values.ravel().tolist()
    # transcripts_y = pd.read_csv(fdir_processed / "all_transcripts.chrY.csv",
    #                             index_col=0).values.ravel().tolist()

    # data_noX = data.drop(columns=data.columns.intersection(transcripts_x))
    # data_noY = data.drop(columns=data.columns.intersection(transcripts_y))
    # data_noXY = data_noY.drop(columns=data.columns.intersection(transcripts_x))

    # data_noX.to_csv(fdir_traintest / 'sex' / 'geuvadis.preprocessed.chrY.csv')
    # data_noY.to_csv(fdir_traintest / 'sex' / 'geuvadis.preprocessed.chrX.csv')
    # data_noXY.to_csv(fdir_traintest / 'sex' / 'geuvadis.preprocessed.autosome.csv')
    # data.to_csv(fdir_traintest / 'sex' / 'geuvadis.preprocessed.chrXY.csv')


@flow
def make_train_dataset(organ="None", splitby=None):
    value_to_predict = "sex"
    # value_to_predict = value_to_predict.lower()
    # value_to_predict = 'Age'

    dataset_name = organ
    if organ == "None":
        dataset_name = "geuvadis"

    logger.info(f"{dataset_name=}")
    adata = ad.read_h5ad(FDIR_INTEMEDIATE / f"{dataset_name.upper()}.raw.h5ad")

    datasets = {}
    if splitby:
        logger.info(f"{dataset_name} splet by {splitby}")
        categories = adata.obs[splitby].unique()
        for category in categories:
            datasets[category] = adata.to_df()[adata.obs[splitby] == category]
    else:
        datasets["RAW"] = adata.to_df()

    columns_ = pd.Index([])

    for key, data_raw in datasets.items():
        logger.info(f"{key=}")
        logger.info(f"{dataset_name} under key {key} has {len(data_raw.columns)} transcripts")
        logger.info(f"{dataset_name} under key {key} zero median filtered")
        data_ = filter_zero_median(data_raw)

        if dataset_name != "geuvadis":
            columns_ = columns_.union(data_.columns)
            continue

        # data_ = filter_correlated(data_, adata.obs[value_to_predict].loc[data_.index])
        logger.info(f"{dataset_name} under key {key}  logarithmized")
        data_ = logarithmization(data_)

        # data_ = filter_cv_threshold(data_, 0.7)
        logger.info(f"{dataset_name} under key {key}  mean > q34 filtered")
        data_ = filter_median_q34(data_)
        # data_ = filter_cv_q34(data_)

        logger.info(f"store {len(data_.columns)} transcripts")
        columns_ = columns_.union(data_.columns)

    adata = adata[:, columns_]
    logger.info(f"shape after key-wised filtration: {adata.shape=}")

    logger.info(f"{dataset_name} logarithmized")
    data = logarithmization(adata.to_df())

    # CV together, median - separate
    if dataset_name == "geuvadis":
        logger.info(f"{dataset_name} logarithmized")
        data = filter_cv_q34(data)

    data = data.astype(np.float32)

    adata = adata[data.index, data.columns]
    adata.layers["raw"] = adata.X.copy()
    adata.X = data.values

    logger.info("split by sex")
    adata = split_by_sex_transcripts(adata)

    # gtf_data = gtf_data.loc[data.columns]
    # data_header = data_header.loc[data.index]

    # data.to_hdf(fdir_intermediate / f'{dataset_name}.preprocessed.h5', key="data", format='f')
    # data_header.to_hdf(fdir_intermediate / f'{dataset_name}.preprocessed.h5', key="header", format='f')
    # gtf_data.to_hdf(fdir_intermediate / f'{dataset_name}.preprocessed.h5', key="gtf", format='table')

    # data_Y.to_hdf(fdir_processed / value_to_predict / f'{dataset_name}.preprocessed.{value_to_predict}.h5', key='chrY', format='f')
    # data_X.to_hdf(fdir_processed / value_to_predict / f'{dataset_name}.preprocessed.{value_to_predict}.h5', key='chrX', format='f')
    # data_autosomes.to_hdf(fdir_processed / value_to_predict / f'{dataset_name}.preprocessed.{value_to_predict}.h5', key='autosome', format='f')
    # data_XY.to_hdf(fdir_processed / value_to_predict / f'{dataset_name}.preprocessed.{value_to_predict}.h5', key='chrXY', format='f')

    adata.write(
        FDIR_PROCESSED
        / value_to_predict
        / f"{dataset_name.upper()}.preprocessed.{value_to_predict}.h5ad"
    )


if __name__ == "__main__":
    organ = "None"
    make_train_dataset(organ=organ, splitby="sex")

    # organ = "HEART"
    # organ = 'BRAIN0'
    # organ = 'BRAIN1'
    for organ in ["HEART", "BRAIN0", "BRAIN1"]:
        make_train_dataset(organ=organ)
