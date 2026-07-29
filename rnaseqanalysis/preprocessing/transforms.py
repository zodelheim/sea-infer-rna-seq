import anndata as ad
import numpy as np
import pandas as pd
from loguru import logger
from rich.progress import track
from scipy.stats import pointbiserialr, spearmanr
from sklearn.preprocessing import LabelEncoder
import functools

# from https://www.ensembl.org/info/genome/genebuild/human_PARS.html
PSEUDOAUTOSOMS_Y1 = [10001, 2781479]
PSEUDOAUTOSOMS_X1 = [10001, 2781479]
PSEUDOAUTOSOMS_Y2 = [56887903, 57217415]
PSEUDOAUTOSOMS_X2 = [155701383, 156030895]


def log_step(func):
    @functools.wraps(func)
    def wrapper(df, *args, **kwargs):
        logger.info(f"{func.__name__}: start")
        result = func(df, *args, **kwargs)
        shape = getattr(result, "shape", None)
        logger.info(f"{func.__name__}: done" + (f" (Dataset shape: {shape})" if shape else ""))
        return result

    return wrapper


@log_step
def logarithmization(df: pd.DataFrame):
    df = df + 1
    df.apply(np.log2, inplace=True)
    return df


@log_step
def filter_zero_median(df: pd.DataFrame) -> pd.DataFrame:
    df_median = df.median()
    if (df_median == 0).any():
        cols_to_drop = df.columns[df_median == 0]
        logger.warning(
            f"{len(cols_to_drop)} transcripts will be removed, due to a zero median value"
        )
        df = df.drop(columns=cols_to_drop)
        # logger.info(f"Current dataset size: {df.shape}")
        return df

    logger.warning("Zero median columns aren't found")
    # logger.info(f"Dataset shape: {df.shape}")
    return df


@log_step
def filter_correlated(X: pd.DataFrame, y: pd.DataFrame | pd.Series, threshold=0.8) -> pd.DataFrame:
    X_corr = X
    y_corr = y
    y_encoded = LabelEncoder().fit_transform(y_corr.values)

    if len(np.unique(y_encoded)) == 2:
        corr_function = pointbiserialr
    else:
        corr_function = spearmanr

    columns_to_drop = []
    for c in track(X_corr.columns):
        corr, pvalue = corr_function(X_corr[c], y_encoded)
        if np.abs(corr) > threshold:
            columns_to_drop.append(c)

    X = X.drop(columns=columns_to_drop)
    # logger.info(f"Dataset shape: {X.shape}")
    return X


@log_step
def filter_cv_threshold(df: pd.DataFrame, threshold: float):
    cv = df.std() / df.mean()
    low_cv_cols = cv[cv < threshold].index

    if len(low_cv_cols) > 0:
        df = df.drop(columns=low_cv_cols)

    # logger.info(f"Dataset shape: {df.shape}")
    return df


@log_step
def filter_median_q34(data: pd.DataFrame):
    mean = data.mean(axis=0)
    median = mean.median()
    data = data.loc[:, mean > median]
    # logger.info(f"Dataset shape: {data.shape}")
    return data


@log_step
def filter_cv_q34(data: pd.DataFrame):
    cv = data.std() / data.mean()
    median_cv = cv.median()
    data = data.loc[:, cv > median_cv]
    # logger.info(f"Dataset shape: {data.shape}")
    return data


def locate_sex_transcripts(gtf_data: pd.DataFrame, drop_duplicates) -> tuple[pd.Series, pd.Series]:

    transcripts_x = gtf_data.loc[gtf_data["seqname"] == "chrX"]
    transcripts_y = gtf_data.loc[gtf_data["seqname"] == "chrY"]

    true_transcripts_x = transcripts_x.loc[
        (
            (transcripts_x["end"] < PSEUDOAUTOSOMS_X1[0])
            | (
                (transcripts_x["start"] > PSEUDOAUTOSOMS_X1[1])
                & (transcripts_x["end"] < PSEUDOAUTOSOMS_X2[0])
            )
            | (transcripts_x["start"] > PSEUDOAUTOSOMS_X2[1])
        )
    ]

    true_transcripts_y = transcripts_y.loc[
        (
            (transcripts_y["end"] < PSEUDOAUTOSOMS_Y1[0])
            | (
                (transcripts_y["start"] > PSEUDOAUTOSOMS_Y1[1])
                & (transcripts_y["end"] < PSEUDOAUTOSOMS_Y2[0])
            )
            | (transcripts_y["start"] > PSEUDOAUTOSOMS_Y2[1])
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


def labeling_sex_transcripts(adata: ad.AnnData, drop_duplicates=True) -> ad.AnnData:
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

    logger.info(f"dataXY shape: {adata.varm['chr_aXY'].shape}")
    logger.info(f"dataX shape: {adata.varm['chr_aX'].shape}")
    logger.info(f"dataY shape: {adata.varm['chr_aY'].shape}")
    logger.info(f"data_autosome shape: {adata.varm['autosomes'].shape}")

    return adata
