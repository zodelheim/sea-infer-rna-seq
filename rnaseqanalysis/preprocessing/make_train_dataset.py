import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from scipy.stats import pointbiserialr

from tqdm import tqdm
from gtfparse import read_gtf
from prefect import flow, task


@task(log_prints=True, description='drop zero median transcripts')
def filter_zero_median(df: pd.DataFrame) -> pd.DataFrame:
    df_median = df.median()
    if (df_median == 0).any():
        cols_to_drop = df.columns[df_median == 0]
        # print(len(cols_to_drop),
        #       " features will be removed, due to a zero median value")
        df = df.drop(columns=cols_to_drop)
        # print("Current dataset size: ", df.shape)
        print('Dataset shape: ', df.shape)
        return df

    # print("Zero median columns aren't found")
    print('Dataset shape: ', df.shape)

    return df


@task(log_prints=True, description='drop correlated with sex transcripts')
def filter_correlated(X: pd.DataFrame, y: pd.DataFrame | pd.Series, threshold=0.8) -> pd.DataFrame:
    X_corr = X
    y_corr = y

    columns_to_drop = []
    for c in tqdm(X_corr.columns):
        corr, pvalue = pointbiserialr(X_corr[c], LabelEncoder().fit_transform(y_corr.values))
        if np.abs(corr) > threshold:
            columns_to_drop.append(c)

    X = X.drop(columns=columns_to_drop)
    print('Dataset shape: ', X.shape)
    return X


@task(log_prints=True, description='logarithmization (log2(x+1))')
def logarithmization(df: pd.DataFrame):
    df = np.log2(df + 1)
    return df


@task(log_prints=True, description='drop transcripts with cv < threshold')
def filter_cv_threshold(df: pd.DataFrame, threshold: float):
    cv = df.std() / df.mean()
    low_cv_cols = cv[cv < threshold].index

    if len(low_cv_cols) > 0:
        # print(f"{len(low_cv_cols)} features have coefficient of variation below {threshold} and will be removed.")
        df = df.drop(columns=low_cv_cols)
    # else:
    #     print("No features found with coefficient of variation below the threshold.")
    # print(f"Current amount of features is {len(df.columns)}")

    print('Dataset shape: ', df.shape)
    return df


@task(log_prints=True, description='read all raw data')
def read_geuvadis(fname_data: Path | str,
                  fname_header: Path | str,
                  fname_gtf: Path | str):
    data_raw = pd.read_csv(fname_data, index_col=0).T
    data_raw = data_raw.astype(np.float32)

    data_header = pd.read_csv(fname_header, index_col=0)

    gtf_rawdata = read_gtf(fname_gtf)
    gtf_data = gtf_rawdata.to_pandas()
    gtf_data = gtf_data.set_index('transcript_id')
    gtf_data['transcript_id'] = gtf_data.index

    gtf_data = gtf_data.drop_duplicates("transcript_id")

    print('Dataset shape: ', data_raw.shape)

    return data_raw, data_header, gtf_data


@task(log_prints=True, description='filter transcripts with mean < median')
def filter_median_q34(data: pd.DataFrame):
    mean = data.mean(axis=0)
    median = mean.median()
    data = data.loc[:, mean > median]
    print('Dataset shape: ', data.shape)
    return data


@task(log_prints=True, description='filter transcripts with cv mean < cv median')
def filter_cv_q34(data: pd.DataFrame):
    cv = data.std() / data.mean()
    median_cv = cv.median()
    data = data.loc[:, cv > median_cv]
    print('Dataset shape: ', data.shape)
    return data


@task(log_prints=True, description='extract Sex (chrX, chrY) transcripts from gtf')
def locate_sex_transcripts(gtf_data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:

    # from https://www.ensembl.org/info/genome/genebuild/human_PARS.html
    pseudoautosoms_Y1 = [10000, 2781479]
    pseudoautosoms_X1 = [10000, 2781479]
    pseudoautosoms_Y2 = [56887902, 57217415]
    pseudoautosoms_X2 = [155701382, 156030895]

    transcripts_x = gtf_data.loc[gtf_data['seqname'] == 'chrX']  # , 'transcript_id'
    transcripts_y = gtf_data.loc[gtf_data['seqname'] == 'chrY']  # , 'transcript_id'

    pseudoauto_tr1 = (transcripts_x.loc[(transcripts_x['start'] >= pseudoautosoms_X1[0]) & (transcripts_x['end'] <= pseudoautosoms_X1[0])]).index
    pseudoauto_tr2 = (transcripts_y.loc[(transcripts_y['start'] >= pseudoautosoms_Y1[0]) & (transcripts_y['end'] <= pseudoautosoms_Y1[0])]).index
    pseudoauto_tr3 = (transcripts_x.loc[(transcripts_x['start'] >= pseudoautosoms_X2[0]) & (transcripts_x['end'] <= pseudoautosoms_X2[0])]).index
    pseudoauto_tr4 = (transcripts_y.loc[(transcripts_y['start'] >= pseudoautosoms_Y2[0]) & (transcripts_y['end'] <= pseudoautosoms_Y2[0])]).index

    pseudoautosom_transcripts = pseudoauto_tr1.union(pseudoauto_tr2).union(pseudoauto_tr3).union(pseudoauto_tr4)

    gtf_data = gtf_data.drop(index=pseudoautosom_transcripts)

    transcripts_x = gtf_data.loc[gtf_data['seqname'] == 'chrX', 'transcript_id']
    transcripts_y = gtf_data.loc[gtf_data['seqname'] == 'chrY', 'transcript_id']

    transcripts_x = transcripts_x.unique()
    transcripts_y = transcripts_y.unique()
    print('# chrX transcripts: ', len(transcripts_x))
    print('# chrY transcripts: ', len(transcripts_y))

    return transcripts_x, transcripts_y


@ task(log_prints=True, description='drop sex (chrX, chrY) transcripts from data')
def remove_sex_transcripts(data: pd.DataFrame, gtf_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    transcripts_x, transcripts_y = locate_sex_transcripts(gtf_data)
    transcripts_x = transcripts_x.tolist()
    transcripts_y = transcripts_y.tolist()

    data_noX = data.drop(columns=data.columns.intersection(transcripts_x))
    data_noY = data.drop(columns=data.columns.intersection(transcripts_y))
    data_noXY = data_noY.drop(columns=data.columns.intersection(transcripts_x))

    print('dataXY shape: ', data.shape)
    print('dataY shape: ', data_noX.shape)
    print('dataX shape: ', data_noY.shape)
    print('data_autosome shape: ', data_noXY.shape)

    return data, data_noX, data_noY, data_noXY

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


@ flow
def make_train_dataset():

    fdir_raw = Path("data/raw/")
    fdir_processed = Path("data/interim")
    fdir_traintest = Path("data/processed")

    data_raw, data_header, gtf_data = read_geuvadis(
        fdir_raw / 'Geuvadis.all.csv',
        fdir_raw / 'Geuvadis.SraRunTable.txt',
        fdir_raw / 'all_transcripts_strigtie_merged.gtf'
    )

    data = filter_zero_median(data_raw)
    # data = filter_correlated(data, data_header['Sex'].loc[data.index])
    data = logarithmization(data)
    data = filter_cv_threshold(data, 0.7)

    data = filter_median_q34(data)
    data = filter_cv_q34(data)
    data = data.astype(np.float32)

    data, data_noX, data_noY, data_noXY = remove_sex_transcripts(data, gtf_data)

    gtf_data = gtf_data.loc[data.columns]

    # print(gtf_data)

    data.to_hdf(fdir_processed / 'geuvadis.preprocessed.h5', key="geuvadis", format='f')
    data_header.to_hdf(fdir_processed / 'geuvadis.preprocessed.h5', key="header", format='f')
    gtf_data.to_hdf(fdir_processed / 'geuvadis.preprocessed.h5', key="gtf", format='table')

    # # data_noX.to_csv(fdir_traintest / 'sex' / 'geuvadis.preprocessed.chrY.csv')
    # # data_noY.to_csv(fdir_traintest / 'sex' / 'geuvadis.preprocessed.chrX.csv')
    # # data_noXY.to_csv(fdir_traintest / 'sex' / 'geuvadis.preprocessed.autosome.csv')
    # # data.to_csv(fdir_traintest / 'sex' / 'geuvadis.preprocessed.chrXY.csv')


if __name__ == "__main__":

    make_train_dataset()
