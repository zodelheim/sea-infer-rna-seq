import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from scipy.stats import pointbiserialr, pearsonr, spearmanr

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
def filter_sex_correlated(X: pd.DataFrame, y: pd.DataFrame | pd.Series, threshold=0.8) -> pd.DataFrame:
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


@task(log_prints=True, description='drop correlated')
def filter_correlated(X: pd.DataFrame, y: pd.DataFrame | pd.Series, threshold=0.8):
    X_corr = X
    y_corr = y
    columns_to_drop = []

    for c in tqdm(X_corr.columns):
        corr, pvalue = spearmanr(X_corr[c], LabelEncoder().fit_transform(y_corr.values))
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


@task(log_prints=True, description='read all raw geuvadis data')
def read_dataset(fname_data: Path | str,
                 fname_header: Path | str,
                 fname_gtf: Path | str,
                 separator=','):
    data_raw = pd.read_csv(fname_data, index_col=0, sep=separator).T
    data_raw = data_raw.astype(np.float32)

    data_header = pd.read_csv(fname_header, index_col=0, sep=',')

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

    pseudoautosoms_Y1 = [10001, 2781479]
    pseudoautosoms_X1 = [10001, 2781479]
    pseudoautosoms_Y2 = [56887903, 57217415]
    pseudoautosoms_X2 = [155701383, 156030895]

    transcripts_x = gtf_data.loc[gtf_data['seqname'] == 'chrX']
    transcripts_y = gtf_data.loc[gtf_data['seqname'] == 'chrY']

    true_transcripts_x = transcripts_x.loc[((transcripts_x['end'] < pseudoautosoms_X1[0])
                                            | ((transcripts_x["start"] > pseudoautosoms_X1[1]) & (transcripts_x["end"] < pseudoautosoms_X2[0]))
                                            | (transcripts_x["start"] > pseudoautosoms_X2[1])
                                            )]

    true_transcripts_y = transcripts_y.loc[((transcripts_y['end'] < pseudoautosoms_Y1[0])
                                            | ((transcripts_y["start"] > pseudoautosoms_Y1[1]) & (transcripts_y["end"] < pseudoautosoms_Y2[0]))
                                            | (transcripts_y["start"] > pseudoautosoms_Y2[1])
                                            )]

    transcripts_x = transcripts_x['transcript_id'].unique()
    transcripts_y = transcripts_y['transcript_id'].unique()

    true_transcripts_x = true_transcripts_x['transcript_id'].unique()
    true_transcripts_y = true_transcripts_y['transcript_id'].unique()

    return true_transcripts_x, true_transcripts_y


@task(log_prints=True, description='drop sex (chrX, chrY) transcripts from data')
def remove_sex_transcripts(data: pd.DataFrame, gtf_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    transcripts_x, transcripts_y = locate_sex_transcripts(gtf_data)

    transcripts_x = transcripts_x.tolist()
    transcripts_y = transcripts_y.tolist()

    transcripts_x = data.columns.intersection(transcripts_x)
    transcripts_y = data.columns.intersection(transcripts_y)

    gtf_transcripts = gtf_data.loc[data.columns]
    transcripts_autosomes = gtf_transcripts.loc[(gtf_transcripts['seqname'] != "chrX") & (gtf_transcripts['seqname'] != "chrY")].index

    data_XY = data
    data_X = data[transcripts_x.union(transcripts_autosomes)]
    data_Y = data[transcripts_y.union(transcripts_autosomes)]
    data_autosomes = data[transcripts_autosomes]

    print('dataXY shape: ', data_XY.shape)
    print('dataX shape: ', data_X.shape)
    print('dataY shape: ', data_Y.shape)
    print('data_autosome shape: ', data_autosomes.shape)

    return data_XY, data_X, data_Y, data_autosomes

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
def make_train_dataset(organ="None"):
    fdir_raw = Path("data/raw/")
    fdir_processed = Path("data/interim")
    fdir_traintest = Path("data/processed")
    fdir_external = Path("data/external")

    # value_to_predict = 'Sex'
    # value_to_predict = value_to_predict.lower()

    value_to_predict = 'Age'

    dataset_name = organ
    if organ == "None":
        dataset_name = 'geuvadis'

    if organ == "None":
        data_raw, data_header, gtf_data = read_dataset(
            fdir_raw / 'Geuvadis.all.csv',
            fdir_raw / 'Geuvadis.SraRunTable.txt',
            fdir_raw / 'all_transcripts_strigtie_merged.gtf'
        )

    if organ in ["HEART", "BRAIN0", "BRAIN1"]:
        if organ == "BRAIN1":
            fname = next((fdir_external / organ / 'reg').glob("*.csv"))
            separator = ","
        else:
            fname = next((fdir_external / organ / 'reg').glob("*TPM.txt"))
            separator = "\t"
        fname = fname.name

        data_raw, data_header, gtf_data = read_dataset(
            fdir_external / organ / 'reg' / fname,
            fdir_external / organ / 'reg' / 'SraRunTable.txt',
            fdir_raw / 'all_transcripts_strigtie_merged.gtf',
            separator
        )
    data = filter_zero_median(data_raw)

    if organ == "None":
        data = filter_sex_correlated(data, data_header[value_to_predict].loc[data.index])
    else:
        data = filter_correlated(data, data_header[value_to_predict].loc[data.index])

    data = logarithmization(data)
    data = filter_cv_threshold(data, 0.7)

    data = filter_median_q34(data)
    data = filter_cv_q34(data)
    data = data.astype(np.float32)

    data_XY, data_X, data_Y, data_autosomes = remove_sex_transcripts(data, gtf_data)

    gtf_data = gtf_data.loc[data.columns]
    data_header = data_header.loc[data.index]

    data.to_hdf(fdir_processed / f'{dataset_name}.preprocessed.h5', key="data", format='f')
    data_header.to_hdf(fdir_processed / f'{dataset_name}.preprocessed.h5', key="header", format='f')
    gtf_data.to_hdf(fdir_processed / f'{dataset_name}.preprocessed.h5', key="gtf", format='table')

    data_Y.to_hdf(fdir_traintest / value_to_predict / f'{dataset_name}.preprocessed.{value_to_predict}.h5', key='chrY', format='f')
    data_X.to_hdf(fdir_traintest / value_to_predict / f'{dataset_name}.preprocessed.{value_to_predict}.h5', key='chrX', format='f')
    data_autosomes.to_hdf(fdir_traintest / value_to_predict / f'{dataset_name}.preprocessed.{value_to_predict}.h5', key='autosome', format='f')
    data_XY.to_hdf(fdir_traintest / value_to_predict / f'{dataset_name}.preprocessed.{value_to_predict}.h5', key='chrXY', format='f')


if __name__ == "__main__":

    # organ = 'None'
    # organ = 'HEART'
    # organ = 'BRAIN0'
    organ = 'BRAIN1'

    make_train_dataset(organ=organ)
