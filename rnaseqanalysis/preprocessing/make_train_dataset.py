import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from scipy.stats import pointbiserialr

from tqdm import tqdm
from gtfparse import read_gtf


def filter_zero_median(df: pd.DataFrame) -> pd.DataFrame:
    df_median = df.median()
    if (df_median == 0).any():
        cols_to_drop = df.columns[df_median == 0]
        print(len(cols_to_drop),
              " features will be removed, due to a zero median value")
        df = df.drop(columns=cols_to_drop)
        print("Current dataset size: ", df.shape)
        return df

    print("Zero median columns aren't found")
    print('Dataset shape: ', df.shape)
    return df


def filter_correlated(X: pd.DataFrame, y: pd.DataFrame | pd.Series, threshold=0.8) -> pd.DataFrame:
    X_corr = X
    y_corr = y

    columns_to_drop = []
    for c in tqdm(X_corr.columns):
        corr, pvalue = pointbiserialr(X_corr[c], LabelEncoder().fit_transform(y_corr.values))
        if np.abs(corr) > threshold:
            columns_to_drop.append(c)

    return X.drop(columns=columns_to_drop)


def logarithmization(df: pd.DataFrame):
    # numerical_cols = df.iloc[:1].select_dtypes(include=[np.number]).columns
    # df = df.replace(0, 1e-6)
    df = np.log2(df + 1)
    return df


def filter_by_cv(df, threshold):
    cv = df.std() / df.mean()
    low_cv_cols = cv[cv < threshold].index

    if len(low_cv_cols) > 0:
        print(f"{len(low_cv_cols)} features have coefficient of variation below {threshold} and will be removed.")
        df = df.drop(columns=low_cv_cols)
    else:
        print("No features found with coefficient of variation below the threshold.")
    print(f"Current amount of features is {len(df.columns)}")
    return df


fdir_raw = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/raw/")
fdir_processed = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/interim")
fdir_traintest = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/processed")


# -----------------------------------Read data---------------------------------------------

data = pd.read_csv(fdir_raw / 'Geuvadis.all.csv', index_col=0).T
data = data.astype(np.float32)
# data.rename(columns={"Unnamed: 0": "trascripts"})

data_header = pd.read_csv(fdir_raw / 'Geuvadis.SraRunTable.txt', index_col=0)
data_header = data_header[['Sex',
                           'Experimental_Factor:_population (exp)']]
print(data.shape)
# -----------------------------------Pipeline---------------------------------------------

data = filter_zero_median(data)
print(data.shape)

data = filter_correlated(data,
                         data_header['Sex'].loc[data.index])
print(data.shape)

data = logarithmization(data)

data = filter_by_cv(data, 0.7)
print(data.shape)

mean = data.mean(axis=0)
median = mean.median()
data = data.loc[:, mean > median]
print(data.shape)

cv = data.std() / data.mean()
median_cv = cv.median()
data = data.loc[:, cv > median_cv]

data = data.astype(np.float32)

gencode_intersect = False
if gencode_intersect:
    gtf_rawdata = read_gtf(fdir_raw / 'all_transcripts_strigtie_merged.gtf')
    gtf_data = gtf_rawdata.to_pandas()
    gtf_data = gtf_data[['seqname', 'transcript_id']]

    gtf_data = gtf_data.set_index('transcript_id')

    transcripts_gencode = gtf_data.index
    transcripts_geu = data.columns

    intersection = transcripts_geu.intersection(transcripts_gencode)

    data = data[intersection]
    print(data.shape)

    # data['Sex'] = data_header['Sex']

data.to_csv(fdir_processed / 'geuvadis.preprocessed.csv')

# ----------------------------------- Remove sex chr transcripts -------------------------------------
# data = pd.read_csv(fdir_processed / 'geuvadis.preprocessed.csv', index_col=0)
transcripts_x = pd.read_csv(fdir_processed / "all_transcripts.chrX.csv",
                            index_col=0).values.ravel().tolist()
transcripts_y = pd.read_csv(fdir_processed / "all_transcripts.chrY.csv",
                            index_col=0).values.ravel().tolist()

data_noX = data.drop(columns=data.columns.intersection(transcripts_x))
data_noY = data.drop(columns=data.columns.intersection(transcripts_y))
data_noXY = data_noY.drop(columns=data.columns.intersection(transcripts_x))

data_noX.to_csv(fdir_traintest / 'sex' / 'geuvadis.preprocessed.chrY.csv')
data_noY.to_csv(fdir_traintest / 'sex' / 'geuvadis.preprocessed.chrX.csv')
data_noXY.to_csv(fdir_traintest / 'sex' / 'geuvadis.preprocessed.autosome.csv')
data.to_csv(fdir_traintest / 'sex' / 'geuvadis.preprocessed.chrXY.csv')
