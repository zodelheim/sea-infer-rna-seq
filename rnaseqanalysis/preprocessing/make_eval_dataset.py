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


def logarithmization(df: pd.DataFrame):
    # numerical_cols = df.iloc[:1].select_dtypes(include=[np.number]).columns
    # df = df.replace(0, 1e-6)
    df = np.log2(df + 1)
    return df


fdir_raw = Path("data/raw/")
fdir_processed = Path("data/interim")
fdir_traintest = Path("data/processed")
fdir_external = Path("data/external")


for subdir in ["BLOOD1", 'BRAIN0', "HEART", "BRAIN1"]:

    # fname = Path("heart.merged.TPM.txt")
    fname = next((fdir_external / subdir / 'reg').glob("*.txt"))
    fname = fname.name

    data = pd.read_csv(fdir_external / subdir / 'reg' / fname, sep='\t').T
    data_header = pd.read_csv(fdir_external / subdir / 'reg' / 'SraRunTable.txt', sep=',')
    print(data_header.columns)
    data = filter_zero_median(data)
    print(data.shape)
    data = logarithmization(data)
    data = data.astype(np.float32)

    data.to_hdf((fdir_external / subdir / 'reg' / fname).with_suffix('.processed.h5'),
                key='processed', format='f')
