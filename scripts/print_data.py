import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from scipy.stats import pointbiserialr

from tqdm import tqdm
from gtfparse import read_gtf
from prefect import flow, task


fdir_raw = Path("data/raw/")
fdir_processed = Path("data/interim")
fdir_traintest = Path("data/processed")


# data = pd.read_hdf(fdir_processed / 'geuvadis.preprocessed.h5', key="geuvadis")
# data_header = pd.read_hdf(fdir_processed / 'geuvadis.preprocessed.h5', key="header")
gtf_data = pd.read_hdf(fdir_processed / 'geuvadis.preprocessed.h5', key="gtf")

# print(gtf_data)

pseudoautosoms_Y1 = [10001, 2781479]
pseudoautosoms_X1 = [10001, 2781479]
pseudoautosoms_Y2 = [56887903, 57217415]
pseudoautosoms_X2 = [155701383, 156030895]

transcripts_x = gtf_data.loc[gtf_data['seqname'] == 'chrX']  # , 'transcript_id'
transcripts_y = gtf_data.loc[gtf_data['seqname'] == 'chrY']  # , 'transcript_id'

psauto_tr1 = (transcripts_x.loc[(transcripts_x['start'] >= pseudoautosoms_X1[0]) & (transcripts_x['end'] <= pseudoautosoms_X1[0])]).index
psauto_tr2 = (transcripts_y.loc[(transcripts_y['start'] >= pseudoautosoms_Y1[0]) & (transcripts_y['end'] <= pseudoautosoms_Y1[0])]).index
psauto_tr3 = (transcripts_x.loc[(transcripts_x['start'] >= pseudoautosoms_X2[0]) & (transcripts_x['end'] <= pseudoautosoms_X2[0])]).index
psauto_tr4 = (transcripts_y.loc[(transcripts_y['start'] >= pseudoautosoms_Y2[0]) & (transcripts_y['end'] <= pseudoautosoms_Y2[0])]).index

pseudoautosom_transcripts = psauto_tr1.union(psauto_tr2).union(psauto_tr3).union(psauto_tr4)
# print(psauto_tr1.union(psauto_tr2).union(psauto_tr3).union(psauto_tr4))
gtf_data = gtf_data.drop(index=pseudoautosom_transcripts)
