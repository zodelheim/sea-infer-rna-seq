from gtfparse import read_gtf
from pathlib import Path
import pandas as pd
import numpy as np

fdir_raw = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/raw/")
fdir_processed = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/interim")


gtf_rawdata = read_gtf(fdir_raw / 'all_transcripts_strigtie_merged.gtf')

gtf_data = gtf_rawdata.to_pandas()

gtf_data = gtf_data[['seqname', 'transcript_id']]
transcripts_x = gtf_data.loc[gtf_data['seqname'] == 'chrX', 'transcript_id']
transcripts_y = gtf_data.loc[gtf_data['seqname'] == 'chrY', 'transcript_id']
transcripts_x = transcripts_x.unique()
transcripts_y = transcripts_y.unique()

print(len(transcripts_x))
print(len(transcripts_y))


pd.Series(transcripts_x).to_csv(fdir_processed / "all_transcripts.chrX.csv")
pd.Series(transcripts_y).to_csv(fdir_processed / "all_transcripts.chrY.csv")

# ------------------ test if transcripts from geuvadis are in gencode v44 ---------------
# print(gtf_data.loc[gtf_data['transcript_id'] == "MSTRG.35846.29"])

data = pd.read_csv(fdir_raw / 'Geuvadis.all.csv', index_col=0).T
data = data.astype(np.float32)

transcripts_geu = data.columns
print(f"{len(transcripts_geu)=}")

gtf_data = gtf_data.set_index('transcript_id')
transcripts_gencode = gtf_data.index
print(f"{len(transcripts_gencode)=}")

print(len(transcripts_geu.intersection(transcripts_gencode)))

print(len(transcripts_geu.intersection(transcripts_x)))
print(len(transcripts_geu.intersection(transcripts_y)))
