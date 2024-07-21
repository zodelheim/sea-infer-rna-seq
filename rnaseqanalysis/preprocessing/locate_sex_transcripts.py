from gtfparse import read_gtf
from pathlib import Path
import pandas as pd
import numpy as np

fdir_raw = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/raw/")
fdir_processed = Path("/home/ar3/Documents/PYTHON/RNASeqAnalysis/data/interim")


gtf_rawdata = read_gtf(fdir_raw / 'all_transcripts_strigtie_merged.gtf')
gtf_data = gtf_rawdata.to_pandas()

gtf_data = gtf_data[['seqname', 'transcript_id', 'start', 'end']]
transcripts_x = gtf_data.loc[gtf_data['seqname'] == 'chrX']
transcripts_y = gtf_data.loc[gtf_data['seqname'] == 'chrY']

# drop pseudoautosomes
pseudoautosoms_Y1 = [10001, 2781479]
pseudoautosoms_X1 = [10001, 2781479]
pseudoautosoms_Y2 = [56887903, 57217415]
pseudoautosoms_X2 = [155701383, 156030895]

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

print(len(transcripts_x))
print(len(transcripts_y))
print(len(true_transcripts_x))
print(len(true_transcripts_y))

pd.Series(transcripts_x).to_csv(fdir_processed / "all_transcripts.chrX.csv")
pd.Series(transcripts_y).to_csv(fdir_processed / "all_transcripts.chrY.csv")
pd.Series(true_transcripts_x).to_csv(fdir_processed / "true_transcripts.chrX.csv")
pd.Series(true_transcripts_y).to_csv(fdir_processed / "true_transcripts.chrY.csv")

exit()
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
