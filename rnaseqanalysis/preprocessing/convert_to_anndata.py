import pandas as pd
import numpy as np

from gtfparse import read_gtf
import anndata as ad
from pathlib import Path

from config import FDIR_RAW, FDIR_INTEMEDIATE, FDIR_EXTERNAL


def read_dataset(fname_data: Path | str,
                 fname_header: Path | str,
                 fname_gtf: Path | str,
                 separator=','):
    data_raw = pd.read_csv(fname_data, index_col=0, sep=separator).T
    data_raw = data_raw.astype(np.float32)

    samples_annot = pd.read_csv(fname_header, index_col=0, sep=',')

    gtf_rawdata = read_gtf(fname_gtf)
    gtf_data = gtf_rawdata.to_pandas()
    gtf_data = gtf_data.set_index('transcript_id')
    gtf_data['transcript_id'] = gtf_data.index

    gtf_data = gtf_data.drop_duplicates("transcript_id")

    print('Dataset shape: ', data_raw.shape)

    columns = data_raw.columns.intersection(gtf_data.index)
    indices = data_raw.index.intersection(samples_annot.index)

    data_raw = data_raw.loc[indices, columns]
    samples_annot = samples_annot.loc[indices]
    gtf_data = gtf_data.loc[columns]

    adata = ad.AnnData(X=data_raw, obs=samples_annot, var=gtf_data)

    return adata


def convert_geuvadis(fname_data: Path | str,
                     fname_header: Path | str,
                     fname_gtf: Path | str):
    data_raw, data_header, gtf_data = read_dataset(
        FDIR_RAW / 'Geuvadis.all.csv',
        FDIR_RAW / 'Geuvadis.SraRunTable.txt',
        FDIR_RAW / 'all_transcripts_strigtie_merged.gtf'
    )


def convert_heart(fname_data: Path | str,
                  fname_header: Path | str,
                  fname_gtf: Path | str,):
    fname = next((FDIR_EXTERNAL / "HEART" / 'reg').glob("*TPM.txt"))
    separator = "\t"
    data_raw, data_header, gtf_data = read_dataset(
        FDIR_EXTERNAL / "HEART" / 'reg' / fname,
        FDIR_EXTERNAL / "HEART" / 'reg' / 'SraRunTable.txt',
        FDIR_RAW / 'all_transcripts_strigtie_merged.gtf',
        separator
    )


def convert_brain_0(fname_data: Path | str,
                    fname_header: Path | str,
                    fname_gtf: Path | str,):
    fname = next((FDIR_EXTERNAL / "BRAIN0" / 'reg').glob("*.csv"))
    separator = ","
    data_raw, data_header, gtf_data = read_dataset(
        FDIR_EXTERNAL / "BRAIN0" / 'reg' / fname,
        FDIR_EXTERNAL / "BRAIN0" / 'reg' / 'SraRunTable.txt',
        FDIR_RAW / 'all_transcripts_strigtie_merged.gtf',
        separator
    )


def convert_brain1(fname_data: Path | str,
                   fname_header: Path | str,
                   fname_gtf: Path | str,):
    fname = next((FDIR_EXTERNAL / "BRAIN1" / 'reg').glob("*.csv"))
    separator = ","
    data_raw, data_header, gtf_data = read_dataset(
        FDIR_EXTERNAL / "BRAIN1" / 'reg' / fname,
        FDIR_EXTERNAL / "BRAIN1" / 'reg' / 'SraRunTable.txt',
        FDIR_RAW / 'all_transcripts_strigtie_merged.gtf',
        separator
    )


if __name__ == "__main__":

    geuvadis = read_dataset(
        FDIR_RAW / 'Geuvadis.all.csv',
        FDIR_RAW / 'Geuvadis.SraRunTable.txt',
        FDIR_RAW / 'all_transcripts_strigtie_merged.gtf'
    )
    geuvadis.obs['sex'] = geuvadis.obs['sex']
    geuvadis.obs.drop(columns=['Sex'])

    geuvadis.write(FDIR_INTEMEDIATE / 'GEUVADIS.raw.h5ad')

    fname = next((FDIR_EXTERNAL / "HEART" / 'reg').glob("*TPM.txt"))
    heart = read_dataset(
        FDIR_EXTERNAL / "HEART" / 'reg' / fname,
        FDIR_EXTERNAL / "HEART" / 'reg' / 'SraRunTable.txt',
        FDIR_RAW / 'all_transcripts_strigtie_merged.gtf',
        separator="\t"
    )
    heart.write(FDIR_INTEMEDIATE / 'HEART.raw.h5ad')

    fname = next((FDIR_EXTERNAL / "BRAIN0" / 'reg').glob("*.csv"))
    brain_0 = read_dataset(
        FDIR_EXTERNAL / "BRAIN0" / 'reg' / fname,
        FDIR_EXTERNAL / "BRAIN0" / 'reg' / 'SraRunTable.txt',
        FDIR_RAW / 'all_transcripts_strigtie_merged.gtf',
        separator=","
    )
    brain_0.write(FDIR_INTEMEDIATE / 'BRAIN0.raw.h5ad')

    fname = next((FDIR_EXTERNAL / "BRAIN1" / 'reg').glob("*.csv"))
    brain_1 = read_dataset(
        FDIR_EXTERNAL / "BRAIN1" / 'reg' / fname,
        FDIR_EXTERNAL / "BRAIN1" / 'reg' / 'SraRunTable.txt',
        FDIR_RAW / 'all_transcripts_strigtie_merged.gtf',
        separator=","
    )
    brain_1.obs['sex'] = brain_1.obs['gender']
    brain_1.obs.drop(columns=['gender'])
    brain_1.write(FDIR_INTEMEDIATE / 'BRAIN1.raw.h5ad')

    # brain_0 = 0
    # brain1 = 0
