from abc import ABC, abstractmethod
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from config.loader import DatasetConfig
from gtfparse import read_gtf
from loguru import logger
from utils.errors import UnsupportedFormatError


class BaseConverter(ABC):
    def __init__(self, config: DatasetConfig):
        self.fname_data: Path = Path(config.counts_file)
        self.fname_header: Path = Path(config.metadata_file)
        self.fname_annotator: Path = Path(config.annotation_file)

        self._suffices = [".csv", ".txt", ".tsv"]

        self._col_mapping = {"Sex": "sex", "gender": "sex", "Gender": "sex"}
        self._sex_mapping = {"M": "male", "Male": "male", "F": "female", "Female": "female"}

    def load_data(self):
        if self.fname_data.suffix not in self._suffices:
            raise UnsupportedFormatError(
                f"{self._suffices} are only supported formats! Reimplement `self.load_data(...)` module for {self.fname_data}"
            )
        data = pd.read_csv(self.fname_data, index_col=0, sep=None, engine="python").T
        return data.astype(np.float32)

    def load_header(self):

        if self.fname_header.suffix not in self._suffices:
            raise UnsupportedFormatError(
                f"{self._suffices} are only supported formats! Reimplement `self.load_header(...)` module for {self.fname_header}"
            )
        data = pd.read_csv(self.fname_header, index_col=0, sep=None, engine="python")

        if "sex" not in data.columns:
            logger.warning(f"No 'sex' column in {self.fname_header}!")
            column_name = pd.Index(self._col_mapping.keys()).intersection(data.columns)

            if len(column_name) == 0:
                raise KeyError(f"No {list(self._col_mapping)} found! Aborting")

            logger.warning(f"found following columns: `{list(column_name)}`")
            logger.warning(f"renaming `{column_name[0]}` to 'sex'!")
            data["sex"] = data[column_name[0]]
            data.drop(columns=[column_name[0]], inplace=True)

        return data

    def load_annotator(self):
        if not self.fname_annotator.suffix == ".gtf":
            raise UnsupportedFormatError(
                f".gtf `is only supported formats! Reimplement `self.load_annotator(...)` module for {self.fname_annotator}"
            )
        if self.fname_annotator.suffix == ".gtf":
            annotator = self._load_gtf(self.fname_annotator)

        return annotator

    def run(self) -> ad.AnnData:
        logger.info(f"read {self.fname_header}")
        data_header = self.load_header()
        logger.info(f"read {self.fname_data}")
        data_raw = self.load_data()
        logger.info(f"read {self.fname_annotator}")
        annotator = self.load_annotator()

        adata = self.to_anndata(data_raw, data_header, annotator)
        self._validate(adata)
        return adata

    def to_anndata(self, data_raw, data_header, annotator) -> ad.AnnData:
        columns = data_raw.columns.intersection(annotator.index)
        indices = data_raw.index.intersection(data_header.index)

        data_raw = data_raw.loc[indices, columns]
        data_header = data_header.loc[indices]
        annotator = annotator.loc[columns]

        adata = ad.AnnData(X=data_raw, obs=data_header, var=annotator)
        return adata

    def _load_gtf(self, fname: Path):
        gtf_rawdata = read_gtf(fname)
        gtf_data = gtf_rawdata.to_pandas()
        gtf_data = gtf_data.set_index("transcript_id")
        gtf_data["transcript_id"] = gtf_data.index
        gtf_data = gtf_data.drop_duplicates("transcript_id")
        return gtf_data

    # def _load_cage(self, fname_data: Path | str, fname_header: Path | str, fname_gtf: Path | str):
    #     data_raw = pd.read_csv((fname_data), sep="\t").T
    #     samples_annot = pd.read_excel(
    #         fname_header,
    #         parse_dates=False,
    #     )
    #     samples_annot.set_index("samples", inplace=True)
    #     samples_annot["donor"] = samples_annot["donor"].astype(str)

    #     genes_annot = pd.read_csv(fname_gtf)

    #     columns = data_raw.columns.intersection(genes_annot.index)
    #     indices = data_raw.index.intersection(samples_annot.index)

    #     data_raw = data_raw.loc[indices, columns]
    #     samples_annot = samples_annot.loc[indices]
    #     genes_annot = genes_annot.loc[columns]

    #     data_raw.columns = genes_annot["transcriptId"]
    #     genes_annot.set_index("transcriptId", inplace=True)

    #     adata = ad.AnnData(X=data_raw, obs=samples_annot, var=genes_annot)

    #     return adata

    def _validate(self, data):
        pass
