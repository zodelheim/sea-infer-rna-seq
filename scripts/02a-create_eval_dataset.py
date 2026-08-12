import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from config.loader import Config, load_yaml
from loguru import logger
from preprocessing.base_converter import BaseConverter
from preprocessing.transforms import *
from rich.console import Console
from rich.logging import RichHandler
from utils.errors import *

logger.remove()
logger.add(
    RichHandler(markup=True, rich_tracebacks=True),
    level="INFO",
    format="{message}",
)

console = Console()


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

cfg = Config(**load_yaml(args.config))


for name in cfg.models.eval_dataset:
    adata = ad.read_h5ad(cfg.paths.interim_path / f"{name}.raw.h5ad")

    console.rule(f"Creating train dataset from {name}")

    datasets = {}
    datasets["RAW"] = adata.to_df()

    columns_ = pd.Index([])

    for key, data_raw in datasets.items():
        logger.info(f"{name}:'{key}' has {len(data_raw.columns)} transcripts")

        with logger.contextualize(dataset=name, key=key):
            data_ = data_raw.pipe(filter_zero_median)

        logger.info(f"store {len(data_.columns)} transcripts")
        columns_ = columns_.union(data_.columns)

    adata = adata[:, columns_]
    logger.info(f"shape after {cfg.models.split_by}-wise filtration: {adata.shape=}")

    with logger.contextualize(dataset=name):
        data = adata.to_df().pipe(logarithmization)

    data = data.astype(np.float32)
    adata = adata[data.index, data.columns]
    adata.layers["raw"] = adata.X.copy()
    adata.X = data.values

    logger.info(f"{name} labeling {cfg.models.value_to_predict} transcripts")
    adata = labeling_sex_transcripts(adata)

    adata.write(
        cfg.paths.processed_path / f"{name.upper()}.preprocessed.{cfg.models.value_to_predict}.h5ad"
    )
