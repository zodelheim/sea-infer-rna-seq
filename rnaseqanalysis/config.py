from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path("/mnt/89ea702f-c213-4e9a-acb8-8c72a41b672e/Documents/PYTHON/RNASeqAnalysis/data")

FDIR_RAW = DATA_DIR / "raw"
FDIR_INTEMEDIATE = DATA_DIR / "interim"
FDIR_PROCESSED = DATA_DIR / "processed"
FDIR_EXTERNAL = DATA_DIR / "external"
