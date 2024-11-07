from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]


FDIR_RAW = PROJ_ROOT / "data/raw/"
FDIR_INTEMEDIATE = PROJ_ROOT / "data/interim"
FDIR_PROCESSED = PROJ_ROOT / "data/processed"
FDIR_EXTERNAL = PROJ_ROOT / "data/external"
