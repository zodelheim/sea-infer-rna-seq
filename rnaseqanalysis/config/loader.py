from pathlib import Path
from pydantic import BaseModel, model_validator, field_validator
import yaml
import json
from enum import StrEnum


class ModelsEnum(StrEnum):
    xgboost = "xgboost"
    catboost = "catboost"
    knn = "knn"
    logreg = "logreg"


class FeatureSelectionMethodsEnum(StrEnum):
    top50 = "top50"
    top100 = "top100"
    elbow = "elbow"


class PathsConfig(BaseModel):
    """Paths config class"""

    workdir: Path = Path("./data")
    interim_path: Path = None
    processed_path: Path = None
    external_path: Path = None

    models: Path = Path("./models")
    results: Path = Path("./results")
    logs: Path = Path("./logs")
    figures: Path = Path("./figures")

    @model_validator(mode="after")
    def make_absolute(self):
        self.interim_path: Path = self.workdir / "interim"
        self.processed_path: Path = self.workdir / "processed"
        self.external_path: Path = self.workdir / "external"

        self.workdir = self.workdir.absolute()
        self.interim_path = self.interim_path.absolute()
        self.processed_path = self.processed_path.absolute()
        self.external_path = self.external_path.absolute()
        self.models = self.models.absolute()
        self.results = self.results.absolute()
        self.logs = self.logs.absolute()
        self.figures = self.figures.absolute()

        self.workdir.mkdir(exist_ok=True)
        self.interim_path.mkdir(exist_ok=True)
        self.processed_path.mkdir(exist_ok=True)
        self.external_path.mkdir(exist_ok=True)
        self.models.mkdir(exist_ok=True)
        self.results.mkdir(exist_ok=True)
        self.logs.mkdir(exist_ok=True)
        self.figures.mkdir(exist_ok=True)

        return self


class DatasetConfig(BaseModel):
    """Datasets config class"""

    path: Path | None = None
    counts_file: Path | None = None
    metadata_file: Path | None = None
    annotation_file: Path | None = None

    @model_validator(mode="after")
    def make_absolute(self):
        if not isinstance(self.counts_file, Path):
            raise FileNotFoundError(f"No such file {self.counts_file}")
        if not isinstance(self.metadata_file, Path):
            raise FileNotFoundError(f"No such file {self.metadata_file}")
        if not isinstance(self.annotation_file, Path):
            raise FileNotFoundError(f"No such file {self.annotation_file}")

        if not self.counts_file.is_absolute() and self.path:
            self.counts_file = self.path / self.counts_file
        if not self.metadata_file.is_absolute() and self.path:
            self.metadata_file = self.path / self.metadata_file
        if not self.annotation_file.is_absolute() and self.path:
            self.annotation_file = self.path / self.annotation_file

        return self


class ModelsConfig(BaseModel):
    """Preprocessing config class"""

    train_dataset: str
    eval_dataset: str | list[str]
    value_to_predict: str
    split_by: str


class MLModelParamsConfig(BaseModel):
    "Configure ML model"

    model_type: ModelsEnum = ModelsEnum.xgboost
    feature_importance_method: str
    config_file: Path
    use_CV: bool = True
    feature_selection_method: FeatureSelectionMethodsEnum = FeatureSelectionMethodsEnum.top50

    @model_validator(mode="after")
    def check_model_conf(self):
        if not self.config_file.absolute().is_file():
            raise FileNotFoundError(f"No such file {self.config_file.absolute()}")

        return self


class Config(BaseModel):
    paths: PathsConfig
    datasets: dict[str, DatasetConfig]
    models: ModelsConfig
    model_params: MLModelParamsConfig

    @model_validator(mode="after")
    def validate_data(self):
        if self.models.train_dataset not in self.datasets:
            raise ValueError(
                f"Training dataset '{self.models.train_dataset}' "
                f"not found. Available datasets: {list(self.datasets.keys())}"
            )
        eval_datasets = self.models.eval_dataset
        if isinstance(eval_datasets, str):
            eval_datasets = [eval_datasets]

        missing = set(eval_datasets) - self.datasets.keys()

        if missing:
            raise ValueError(f"Evaluation datasets not found: {missing}")

        return self


def load_yaml(filename: str | Path):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    cfg = Config(**load_yaml("config.yaml"))
    print(cfg.paths.workdir)
    print(cfg.datasets["BRAIN0"])
