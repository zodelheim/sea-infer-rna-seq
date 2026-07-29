from pathlib import Path
from pydantic import BaseModel, model_validator, field_validator
import yaml
import json


class PathsConfig(BaseModel):
    """Paths config class"""

    workdir: Path = Path("./data")
    interim_path: Path = workdir / "interim"
    processed_path: Path = workdir / "processed"
    external_path: Path = workdir / "external"

    models: Path = Path("./models")
    results: Path = Path("./results")
    logs: Path = Path("./logs")

    @model_validator(mode="after")
    def make_absolute(self):
        self.workdir = self.workdir.absolute()
        self.interim_path = self.interim_path.absolute()
        self.processed_path = self.processed_path.absolute()
        self.external_path = self.external_path.absolute()
        self.models = self.models.absolute()
        self.results = self.results.absolute()
        self.logs = self.logs.absolute()

        self.workdir.mkdir(exist_ok=True)
        self.interim_path.mkdir(exist_ok=True)
        self.processed_path.mkdir(exist_ok=True)
        self.external_path.mkdir(exist_ok=True)
        self.models.mkdir(exist_ok=True)
        self.results.mkdir(exist_ok=True)
        self.logs.mkdir(exist_ok=True)

        return self


class DatasetConfig(BaseModel):
    """Datasets config class"""

    path: Path
    counts_file: Path
    metadata_file: Path
    annotation_file: Path

    @model_validator(mode="after")
    def make_absolute(self):
        self.counts_file = self.path / self.counts_file
        self.metadata_file = self.path / self.metadata_file

        if not self.annotation_file.is_absolute():
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

    model_type: str
    feature_importance_method: str
    config_file: Path
    use_CV: bool = True

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
