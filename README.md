# RNASeqAnalysis

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

# Code Usage:
Firstly you should setup the environment
It is highly recomended to install/use [uv](https://docs.astral.sh/uv/) as python environment management tool

Install `uv` and then do in terminal:
```bash
uv sync
```
to install all project dependences.

Then you need to export `PYTHONPATH` variable.
In general 
```bash
export PYTHONPATH=./rnaseqanalysis
```
should work

Additionally you should change paths (`DATADIR`) in [rnaseqanalysis/config.py](rnaseqanalysis/config.py) file

Pipeline is following:

- [convert_to_anndata.py](rnaseqanalysis/preprocessing/convert_to_anndata.py) to convert all raw data to unified [anndata](https://anndata.readthedocs.io/en/latest/) format
- [make_train_dataset.py](rnaseqanalysis/preprocessing/make_train_dataset.py) to create train dataset based on geuvadis data
[feature_ranging.py](rnaseqanalysis/models/feature_ranging.py) - for transcripts ranging based on [SHAP](https://shap.readthedocs.io/en/latest/) index
[feature_selection.py](rnaseqanalysis/models/feature_selection.py) - plot boxplot for manual N features selection


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources: contains datasets to evaluate model on.
│   ├── interim        <- Intermediate data that has been transformed, feature importance data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable geuvadis data.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for rnaseqanalysis
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── setup.cfg          <- Configuration file for flake8
│
└── rnaseqanalysis                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes rnaseqanalysis a Python module
    │
    ├── preprocessing  <- Scripts to turn raw data into features for modeling
    │   ├── __init__.py
    │   └── ...
    │
    ├── models         <- Scripts to make feature selection, feature ranging, to train models and then use trained models to make
    │   │                 predictions
    │   ├── __init__.py
    │   └── ...
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── ...
```

