import argparse
from pathlib import Path

from config.loader import Config, load_yaml
from preprocessing.base_converter import BaseConverter
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler


logger.remove()
logger.add(
    RichHandler(markup=True, rich_tracebacks=True),
    level="INFO",
    format="{message}",
)


console = Console()
logs_dir = Path("./logs")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--exp_id", default="001")
args = parser.parse_args()

cfg = Config(**load_yaml(args.config))
cfg_dict = cfg.model_dump()

logs_dir = logs_dir / args.exp_id
logs_dir.mkdir(exist_ok=True)

# console.log("Current configuration")
# console.log(cfg.paths)
console.log("Using following dataset paths:")
console.log(cfg.datasets)

for name in cfg.datasets:
    console.rule(f"Preprocessing [bold red]{name}[/bold red] dataset")
    converter = BaseConverter(cfg.datasets[name])
    anndata = converter.run()
    anndata.write(cfg.paths.interim_path / f"{name}.raw.h5ad")
