from dataclasses import dataclass
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Paths:
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    outputs_dir: Path = ROOT / "outputs"
    metrics_dir: Path = outputs_dir / "metrics"
    figures_dir: Path = outputs_dir / "figures"
    tables_dir: Path = outputs_dir / "tables"
    config_path: Path = ROOT / "config" / "model_config.yaml"

def load_cfg(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(paths: Paths):
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
