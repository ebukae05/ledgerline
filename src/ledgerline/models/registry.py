"""Save and load versioned fraud models.

Each model lives in artifacts/models/<version>/ with two files:
- model.txt: the LightGBM trees
- meta.json: everything needed to use and audit it (features, category
  codes, flag threshold, validation metrics, git commit, library versions)

The version is a hash of the trees, so retraining the same code on the same
data produces the same version, and any change produces a new one.
"""

import hashlib
import json
from importlib.metadata import version as package_version
from pathlib import Path

import lightgbm as lgb

from ledgerline.data.load import PROJECT_ROOT
from ledgerline.features.pipeline import CategoryEncoder
from ledgerline.models.lgbm import FraudModel

MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
LATEST = "LATEST"
TRACKED_PACKAGES = ["lightgbm", "polars", "numpy", "scikit-learn"]


def model_version(model: FraudModel) -> str:
    trees = model.booster.model_to_string()
    return "lgbm-" + hashlib.sha256(trees.encode()).hexdigest()[:10]


def save_model(model: FraudModel, meta: dict, models_dir: Path = MODELS_DIR) -> Path:
    version = model_version(model)
    folder = models_dir / version
    folder.mkdir(parents=True, exist_ok=True)
    model.booster.save_model(folder / "model.txt")
    full_meta = {
        "version": version,
        "features": model.features,
        "category_vocab": model.encoder.vocab,
        "library_versions": {p: package_version(p) for p in TRACKED_PACKAGES},
        **meta,
    }
    (folder / "meta.json").write_text(json.dumps(full_meta, indent=2) + "\n")
    (models_dir / LATEST).write_text(version + "\n")
    return folder


def load_model(
    version: str | None = None, models_dir: Path = MODELS_DIR
) -> tuple[FraudModel, dict]:
    """Load a model by version, or the most recently trained one."""
    if version is None:
        latest = models_dir / LATEST
        if not latest.exists():
            raise FileNotFoundError(
                f"No trained model in {models_dir}. Run `python -m ledgerline.train` first."
            )
        version = latest.read_text().strip()
    folder = models_dir / version
    if not (folder / "model.txt").exists():
        raise FileNotFoundError(f"Model {version} not found in {models_dir}")

    meta = json.loads((folder / "meta.json").read_text())
    booster = lgb.Booster(model_file=folder / "model.txt")
    encoder = CategoryEncoder(vocab=meta["category_vocab"])
    return FraudModel(booster, encoder, meta["features"]), meta
