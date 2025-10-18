import pandas as pd
import typer
import yaml
import joblib
import json
from pathlib import Path

from loguru import logger

from src.config import TrainingConfig, CONFIGS_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from src.modeling.model import create_model


app = typer.Typer()


class Trainer:
    def __init__(self, config: TrainingConfig, features_path: Path, labels_path: Path):
        self.config = config
        self.model = create_model(self.config.model)
        self.features_path = features_path
        self.labels_path = labels_path

    def train(self):
        """Train the model"""
        x, y = pd.read_csv(self.features_path), pd.read_csv(self.labels_path).squeeze()
        self.model.fit(x, y)
        logger.info(f"Model trained")

    def save_model(self):
        """Save trained model and metadata"""
        output_dir = MODELS_DIR / f"{self.config.model.model_type.value}"
        output_dir.mkdir(exist_ok=True)

        model_filename = output_dir / f"model.joblib"
        config_filename = output_dir / f"config.json"

        joblib.dump(self.model, model_filename)
        with open(config_filename, 'w') as f:
            json.dump(self.config.model_dump(), f, indent=2, default=str)

        logger.info(f"Model saved to {model_filename}")
        logger.info(f"Config saved to {config_filename}")


@app.command()
def main(
    config_path: Path = CONFIGS_DIR / "config.yaml",
    features_path: Path = PROCESSED_DATA_DIR / "train_features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "train_labels.csv",
):
    with open(config_path) as cfg_file:
        load_config = yaml.safe_load(cfg_file)

    config = TrainingConfig(**load_config)
    trainer = Trainer(config, features_path, labels_path)
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    app()