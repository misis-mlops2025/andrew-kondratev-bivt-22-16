from pathlib import Path

import pandas as pd
import typer
import yaml
from loguru import logger
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from typing import Tuple

from src.config import DataConfig, PROCESSED_DATA_DIR, TrainingConfig, CONFIG_PATH

app = typer.Typer()


def generate_dataset(config: DataConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic classification dataset"""
    x, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        n_redundant=config.n_redundant,
        n_repeated=config.n_repeated,
        n_classes=config.n_classes,
        random_state=config.random_state
    )

    feature_names = [f"feature_{i}" for i in range(config.n_features)]
    df = pd.DataFrame(x, columns=feature_names)
    target = pd.Series(y, name="target")

    return df, target

@app.command()
def main(
    output_dir: Path = PROCESSED_DATA_DIR,
    config_path: Path = CONFIG_PATH
):
    with open(config_path) as cfg_file:
        load_config = yaml.safe_load(cfg_file)
    config = TrainingConfig(**load_config).data

    x, y = generate_dataset(config)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )

    train_features_path = output_dir / config.train_features_filename
    train_labels_path = output_dir / config.train_labels_filename
    test_features_path = output_dir / config.test_features_filename
    test_labels_path = output_dir / config.test_labels_filename

    x_train.to_csv(train_features_path, index=False)
    y_train.to_csv(train_labels_path, index=False)
    x_test.to_csv(test_features_path, index=False)
    y_test.to_csv(test_labels_path, index=False)

    logger.info(f"Train features saved to {train_features_path}")
    logger.info(f"Train labels saved to {train_labels_path}")
    logger.info(f"Test features saved to {test_features_path}")
    logger.info(f"Test labels saved to {test_labels_path}")


if __name__ == "__main__":
    app()