# pylint: disable=no-member
from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from sklearn.metrics import log_loss
import typer
import yaml

from src.config import CONFIG_PATH, MODELS_DIR, PROCESSED_DATA_DIR, TrainingConfig, DataConfig

app = typer.Typer()


@app.command()
def main(
    output_dir: Path = PROCESSED_DATA_DIR,
    config_path: Path = CONFIG_PATH,
):
    """Evaluate the model"""
    with open(config_path, encoding='UTF-8') as cfg_file:
        load_config = yaml.safe_load(cfg_file)
    config = TrainingConfig(**load_config)
    data_config: DataConfig = config.data
    test_features_path = output_dir / data_config.test_features_filename
    test_labels_path = output_dir / data_config.test_labels_filename
    x_test, y_test = pd.read_csv(test_features_path), pd.read_csv(test_labels_path)
    logger.info("Test features and labels loaded")

    model = joblib.load(MODELS_DIR / f"{config.model.model_type.value}/model.joblib")
    logger.info(f"Model {config.model.model_type} loaded")

    predictions = model.predict_proba(x_test)

    loss = log_loss(y_test, predictions)
    logger.info(f"Loss: {loss}")


if __name__ == "__main__":
    app()
