from pathlib import Path

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class LogisticRegressionConfig(BaseModel):
    C: float = Field(1.0, description="Inverse of regularization strength")
    max_iter: int = Field(100, description="Maximum number of iterations")
    solver: str = Field("lbfgs", description="Solver algorithm")
    random_state: int = Field(42, description="Random state")


class DecisionTreeConfig(BaseModel):
    max_depth: Optional[int] = Field(None, description="Maximum depth of tree")
    min_samples_split: int = Field(2, description="Minimum samples required to split")
    min_samples_leaf: int = Field(1, description="Minimum samples required at leaf")
    random_state: int = Field(42, description="Random state")


class RandomForestConfig(BaseModel):
    n_estimators: int = Field(100, description="Number of trees in the forest")
    max_depth: Optional[int] = Field(None, description="Maximum depth of trees")
    min_samples_split: int = Field(2, description="Minimum samples required to split")
    min_samples_leaf: int = Field(1, description="Minimum samples required at leaf")
    random_state: int = Field(42, description="Random state")


class ModelType(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"


class DataConfig(BaseModel):
    n_samples: int = Field(1000, description="Number of samples to generate")
    n_features: int = Field(20, description="Number of features")
    n_informative: int = Field(15, description="Number of informative features")
    n_redundant: int = Field(2, description="Number of redundant features")
    n_repeated: int = Field(0, description="Number of repeated features")
    n_classes: int = Field(2, description="Number of classes")
    test_size: float = Field(0.2, description="Test set size ratio")
    random_state: int = Field(42, description="Random state")

    train_features_filename: str = Field("train_features.csv", description="Train features filename")
    train_labels_filename: str = Field("train_labels.csv", description="Train labels filename")
    test_features_filename: str = Field("test_features.csv", description="Test features filename")
    test_labels_filename: str = Field("test_labels.csv", description="Test labels filename")


class ModelConfig(BaseModel):
    model_type: ModelType = Field(ModelType.RANDOM_FOREST, description="Type of model to train")
    logistic_regression: LogisticRegressionConfig = Field(default_factory=LogisticRegressionConfig)
    decision_tree: DecisionTreeConfig = Field(default_factory=DecisionTreeConfig)
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)


class TrainingConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


PROJ_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJ_ROOT / "models"
CONFIGS_DIR = PROJ_ROOT / "configs"
MODEL_CONFIG_DIR = CONFIGS_DIR / "model_config"
DATA_DIR = PROJ_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CONFIG_PATH = CONFIGS_DIR / "config.yaml"