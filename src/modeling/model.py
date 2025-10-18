from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.config import ModelConfig, ModelType


def create_model(config: ModelConfig):
    """Create model based on configuration"""
    model_type = config.model_type

    if model_type == ModelType.LOGISTIC_REGRESSION:
        return LogisticRegression(
            C=config.logistic_regression.C,
            max_iter=config.logistic_regression.max_iter,
            solver=config.logistic_regression.solver,
            random_state=config.logistic_regression.random_state,
        )
    if model_type == ModelType.DECISION_TREE:
        return DecisionTreeClassifier(
            max_depth=config.decision_tree.max_depth,
            min_samples_split=config.decision_tree.min_samples_split,
            min_samples_leaf=config.decision_tree.min_samples_leaf,
            random_state=config.decision_tree.random_state,
        )
    if model_type == ModelType.RANDOM_FOREST:
        return RandomForestClassifier(
            n_estimators=config.random_forest.n_estimators,
            max_depth=config.random_forest.max_depth,
            min_samples_split=config.random_forest.min_samples_split,
            min_samples_leaf=config.random_forest.min_samples_leaf,
            random_state=config.random_forest.random_state,
        )

    raise ValueError(f"Unknown model type: {model_type}")
