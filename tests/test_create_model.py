from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.modeling.model import create_model
from src.config import ModelConfig, ModelType


def test_create_logistic_regression():
    """Test creating logistic regression model"""
    config = ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)

    model = create_model(config)

    assert isinstance(model, LogisticRegression)
    assert model.random_state == 42


def test_create_random_forest():
    """Test creating random forest model"""
    config = ModelConfig(model_type=ModelType.RANDOM_FOREST)

    model = create_model(config)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100


def test_create_decision_tree():
    """Test creating decision tree model"""
    config = ModelConfig(model_type=ModelType.DECISION_TREE)

    model = create_model(config)

    assert isinstance(model, DecisionTreeClassifier)
    assert model.min_samples_split == 2