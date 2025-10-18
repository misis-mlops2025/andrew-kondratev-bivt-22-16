from src.dataset import generate_dataset
from src.config import DataConfig


def test_generate_dataset():
    """Test that generate_dataset creates correct dataset structure"""
    cfg = DataConfig(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        test_size=0.2,
        random_state=42,
        train_features_filename="train_features.csv",
        train_labels_filename="train_labels.csv",
        test_features_filename="test_features.csv",
        test_labels_filename="test_labels.csv",
    )

    x, y = generate_dataset(cfg)

    # Assert - Check shapes
    assert x.shape[0] == cfg.n_samples, f"Expected {cfg.n_samples} samples"
    assert x.shape[1] == cfg.n_features, f"Expected {cfg.n_features} features"
    assert len(y) == cfg.n_samples, f"Expected {cfg.n_samples} labels"
    # Assert - Check target classes
    unique_classes = y.unique()
    assert len(unique_classes) == cfg.n_classes, \
        f"Expected {cfg.n_classes} unique classes"
    assert set(unique_classes) == {0, 1}, "Binary classification should have classes 0 and 1"
    # Assert - Check no missing values
    assert not x.isnull().any().any(), "Features should not contain NaN values"
    assert not y.isnull().any(), "Labels should not contain NaN values"

