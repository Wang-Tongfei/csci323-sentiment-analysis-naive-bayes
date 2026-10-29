import sys
from pathlib import Path

# Allow this test file to import files from the project folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_tweeteval_sentiment
from features import load_features


def check_dataset():
    """Check the TweetEval dataset structure and labels."""

    dataset = load_tweeteval_sentiment()

    for split_name in ["train", "validation", "test"]:
        assert split_name in dataset
        assert "text" in dataset[split_name].column_names
        assert "label" in dataset[split_name].column_names

        labels = dataset[split_name]["label"]

        assert all(
            label in [0, 1, 2]
            for label in labels
        )

    print("Dataset checks passed.")


def check_features():
    """Check the Bag-of-Words feature matrices."""

    features = load_features()

    X_train = features["X_train"]
    X_val = features["X_val"]
    X_test = features["X_test"]

    y_train = features["y_train"]
    y_val = features["y_val"]
    y_test = features["y_test"]

    # Number of rows should match the number of labels
    assert X_train.shape[0] == len(y_train)
    assert X_val.shape[0] == len(y_val)
    assert X_test.shape[0] == len(y_test)

    # All splits should use the same feature columns
    assert X_train.shape[1] == X_val.shape[1]
    assert X_train.shape[1] == X_test.shape[1]

    print("Feature checks passed.")


if __name__ == "__main__":
    check_dataset()
    check_features()