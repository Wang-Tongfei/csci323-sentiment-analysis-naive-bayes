import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Allow this test file to import files from the project folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_tweeteval_sentiment
from features import load_features
from naive_bayes import MultinomialNaiveBayes
from baselines import logistic_regression, linear_svm


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


def check_models():
    """Check that the Part 3B models can train and predict."""

    # Small example dataset used only for testing
    texts = [
        "good amazing",
        "great good",
        "bad terrible",
        "awful bad",
        "okay normal",
        "average okay",
    ]

    labels = np.array([2, 2, 0, 0, 1, 1])

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    # Check our own Naive Bayes
    our_nb = MultinomialNaiveBayes(alpha=1.0)
    our_nb.fit(X, labels)

    our_predictions = our_nb.predict(X)

    assert len(our_predictions) == len(labels)

    # Check that the learned values are valid numbers
    assert np.isfinite(our_nb.class_log_prior).all()
    assert np.isfinite(our_nb.feature_log_prob).all()

    # Compare our Naive Bayes with sklearn
    sklearn_nb = MultinomialNB(alpha=1.0)
    sklearn_nb.fit(X, labels)

    sklearn_predictions = sklearn_nb.predict(X)

    assert np.array_equal(
        our_predictions,
        sklearn_predictions
    )

    # Check Logistic Regression
    lr = logistic_regression(C=1.0)
    lr.fit(X, labels)

    lr_predictions = lr.predict(X)

    assert len(lr_predictions) == len(labels)

    # Check Linear SVM
    svm = linear_svm(C=0.1)
    svm.fit(X, labels)

    svm_predictions = svm.predict(X)

    assert len(svm_predictions) == len(labels)

    print("Model checks passed.")


if __name__ == "__main__":
    check_dataset()
    check_features()
    check_models()

    print("All sanity checks passed.")