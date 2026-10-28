import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from data import get_texts_and_labels, load_tweeteval_sentiment


def build_bag_of_words(train_texts, validation_texts, test_texts, min_df=1):
    """
    Convert tweets into word-count features.

    The vocabulary is learned from the training data only.
    Validation and test data use the same vocabulary.
    """

    vectorizer = CountVectorizer(
        lowercase=True,
        min_df=min_df
    )

    # Learn the vocabulary from training data
    X_train = vectorizer.fit_transform(train_texts)

    # Use the same vocabulary for validation and test data
    X_validation = vectorizer.transform(validation_texts)
    X_test = vectorizer.transform(test_texts)

    return vectorizer, X_train, X_validation, X_test


def load_features(min_df=1):
    """
    Load TweetEval and prepare features for all three data splits.
    """

    dataset = load_tweeteval_sentiment()

    train_texts, y_train = get_texts_and_labels(
        dataset,
        "train"
    )

    val_texts, y_val = get_texts_and_labels(
        dataset,
        "validation"
    )

    test_texts, y_test = get_texts_and_labels(
        dataset,
        "test"
    )

    vectorizer, X_train, X_val, X_test = build_bag_of_words(
        train_texts,
        val_texts,
        test_texts,
        min_df=min_df
    )

    return {
        "vectorizer": vectorizer,

        "X_train": X_train,
        "y_train": np.array(y_train),

        "X_val": X_val,
        "y_val": np.array(y_val),

        "X_test": X_test,
        "y_test": np.array(y_test),

        "train_texts": train_texts,
        "val_texts": val_texts,
        "test_texts": test_texts
    }


def main():

    features = load_features()

    vectorizer = features["vectorizer"]

    print(
        "Vocabulary size:",
        len(vectorizer.vocabulary_)
    )

    # Show the size of each feature matrix
    for name in ["train", "val", "test"]:

        X = features[f"X_{name}"]
        y = features[f"y_{name}"]

        print(
            f"X_{name}: {X.shape}, "
            f"labels: {len(y)}"
        )

    # Show the word counts for one example tweet
    example = features["train_texts"][0]
    row = features["X_train"][0]

    words = vectorizer.get_feature_names_out()[
        row.indices
    ]

    print("\nExample tweet:")
    print(example)

    print("\nWord-count features:")
    print(
        dict(
            zip(words, row.data)
        )
    )


if name == "main":
    main()