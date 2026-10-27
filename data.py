from collections import Counter
from datasets import load_dataset


# Sentiment labels used by TweetEval
LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Dataset parts provided by TweetEval
SPLITS = ["train", "validation", "test"]


def load_tweeteval_sentiment():
    """
    Load the TweetEval sentiment dataset.

    TweetEval already gives us:
    - train data
    - validation data
    - test data
    """
    return load_dataset("cardiffnlp/tweet_eval", "sentiment")


def get_texts_and_labels(dataset, split):
    """
    Get the text and labels from one split.
    """
    texts = list(dataset[split]["text"])
    labels = list(dataset[split]["label"])

    return texts, labels


def inspect_split(split, split_name):
    """
    Check basic information about one dataset split.
    """

    texts = split["text"]
    labels = split["label"]

    # Remove extra spaces so empty tweets can be detected
    cleaned_texts = [
        text.strip() if text is not None else ""
        for text in texts
    ]

    # Count empty tweets
    empty_texts = sum(
        1 for text in cleaned_texts
        if text == ""
    )

    # Count repeated tweets
    duplicate_texts = (
        len(cleaned_texts)
        - len(set(cleaned_texts))
    )

    # Check if any label is not 0, 1 or 2
    invalid_labels = sum(
        1 for label in labels
        if label not in LABEL_NAMES
    )

    # Count examples in each sentiment class
    class_counts = Counter(labels)

    print(f"\n--- {split_name.upper()} ---")
    print(f"Number of examples: {len(texts)}")
    print(f"Empty texts: {empty_texts}")
    print(f"Duplicate texts: {duplicate_texts}")
    print(f"Missing/invalid labels: {invalid_labels}")

    print("Class distribution:")

    for label_id in sorted(class_counts):
        label_name = LABEL_NAMES.get(label_id, "INVALID")
        count = class_counts[label_id]

        print(
            f"  {label_id} ({label_name}): "
            f"{count} ({count / len(labels):.1%})"
        )

    return {
        "examples": len(texts),
        "empty_texts": empty_texts,
        "duplicate_texts": duplicate_texts,
        "invalid_labels": invalid_labels,
        "class_counts": {
            LABEL_NAMES.get(k, "INVALID"): v
            for k, v in sorted(class_counts.items())
        }
    }


def check_split_overlap(dataset):
    """
    Check if the same tweet appears in more than one split.

    This helps us check for possible data leakage.
    """

    train_texts = set(
        text.strip()
        for text in dataset["train"]["text"]
    )

    validation_texts = set(
        text.strip()
        for text in dataset["validation"]["text"]
    )

    test_texts = set(
        text.strip()
        for text in dataset["test"]["text"]
    )

    train_validation_overlap = len(
        train_texts & validation_texts
    )

    train_test_overlap = len(
        train_texts & test_texts
    )

    validation_test_overlap = len(
        validation_texts & test_texts
    )

    print("\n--- SPLIT OVERLAP CHECK ---")
    print(f"Train & Validation: {train_validation_overlap}")
    print(f"Train & Test: {train_test_overlap}")
    print(f"Validation & Test: {validation_test_overlap}")

    return {
        "train_validation": train_validation_overlap,
        "train_test": train_test_overlap,
        "validation_test": validation_test_overlap
    }


def main():

    print("Loading TweetEval sentiment dataset...")

    dataset = load_tweeteval_sentiment()

    # Make sure the expected dataset parts exist
    for split_name in SPLITS:

        assert split_name in dataset, (
            f"Missing split: {split_name}"
        )

        assert {"text", "label"} <= set(
            dataset[split_name].column_names
        )

    # Check train, validation and test data
    for split_name in SPLITS:
        inspect_split(
            dataset[split_name],
            split_name
        )

    # Check if identical tweets appear in different splits
    check_split_overlap(dataset)


if name == "main":
    main()