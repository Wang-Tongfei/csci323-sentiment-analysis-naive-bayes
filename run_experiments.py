from pathlib import Path

import numpy as np

from features import load_features


# Use the same seed throughout the project
RANDOM_SEED = 323

# Folder where experiment results will be saved
RESULTS_DIR = Path("results")


def main():
    np.random.seed(RANDOM_SEED)

    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading features...")
    features = load_features()

    X_train = features["X_train"]
    X_val = features["X_val"]
    X_test = features["X_test"]

    y_train = features["y_train"]
    y_val = features["y_val"]
    y_test = features["y_test"]

    print("Features loaded successfully.")
    print(f"X_train: {X_train.shape}, labels: {len(y_train)}")
    print(f"X_val: {X_val.shape}, labels: {len(y_val)}")
    print(f"X_test: {X_test.shape}, labels: {len(y_test)}")
    print(f"Random seed: {RANDOM_SEED}")


if __name__ == "__main__":
    main()