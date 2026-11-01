
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import recall_score

from features import load_features
from naive_bayes import MultinomialNaiveBayes

# Use the same settings as our main experiment
RANDOM_SEED = 323
ALPHA = 0.5

# Different training-set sizes
TRAINING_PERCENTAGES = [20, 40, 60, 80, 100]

print("LEARNING CURVE EXPERIMENT")
print("=" * 40)

# Load the existing Bag-of-Words features
features = load_features()

X_train = features["X_train"]
y_train = features["y_train"]

X_val = features["X_val"]
y_val = features["y_val"]

# Shuffle training examples reproducibly
rng = np.random.default_rng(RANDOM_SEED)
indices = rng.permutation(len(y_train))

results = []

for percentage in TRAINING_PERCENTAGES:
    n_samples = int(len(y_train) * percentage / 100)
    selected_indices = indices[:n_samples]

    X_subset = X_train[selected_indices]
    y_subset = y_train[selected_indices]

    model = MultinomialNaiveBayes(alpha=ALPHA)
    model.fit(X_subset, y_subset)

    predictions = model.predict(X_val)

    macro_recall = recall_score(
        y_val,
        predictions,
        average="macro"
    )

    results.append({
        "training_percentage": percentage,
        "training_samples": n_samples,
        "validation_macro_recall": macro_recall
    })

    print(
        f"{percentage}% training data "
        f"({n_samples} tweets): "
        f"Macro Recall = {macro_recall:.4f}"
    )

# Save experiment results separately
results_df = pd.DataFrame(results)
results_df.to_csv(
    "results/learning_curve_results.csv",
    index=False
)

# Create the learning-curve graph
plt.figure(figsize=(8, 5))

plt.plot(
    results_df["training_percentage"],
    results_df["validation_macro_recall"] * 100,
    marker="o"
)

plt.xlabel("Training Data Used (%)")
plt.ylabel("Validation Macro Recall (%)")
plt.title("Naive Bayes Learning Curve")
plt.xticks(TRAINING_PERCENTAGES)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "report/figures/naive_bayes_learning_curve.png",
    dpi=300
)

plt.close()

print("\nLearning curve experiment completed!")
print("Saved results/learning_curve_results.csv")
print("Saved report/figures/naive_bayes_learning_curve.png")
