
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Load the existing test predictions
df = pd.read_csv("results/test_predictions.csv")

# TweetEval sentiment labels
labels = [0, 1, 2]
names = ["Negative", "Neutral", "Positive"]

y_true = df["true_label"]
y_pred = df["custom_naive_bayes"]

print("CUSTOM NAIVE BAYES EVALUATION")
print("=" * 40)

print(f"Number of test samples: {len(df)}")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=names,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_true,
    y_pred,
    labels=labels
))


# Create and save the confusion matrix figure
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=names
)

fig, ax = plt.subplots(figsize=(7, 6))
display.plot(ax=ax, cmap="Blues", values_format="d")

ax.set_title("Custom Naive Bayes - Confusion Matrix")
fig.tight_layout()

fig.savefig(
    "report/figures/naive_bayes_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nConfusion matrix figure saved successfully!")


# Compare the performance of all three models
import json
import numpy as np

with open("results/summary.json", "r", encoding="utf-8") as file:
    summary = json.load(file)

test_results = summary["test_results"]

models = list(test_results.keys())
metrics = ["accuracy", "macro_recall", "macro_f1"]
metric_names = ["Accuracy", "Macro Recall", "Macro F1"]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    values = [test_results[model][metric] * 100 for model in models]

    ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=metric_name
    )

ax.set_xlabel("Classification Model")
ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Comparison on TweetEval")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 70)
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

fig.savefig(
    "report/figures/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Model comparison figure saved successfully!")


# Naive Bayes hyperparameter tuning analysis
validation_df = pd.read_csv("results/validation_results.csv")

nb_results = validation_df[
    validation_df["method"] == "Custom Naive Bayes"
].copy()

nb_results = nb_results.sort_values("value")

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    nb_results["value"],
    nb_results["accuracy"] * 100,
    marker="o",
    label="Accuracy"
)

ax.plot(
    nb_results["value"],
    nb_results["macro_recall"] * 100,
    marker="s",
    label="Macro Recall"
)

ax.plot(
    nb_results["value"],
    nb_results["macro_f1"] * 100,
    marker="^",
    label="Macro F1"
)

ax.axvline(
    x=0.5,
    color="gray",
    linestyle="--",
    label="Selected alpha = 0.5"
)

ax.set_xlabel("Smoothing Parameter (alpha)")
ax.set_ylabel("Validation Score (%)")
ax.set_title("Naive Bayes Hyperparameter Tuning")
ax.set_xticks(nb_results["value"])
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    "report/figures/naive_bayes_alpha_tuning.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Naive Bayes tuning figure saved successfully!")


# Error analysis: inspect misclassified tweets
errors = df[
    df["true_label"] != df["custom_naive_bayes"]
].copy()

print("\nERROR ANALYSIS")
print("=" * 40)
print(f"Total misclassified tweets: {len(errors)}")

# Focus on Neutral tweets predicted as Negative
neutral_to_negative = errors[
    (errors["true_label"] == 1) &
    (errors["custom_naive_bayes"] == 0)
]

print("\nNeutral tweets predicted as Negative:")
print(f"Number of cases: {len(neutral_to_negative)}")

# Save examples for manual analysis
neutral_to_negative.head(30).to_csv(
    "results/neutral_to_negative_examples.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Saved 30 examples for error analysis!")


# Sample different types of classification errors
# for qualitative error analysis

error_categories = [
    (0, 1, "Negative_to_Neutral"),
    (0, 2, "Negative_to_Positive"),
    (1, 0, "Neutral_to_Negative"),
    (1, 2, "Neutral_to_Positive"),
    (2, 0, "Positive_to_Negative"),
    (2, 1, "Positive_to_Neutral"),
]

sampled_errors = []

for true_label, predicted_label, category in error_categories:

    category_errors = df[
        (df["true_label"] == true_label) &
        (df["custom_naive_bayes"] == predicted_label)
    ].copy()

    # Sample up to 10 tweets per error category
    sample = category_errors.sample(
        n=min(10, len(category_errors)),
        random_state=323
    )

    sample["error_category"] = category

    sampled_errors.append(sample)

# Combine the sampled tweets
error_analysis_df = pd.concat(
    sampled_errors,
    ignore_index=True
)

# Add blank columns for manual analysis
error_analysis_df["possible_reason"] = ""
error_analysis_df["notes"] = ""

# Save the examples
error_analysis_df.to_csv(
    "results/error_analysis_samples.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSystematic error analysis completed!")
print(f"Total sampled errors: {len(error_analysis_df)}")
print("Saved results/error_analysis_samples.csv")
