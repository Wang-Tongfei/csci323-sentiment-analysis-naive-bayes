
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score

from features import load_features
from naive_bayes import MultinomialNaiveBayes

ALPHA = 0.5

print("ABLATION STUDY")
print("=" * 40)

features = load_features()

train_texts = features["train_texts"]
val_texts = features["val_texts"]

y_train = features["y_train"]
y_val = features["y_val"]

print(f"Training samples: {len(y_train)}")
print(f"Validation samples: {len(y_val)}")
print("Data loaded successfully!")


# Compare different Bag-of-Words feature settings
configurations = [
    ("Original Bag-of-Words", (1, 1), 1),
    ("Unigrams + Bigrams", (1, 2), 1),
    ("Minimum frequency = 2", (1, 1), 2),
]

results = []

for name, ngram_range, min_df in configurations:
    print(f"\nRunning: {name}")

    # Learn vocabulary from training tweets only
    vectorizer = CountVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    # Train the custom Naive Bayes classifier
    model = MultinomialNaiveBayes(alpha=ALPHA)
    model.fit(X_train, y_train)

    # Evaluate using validation data only
    predictions = model.predict(X_val)

    macro_recall = recall_score(
        y_val,
        predictions,
        average="macro"
    )

    results.append({
        "configuration": name,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "validation_macro_recall": macro_recall
    })

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Validation macro recall: {macro_recall:.4f}")

# Save the results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/ablation_results.csv",
    index=False
)

print("\nABLATION STUDY COMPLETED!")
print(results_df)
print("Saved results/ablation_results.csv")


# Create ablation study comparison graph
fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    results_df["configuration"],
    results_df["validation_macro_recall"] * 100,
    color=["steelblue", "orange", "seagreen"]
)

# Display percentage above each bar
for bar in bars:
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.5,
        f"{height:.2f}%",
        ha="center"
    )

ax.set_xlabel("Feature Configuration")
ax.set_ylabel("Validation Macro Recall (%)")
ax.set_title("Naive Bayes Feature Ablation Study")
ax.set_ylim(0, 70)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

fig.savefig(
    "report/figures/naive_bayes_ablation_study.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved report/figures/naive_bayes_ablation_study.png")
