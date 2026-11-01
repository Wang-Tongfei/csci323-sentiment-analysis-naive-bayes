
import numpy as np

from features import load_features
from naive_bayes import MultinomialNaiveBayes

print("TOP-WEIGHTED WORDS ANALYSIS")
print("=" * 40)

# Load the original Bag-of-Words features
features = load_features()

X_train = features["X_train"]
y_train = features["y_train"]
vectorizer = features["vectorizer"]

# Train the original custom Naive Bayes model
model = MultinomialNaiveBayes(alpha=0.5)
model.fit(X_train, y_train)

print("Model trained successfully!")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# Inspect model attributes before extracting word weights
print("\nAvailable model attributes:")
print([
    name for name in vars(model)
    if not name.startswith("_")
])


import pandas as pd

# Get vocabulary words in the correct feature order
feature_names = vectorizer.get_feature_names_out()

# Get the learned word log probabilities
log_probs = model.feature_log_prob
classes = model.classes

class_names = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

results = []

for i, class_label in enumerate(classes):
    class_name = class_names[int(class_label)]

    # Compare each word's probability in this class
    # against its average probability in the other classes
    other_indices = [j for j in range(len(classes)) if j != i]

    scores = (
        log_probs[i]
        - log_probs[other_indices].mean(axis=0)
    )

    # Find the 15 most distinctive words
    top_indices = np.argsort(scores)[-15:][::-1]

    print(f"\nTop 15 distinctive words for {class_name}:")
    print("-" * 40)

    for rank, index in enumerate(top_indices, start=1):
        word = feature_names[index]
        score = float(scores[index])

        print(f"{rank}. {word} (score: {score:.4f})")

        results.append({
            "sentiment": class_name,
            "rank": rank,
            "word": word,
            "distinctiveness_score": score
        })

# Save results for the report
results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/top_weighted_words.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTop-weighted words analysis completed!")
print("Saved results/top_weighted_words.csv")
