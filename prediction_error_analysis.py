import pandas as pd

df = pd.read_csv("results/test_predictions.csv")
names = {0: "Negative", 1: "Neutral", 2: "Positive"}

errors = df[df["true_label"] != df["custom_naive_bayes"]].copy()
errors["row_id"] = errors.index
errors["error_category"] = (
    errors["true_label"].map(names) + "_to_" + errors["custom_naive_bayes"].map(names)
)

lr_ok = errors["logistic_regression"] == errors["true_label"]
svm_ok = errors["linear_svm"] == errors["true_label"]
errors["error_scope"] = "All three wrong"
errors.loc[lr_ok & svm_ok, "error_scope"] = "NB only"
errors.loc[lr_ok & ~svm_ok, "error_scope"] = "NB and SVM wrong"
errors.loc[~lr_ok & svm_ok, "error_scope"] = "NB and LR wrong"

print(errors["error_category"].value_counts())


sample = errors.groupby("error_category").sample(n=10, random_state=323)
sample["possible_reason"] = ""
sample["notes"] = ""
sample.to_csv("results/error_analysis_samples.csv", index=False, encoding="utf-8-sig")
