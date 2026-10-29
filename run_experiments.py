from pathlib import Path
import csv
import json

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

from features import load_features
from naive_bayes import MultinomialNaiveBayes
from baselines import (
    sklearn_naive_bayes,
    logistic_regression,
    linear_svm,
)


# Use the same seed throughout the project
RANDOM_SEED = 323

# Folder where experiment results will be saved
RESULTS_DIR = Path("results")

# We use macro recall to choose the best validation setting
SELECTION_METRIC = "macro_recall"


def calculate_metrics(y_true, y_pred):
    """Calculate the main evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro")
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro")
        ),
    }


def check_custom_naive_bayes(X_train, y_train, X_val, y_val):
    """
    Compare our Naive Bayes implementation with sklearn.

    Both models use the same alpha value and the same data.
    Their results should be almost identical.
    """
    alpha = 1.0

    custom_model = MultinomialNaiveBayes(alpha=alpha)
    sklearn_model = sklearn_naive_bayes(alpha=alpha)

    custom_model.fit(X_train, y_train)
    sklearn_model.fit(X_train, y_train)

    custom_predictions = custom_model.predict(X_val)
    sklearn_predictions = sklearn_model.predict(X_val)

    custom_metrics = calculate_metrics(y_val, custom_predictions)
    sklearn_metrics = calculate_metrics(y_val, sklearn_predictions)

    agreement = np.mean(custom_predictions == sklearn_predictions)

    prior_difference = np.max(
        np.abs(
            custom_model.class_log_prior
            - sklearn_model.class_log_prior_
        )
    )

    likelihood_difference = np.max(
        np.abs(
            custom_model.feature_log_prob
            - sklearn_model.feature_log_prob_
        )
    )

    custom_log_prob = custom_model.predict_log_proba(X_val)
    sklearn_log_prob = sklearn_model.predict_log_proba(X_val)

    posterior_difference = np.max(
        np.abs(custom_log_prob - sklearn_log_prob)
    )

    print("\nCustom Naive Bayes vs sklearn")
    print("--------------------------------")

    print("Custom NB:")
    print(custom_metrics)

    print("sklearn NB:")
    print(sklearn_metrics)

    print(f"Prediction agreement: {agreement:.4f}")
    print(f"Maximum prior difference: {prior_difference:.2e}")
    print(f"Maximum likelihood difference: {likelihood_difference:.2e}")
    print(f"Maximum posterior difference: {posterior_difference:.2e}")


def create_model(method, value):
    """Create one of the three models using the given setting."""

    if method == "Custom Naive Bayes":
        return MultinomialNaiveBayes(alpha=value)

    if method == "Logistic Regression":
        return logistic_regression(C=value)

    if method == "Linear SVM":
        return linear_svm(C=value)

    raise ValueError(f"Unknown method: {method}")


def tune_on_validation(
    method,
    parameter_name,
    values,
    X_train,
    y_train,
    X_val,
    y_val,
):
    """
    Try several parameter values on the validation set.

    The setting with the highest macro recall is selected.
    """
    results = []

    best_value = None
    best_score = -np.inf

    print(f"\nTuning {method}")
    print("--------------------------------")

    for value in values:
        model = create_model(method, value)

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        metrics = calculate_metrics(y_val, predictions)

        row = {
            "method": method,
            "parameter": parameter_name,
            "value": value,
            **metrics,
        }

        results.append(row)

        print(
            f"{parameter_name}={value}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"macro_recall={metrics['macro_recall']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

        score = metrics[SELECTION_METRIC]

        if score > best_score:
            best_score = score
            best_value = value

    print(
        f"Selected {parameter_name}={best_value} "
        f"using validation {SELECTION_METRIC}"
    )

    return best_value, results


def save_validation_results(results):
    """Save all validation experiment results."""
    path = RESULTS_DIR / "validation_results.csv"

    fieldnames = [
        "method",
        "parameter",
        "value",
        "accuracy",
        "macro_recall",
        "macro_f1",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {path}")


def save_test_predictions(
    test_texts,
    y_test,
    predictions,
):
    """Save final predictions so they can be analysed later."""
    path = RESULTS_DIR / "test_predictions.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "text",
                "true_label",
                "custom_naive_bayes",
                "logistic_regression",
                "linear_svm",
            ]
        )

        for i in range(len(y_test)):
            writer.writerow(
                [
                    test_texts[i],
                    y_test[i],
                    predictions["Custom Naive Bayes"][i],
                    predictions["Logistic Regression"][i],
                    predictions["Linear SVM"][i],
                ]
            )

    print(f"Saved: {path}")


def save_summary(best_settings, test_results):
    """Save selected settings and final test scores."""
    path = RESULTS_DIR / "summary.json"

    summary = {
        "selection_metric": SELECTION_METRIC,
        "best_settings": best_settings,
        "test_results": test_results,
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Saved: {path}")


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

    test_texts = features["test_texts"]

    print("Features loaded successfully.")
    print(f"X_train: {X_train.shape}, labels: {len(y_train)}")
    print(f"X_val: {X_val.shape}, labels: {len(y_val)}")
    print(f"X_test: {X_test.shape}, labels: {len(y_test)}")
    print(f"Random seed: {RANDOM_SEED}")

    # Check that our Naive Bayes behaves like sklearn
    check_custom_naive_bayes(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    all_validation_results = []

    # Method 1: Custom Multinomial Naive Bayes
    best_nb_alpha, nb_results = tune_on_validation(
        method="Custom Naive Bayes",
        parameter_name="alpha",
        values=[0.1, 0.5, 1.0, 2.0],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    all_validation_results.extend(nb_results)

    # Method 2: Logistic Regression
    best_lr_c, lr_results = tune_on_validation(
        method="Logistic Regression",
        parameter_name="C",
        values=[0.01, 0.1, 1.0, 10.0],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    all_validation_results.extend(lr_results)

    # Method 3: Linear SVM
    best_svm_c, svm_results = tune_on_validation(
        method="Linear SVM",
        parameter_name="C",
        values=[0.001, 0.01, 0.1, 1.0],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    all_validation_results.extend(svm_results)

    save_validation_results(all_validation_results)

    best_settings = {
        "Custom Naive Bayes": {
            "parameter": "alpha",
            "value": best_nb_alpha,
        },
        "Logistic Regression": {
            "parameter": "C",
            "value": best_lr_c,
        },
        "Linear SVM": {
            "parameter": "C",
            "value": best_svm_c,
        },
    }

    print("\nFinal test evaluation")
    print("--------------------------------")

    final_models = {
        "Custom Naive Bayes": create_model(
            "Custom Naive Bayes",
            best_nb_alpha,
        ),
        "Logistic Regression": create_model(
            "Logistic Regression",
            best_lr_c,
        ),
        "Linear SVM": create_model(
            "Linear SVM",
            best_svm_c,
        ),
    }

    test_results = {}
    test_predictions = {}

    for method, model in final_models.items():
        # Keep training on the training split.
        # Validation was used only to choose the parameter.
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        test_predictions[method] = predictions

        metrics = calculate_metrics(y_test, predictions)
        test_results[method] = metrics

        print(f"\n{method}")
        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1:     {metrics['macro_f1']:.4f}")

    save_test_predictions(
        test_texts,
        y_test,
        test_predictions,
    )

    save_summary(
        best_settings,
        test_results,
    )

    print("\nExperiments completed successfully.")


if __name__ == "__main__":
    main()