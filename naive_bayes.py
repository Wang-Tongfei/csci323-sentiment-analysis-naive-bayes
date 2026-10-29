import numpy as np


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes written from scratch using NumPy.
    """

    def __init__(self, alpha=1.0):
        # alpha controls smoothing
        self.alpha = alpha

    def fit(self, X, y):
        """
        Learn class probabilities and word probabilities
        from the training data.
        """
        y = np.asarray(y)

        # Find the sentiment classes
        self.classes = np.unique(y)

        # Number of words/features in the vocabulary
        n_features = X.shape[1]

        # Store the learned probabilities
        self.class_log_prior = np.zeros(len(self.classes))
        self.feature_log_prob = np.zeros(
            (len(self.classes), n_features)
        )

        for i, current_class in enumerate(self.classes):

            # Get only the training examples from this class
            X_class = X[y == current_class]

            # Calculate how common this class is
            class_probability = X_class.shape[0] / X.shape[0]
            self.class_log_prior[i] = np.log(class_probability)

            # Count how often each word appears in this class
            word_counts = np.asarray(
                X_class.sum(axis=0)
            ).ravel()

            # Add smoothing so no word has probability zero
            smoothed_counts = word_counts + self.alpha

            # Convert the word counts into probabilities
            word_probabilities = (
                smoothed_counts / smoothed_counts.sum()
            )

            # Store them as log probabilities
            self.feature_log_prob[i] = np.log(
                word_probabilities
            )

        return self

    def predict_log_scores(self, X):
        """
        Calculate a score for each sentiment class.
        """

        # Combine tweet word counts with the learned word probabilities
        word_scores = X @ self.feature_log_prob.T

        # Also include how common each class was in training
        scores = word_scores + self.class_log_prior

        return scores

    def predict_log_proba(self, X):
        """
        Convert the scores into log probabilities.
        This is mainly used when checking our model against sklearn.
        """

        scores = self.predict_log_scores(X)

        # Keep the calculation numerically stable
        max_scores = scores.max(axis=1, keepdims=True)

        log_total = max_scores + np.log(
            np.exp(scores - max_scores).sum(
                axis=1,
                keepdims=True
            )
        )

        return scores - log_total

    def predict(self, X):
        """
        Predict the sentiment class with the highest score.
        """

        scores = self.predict_log_scores(X)

        # Pick the class with the highest score
        best_class_index = np.argmax(scores, axis=1)

        return self.classes[best_class_index]