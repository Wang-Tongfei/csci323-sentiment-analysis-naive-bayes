import numpy as np


class MultinomialNaiveBayes:
    """
    Our Multinomial Naive Bayes model built using NumPy.

    The model learns:
    - how common each sentiment class is
    - how common each word is inside each class
    """

    def __init__(self, alpha=1.0):
        # alpha controls smoothing
        self.alpha = alpha


    def fit(self, X, y):
        # Learn the probabilities needed for Naive Bayes.


        y = np.asarray(y)

        # Find the different classes: 0, 1 and 2
        self.classes = np.unique(y)

        # Number of words/features in our vocabulary
        n_features = X.shape[1]

        # Store the class probabilities
        self.class_log_prior = np.zeros(
            len(self.classes)
        )

        # Store word probabilities for each class
        self.feature_log_prob = np.zeros(
            (len(self.classes), n_features)
        )

        # Learn values for each sentiment class
        for i, c in enumerate(self.classes):

            # Only keep training examples from this class
            X_c = X[y == c]

            # Probability of this class
            self.class_log_prior[i] = np.log(
                X_c.shape[0] / X.shape[0]
            )

            # Count each word for this class
            word_counts = np.asarray(
                X_c.sum(axis=0)
            ).ravel()

            # Add smoothing so unseen words do not get probability 0
            smoothed_counts = (
                word_counts + self.alpha
            )

            # Convert word counts into probabilities
            self.feature_log_prob[i] = np.log(
                smoothed_counts
                / smoothed_counts.sum()
            )

        return self


    def predict_log_scores(self, X):
        # Calculate a score for every possible class.
        

        return (
            X @ self.feature_log_prob.T
            + self.class_log_prior
        )


    def predict_log_proba(self, X):
        # Convert the class scores into log probabilities

        scores = self.predict_log_scores(X)

        # Keep the calculation numerically stable
        max_scores = scores.max(
            axis=1,
            keepdims=True
        )

        log_evidence = (
            max_scores
            + np.log(
                np.exp(
                    scores - max_scores
                ).sum(
                    axis=1,
                    keepdims=True
                )
            )
        )

        return scores - log_evidence


    def predict(self, X):
        """
        Predict the sentiment class with the highest score.
        """

        scores = self.predict_log_scores(X)

        return self.classes[
            np.argmax(scores, axis=1)
        ]