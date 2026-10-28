from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# Fixed value so repeated runs stay consistent
RANDOM_SEED = 323


def sklearn_naive_bayes(alpha):
    """
    sklearn version of Multinomial Naive Bayes.

    We use this only to check our own Naive Bayes model.
    It is not counted as another main method.
    """
    return MultinomialNB(alpha=alpha)


def logistic_regression(C):
    """
    Create the Logistic Regression model.

    C controls regularisation:
    smaller C = stronger regularisation.
    """
    return LogisticRegression(
        C=C,
        max_iter=2000,
        random_state=RANDOM_SEED
    )


def linear_svm(C):
    """
    Create the Linear SVM model.

    C controls the balance between a wider margin
    and correctly classifying the training examples.
    """
    return LinearSVC(
        C=C,
        loss="hinge",
        max_iter=5000,
        random_state=RANDOM_SEED
    )