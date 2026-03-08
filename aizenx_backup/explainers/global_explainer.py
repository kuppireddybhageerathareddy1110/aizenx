import numpy as np
from sklearn.metrics import accuracy_score


class GlobalExplainer:

    def __init__(self, model):
        self.model = model

    def feature_importance(self, X, y):

        baseline = accuracy_score(y, self.model.predict(X))
        importances = []

        for col in range(X.shape[1]):

            X_permuted = X.copy()

            np.random.shuffle(X_permuted[:, col])

            score = accuracy_score(y, self.model.predict(X_permuted))

            importance = baseline - score

            importances.append(importance)

        return np.array(importances)