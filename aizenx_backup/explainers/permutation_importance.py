import numpy as np
from sklearn.metrics import accuracy_score


def permutation_importance(model, X, y):

    baseline = accuracy_score(y, model.predict(X))

    importances = []

    for i in range(X.shape[1]):

        X_permuted = X.copy()

        np.random.shuffle(X_permuted[:, i])

        score = accuracy_score(y, model.predict(X_permuted))

        importances.append(baseline - score)

    return np.array(importances)