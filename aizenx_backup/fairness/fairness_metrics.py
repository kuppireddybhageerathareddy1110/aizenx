import numpy as np


def demographic_parity(y_pred, sensitive):

    groups = np.unique(sensitive)

    rates = []

    for g in groups:

        mask = sensitive == g

        rates.append(np.mean(y_pred[mask]))

    return max(rates) - min(rates)


def equal_opportunity(y_true, y_pred, sensitive):

    groups = np.unique(sensitive)

    tpr = []

    for g in groups:

        mask = (sensitive == g) & (y_true == 1)

        if np.sum(mask) == 0:
            continue

        tpr.append(np.mean(y_pred[mask]))

    return max(tpr) - min(tpr)


def equalized_odds(y_true, y_pred, sensitive):

    groups = np.unique(sensitive)

    scores = []

    for g in groups:

        mask = sensitive == g

        scores.append(np.mean(y_pred[mask] == y_true[mask]))

    return max(scores) - min(scores)