import numpy as np
import matplotlib.pyplot as plt


def plot_partial_dependence(model, X, feature_index, feature_name):

    values = np.linspace(
        X[:,feature_index].min(),
        X[:,feature_index].max(),
        50
    )

    preds = []

    for val in values:

        X_copy = X.copy()

        X_copy[:,feature_index] = val

        pred = model.predict_proba(X_copy)[:,1].mean()

        preds.append(pred)

    plt.figure(figsize=(8,5))

    plt.plot(values, preds)

    plt.xlabel(feature_name)

    plt.ylabel("Predicted Probability")

    plt.title("Partial Dependence Plot")

    plt.show()