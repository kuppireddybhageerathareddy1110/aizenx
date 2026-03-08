import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importance(importances, feature_names):

    importances = np.array(importances)

    sorted_idx = np.argsort(importances)

    plt.figure(figsize=(10,6))

    plt.barh(range(len(sorted_idx)), importances[sorted_idx])

    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])

    plt.title("Feature Importance")

    plt.xlabel("Importance")

    plt.tight_layout()

    plt.show()