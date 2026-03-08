import numpy as np
import matplotlib.pyplot as plt


def plot_shap_summary(shap_values, feature_names):

    shap_values = np.array(shap_values)

    order = np.argsort(abs(shap_values))[::-1]

    shap_values = shap_values[order]
    feature_names = np.array(feature_names)[order]

    plt.figure(figsize=(10,6))

    colors = ["green" if v > 0 else "red" for v in shap_values]

    plt.barh(feature_names, shap_values, color=colors)

    plt.title("SHAP Feature Contributions")

    plt.xlabel("Contribution")

    plt.gca().invert_yaxis()

    plt.show()