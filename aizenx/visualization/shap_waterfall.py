import numpy as np
import matplotlib.pyplot as plt


def plot_shap_waterfall(base_value, shap_values, feature_names):

    shap_values = np.array(shap_values)

    order = np.argsort(abs(shap_values))[::-1]

    shap_values = shap_values[order]
    feature_names = np.array(feature_names)[order]

    values = base_value + np.cumsum(shap_values)

    plt.figure(figsize=(10,6))

    for i, val in enumerate(shap_values):

        color = "green" if val > 0 else "red"

        plt.barh(i, val, color=color)

    plt.yticks(range(len(feature_names)), feature_names)

    plt.xlabel("Contribution")

    plt.title("SHAP Waterfall Explanation")

    plt.gca().invert_yaxis()

    plt.show()