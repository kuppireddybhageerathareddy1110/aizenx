import matplotlib.pyplot as plt
import numpy as np

def plot_local_explanation(values, feature_names):

    values = np.array(values)

    sorted_idx = np.argsort(values)

    plt.figure(figsize=(10,6))

    plt.barh(range(len(sorted_idx)), values[sorted_idx])

    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])

    plt.title("Local Feature Influence")

    plt.xlabel("Influence")

    plt.show()