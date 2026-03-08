import matplotlib.pyplot as plt
import numpy as np

def plot_counterfactual(original, counterfactual, feature_names):

    diff = counterfactual - original

    sorted_idx = np.argsort(abs(diff))

    plt.figure(figsize=(10,6))

    plt.barh(range(len(sorted_idx)), diff[sorted_idx])

    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])

    plt.title("Counterfactual Feature Changes")

    plt.xlabel("Change")

    plt.show()