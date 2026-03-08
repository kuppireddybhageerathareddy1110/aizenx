import matplotlib.pyplot as plt
import numpy as np


def plot_waterfall(contributions, feature_names):

    contributions = np.array(contributions)

    order = np.argsort(abs(contributions))[::-1]

    contributions = contributions[order]
    feature_names = np.array(feature_names)[order]

    plt.figure(figsize=(10,6))

    colors = ["green" if v > 0 else "red" for v in contributions]

    plt.barh(range(len(contributions)), contributions, color=colors)

    plt.yticks(range(len(contributions)), feature_names)

    plt.xlabel("Contribution")

    plt.title("Waterfall Explanation")

    plt.gca().invert_yaxis()

    plt.show()