import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_correlation(X, feature_names):

    df = pd.DataFrame(X, columns=feature_names)

    plt.figure(figsize=(12,8))

    sns.heatmap(df.corr(), cmap="coolwarm")

    plt.title("Feature Correlation")

    plt.show()