import matplotlib.pyplot as plt

def plot_bias(groups):

    plt.figure(figsize=(8,5))

    plt.bar(range(len(groups)), groups)

    plt.title("Group Prediction Means")

    plt.xlabel("Group")

    plt.ylabel("Prediction Mean")

    plt.show()