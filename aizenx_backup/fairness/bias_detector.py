import numpy as np


class BiasDetector:

    def __init__(self, model):
        self.model = model

    def detect(self, X, sensitive_feature, bins=5):

        feature = X[:, sensitive_feature]

        bin_edges = np.percentile(feature, np.linspace(0,100,bins+1))

        groups = np.digitize(feature, bin_edges)

        group_means = []

        unique_groups = np.unique(groups)

        for g in unique_groups:

            group_data = X[groups == g]

            if len(group_data) == 0:
                continue

            preds = self.model.predict(group_data)

            group_means.append(np.mean(preds))

        disparity = max(group_means) - min(group_means)

        return {
            "groups": unique_groups.tolist(),
            "group_prediction_mean": group_means,
            "disparity": disparity
        }