import numpy as np
import pandas as pd


def feature_interactions(X, feature_names, threshold=0.7):

    df = pd.DataFrame(X, columns=feature_names)

    corr = df.corr()

    interactions = []

    for i in range(len(feature_names)):

        for j in range(i + 1, len(feature_names)):

            value = corr.iloc[i, j]

            if abs(value) > threshold:

                interactions.append({
                    "feature_1": feature_names[i],
                    "feature_2": feature_names[j],
                    "correlation": value
                })

    interactions = sorted(
        interactions,
        key=lambda x: abs(x["correlation"]),
        reverse=True
    )

    return interactions