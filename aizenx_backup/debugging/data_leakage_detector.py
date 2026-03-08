import numpy as np
import pandas as pd


def detect_data_leakage(X, y, feature_names, threshold=0.95):

    df = pd.DataFrame(X, columns=feature_names)

    leakage = []

    for col in feature_names:

        corr = abs(np.corrcoef(df[col], y)[0,1])

        if corr > threshold:

            leakage.append({
                "feature": col,
                "correlation": corr
            })

    return leakage