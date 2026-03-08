import pandas as pd

def profile_dataset(X, feature_names):

    df = pd.DataFrame(X, columns=feature_names)

    profile = {
        "samples": df.shape[0],
        "features": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "feature_means": df.mean().to_dict(),
        "feature_std": df.std().to_dict()
    }

    return profile