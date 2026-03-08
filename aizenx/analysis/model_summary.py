from sklearn.metrics import accuracy_score

def summarize_model(model, X, y):

    preds = model.predict(X)

    accuracy = accuracy_score(y, preds)

    summary = {
        "model_type": type(model).__name__,
        "samples": X.shape[0],
        "features": X.shape[1],
        "accuracy": accuracy
    }

    return summary