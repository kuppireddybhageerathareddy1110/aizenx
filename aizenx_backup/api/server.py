from fastapi import FastAPI
import numpy as np
from aizenx import Explainer

app = FastAPI()

model = None
explainer = None


@app.post("/load_model")
def load_model():

    global model, explainer

    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier

    data = load_breast_cancer()

    X = data.data
    y = data.target

    model = RandomForestClassifier()

    model.fit(X, y)

    explainer = Explainer(model)

    return {"status": "model loaded"}


@app.post("/explain/local")
def local_explanation(instance: list):

    instance = np.array(instance)

    explanation = explainer.explain_instance(instance)

    return {"explanation": explanation}