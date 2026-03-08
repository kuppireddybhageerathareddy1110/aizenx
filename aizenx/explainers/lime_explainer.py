import numpy as np
from sklearn.linear_model import Ridge


class LimeExplainer:

    def __init__(self, model, samples=500):
        self.model = model
        self.samples = samples

    def explain(self, instance):

        instance = np.array(instance)

        perturbed = []
        preds = []

        for _ in range(self.samples):

            noise = np.random.normal(0, 0.1, size=len(instance))

            sample = instance + noise

            perturbed.append(sample)

            pred = self.model.predict([sample])[0]

            preds.append(pred)

        perturbed = np.array(perturbed)
        preds = np.array(preds)

        model = Ridge(alpha=1)

        model.fit(perturbed, preds)

        return model.coef_