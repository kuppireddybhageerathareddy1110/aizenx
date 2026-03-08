import numpy as np


class LocalExplainer:

    def __init__(self, model):
        self.model = model

    def explain(self, instance, samples=100):

        base_pred = self.model.predict_proba([instance])[0][1]

        influence = []

        for i in range(len(instance)):

            diffs = []

            for _ in range(samples):

                perturbed = instance.copy()
                perturbed[i] = perturbed[i] + np.random.normal(0, 0.1)

                pred = self.model.predict_proba([perturbed])[0][1]

                diffs.append(abs(pred - base_pred))

            influence.append(np.mean(diffs))

        return influence