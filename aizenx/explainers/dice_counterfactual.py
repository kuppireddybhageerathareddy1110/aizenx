import numpy as np


class DiceCounterfactual:

    def __init__(self, model, step=0.05, max_iter=1000):

        self.model = model
        self.step = step
        self.max_iter = max_iter


    def generate(self, instance):

        instance = np.array(instance)

        original_pred = self.model.predict([instance])[0]

        x = instance.copy()

        for _ in range(self.max_iter):

            noise = np.random.normal(0, self.step, size=len(instance))

            candidate = x + noise

            pred = self.model.predict([candidate])[0]

            if pred != original_pred:
                return candidate

        return None