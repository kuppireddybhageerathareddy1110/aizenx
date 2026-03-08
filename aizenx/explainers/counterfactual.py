import numpy as np


class CounterfactualGenerator:

    def __init__(self, model):

        self.model = model

    def generate(self, instance, steps=100):

        original = self.model.predict([instance])[0]

        for i in range(steps):

            new_instance = instance + np.random.normal(0, 0.5, size=len(instance))

            new_pred = self.model.predict([new_instance])[0]

            if new_pred != original:

                return new_instance

        return None