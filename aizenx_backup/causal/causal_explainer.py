import numpy as np


class CausalExplainer:

    def __init__(self, model):

        self.model = model


    def intervention(self, instance, feature_index, value):

        instance = np.array(instance)

        original = self.model.predict([instance])[0]

        intervened = instance.copy()

        intervened[feature_index] = value

        new_pred = self.model.predict([intervened])[0]

        return {
            "original_prediction": original,
            "new_prediction": new_pred,
            "causal_effect": new_pred - original
        }