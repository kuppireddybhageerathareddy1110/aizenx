import numpy as np

class ShapLikeExplainer:

    def __init__(self, model):
        self.model = model

    def explain(self, instance, X_background):

        base_value = np.mean(self.model.predict(X_background))

        shap_values = []

        for i in range(len(instance)):

            modified = instance.copy()

            modified[i] = np.mean(X_background[:, i])

            pred = self.model.predict([modified])[0]

            contribution = base_value - pred

            shap_values.append(contribution)

        return shap_values