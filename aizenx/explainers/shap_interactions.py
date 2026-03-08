import numpy as np


class ShapInteractionExplainer:

    def __init__(self, model, background):
        self.model = model
        self.background = np.array(background)

    def explain(self, instance):

        instance = np.array(instance)

        n = len(instance)

        interactions = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):

                diffs = []

                for bg in self.background:

                    x = bg.copy()

                    pred_base = self.model.predict_proba([x])[0][1]

                    x_i = x.copy()
                    x_i[i] = instance[i]

                    pred_i = self.model.predict_proba([x_i])[0][1]

                    x_j = x.copy()
                    x_j[j] = instance[j]

                    pred_j = self.model.predict_proba([x_j])[0][1]

                    x_ij = x.copy()
                    x_ij[i] = instance[i]
                    x_ij[j] = instance[j]

                    pred_ij = self.model.predict_proba([x_ij])[0][1]

                    interaction = pred_ij - pred_i - pred_j + pred_base

                    diffs.append(interaction)

                interactions[i, j] = np.mean(diffs)
                interactions[j, i] = interactions[i, j]

        return interactions