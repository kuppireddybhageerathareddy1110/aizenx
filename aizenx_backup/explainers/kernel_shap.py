import numpy as np
import math
from sklearn.linear_model import LinearRegression


class KernelShapExplainer:

    def __init__(self, model, background_data, nsamples=100):

        self.model = model
        self.background = np.array(background_data)
        self.nsamples = nsamples


    def base_value(self):

        preds = self.model.predict_proba(self.background)[:, 1]

        return np.mean(preds)


    def _kernel_weight(self, M, s):

        # avoid divide by zero
        if s == 0 or s == M:
            return 1000

        return (M - 1) / (math.comb(M, s) * s * (M - s))


    def _sample_coalitions(self, M):

        coalitions = []

        while len(coalitions) < self.nsamples:

            mask = np.random.randint(0, 2, M)

            s = np.sum(mask)

            # avoid invalid coalitions
            if s != 0 and s != M:
                coalitions.append(mask)

        return np.array(coalitions)


    def explain(self, instance):

        instance = np.array(instance)

        M = len(instance)

        base = self.base_value()

        Z = self._sample_coalitions(M)

        weights = []

        y = []

        for mask in Z:

            s = np.sum(mask)

            weights.append(self._kernel_weight(M, s))

            samples = []

            for background in self.background:

                x = background.copy()

                for i in range(M):

                    if mask[i] == 1:
                        x[i] = instance[i]

                samples.append(x)

            samples = np.array(samples)

            pred = np.mean(self.model.predict_proba(samples)[:, 1])

            y.append(pred)

        weights = np.array(weights)
        y = np.array(y)

        model = LinearRegression()

        model.fit(Z, y, sample_weight=weights)

        shap_values = model.coef_

        return base, shap_values