import numpy as np


class IntegratedGradients:

    def __init__(self, model, baseline=None, steps=50):
        self.model = model
        self.steps = steps
        self.baseline = baseline

    def explain(self, instance):

        instance = np.array(instance)

        if self.baseline is None:
            baseline = np.zeros_like(instance)
        else:
            baseline = self.baseline

        scaled_inputs = [
            baseline + (float(i) / self.steps) * (instance - baseline)
            for i in range(self.steps + 1)
        ]

        grads = []

        for x in scaled_inputs:

            pred1 = self.model.predict([x])[0]

            eps = 1e-4
            grad = []

            for i in range(len(x)):
                x_eps = x.copy()
                x_eps[i] += eps
                pred2 = self.model.predict([x_eps])[0]
                grad.append((pred2 - pred1) / eps)

            grads.append(grad)

        grads = np.array(grads)

        avg_grad = np.mean(grads, axis=0)

        integrated_grad = (instance - baseline) * avg_grad

        return integrated_grad