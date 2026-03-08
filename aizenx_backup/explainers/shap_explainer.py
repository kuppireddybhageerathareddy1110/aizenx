import numpy as np
import itertools


class ShapExplainer:

    def __init__(self, model, background_data):
        self.model = model
        self.background = background_data

    def base_value(self):
        preds = self.model.predict_proba(self.background)[:, 1]
        return np.mean(preds)

    def explain(self, instance):

        instance = np.array(instance)

        n_features = len(instance)

        shap_values = np.zeros(n_features)

        base = self.base_value()

        for i in range(n_features):

            contribs = []

            for j in range(len(self.background)):

                background_sample = self.background[j].copy()

                with_feature = background_sample.copy()
                with_feature[i] = instance[i]

                pred_without = self.model.predict_proba(
                    [background_sample]
                )[0][1]

                pred_with = self.model.predict_proba(
                    [with_feature]
                )[0][1]

                contribs.append(pred_with - pred_without)

            shap_values[i] = np.mean(contribs)

        return base, shap_values