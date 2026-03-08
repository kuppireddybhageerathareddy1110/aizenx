from ..explainers.global_explainer import GlobalExplainer
from ..explainers.local_explainer import LocalExplainer
from ..explainers.counterfactual import CounterfactualGenerator
from ..fairness.bias_detector import BiasDetector


class Explainer:

    def __init__(self, model):

        self.model = model
        self.global_exp = GlobalExplainer(model)
        self.local_exp = LocalExplainer(model)
        self.counterfactual = CounterfactualGenerator(model)
        self.bias_detector = BiasDetector(model)

    def global_importance(self, X, y):

        return self.global_exp.feature_importance(X, y)

    def explain_instance(self, instance):

        return self.local_exp.explain(instance)

    def generate_counterfactual(self, instance):

        return self.counterfactual.generate(instance)

    def detect_bias(self, X, sensitive_feature):

        return self.bias_detector.detect(X, sensitive_feature)