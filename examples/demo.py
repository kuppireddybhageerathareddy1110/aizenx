from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from aizenx import Explainer

# visualizations
from aizenx.visualization import plot_feature_importance
from aizenx.visualization.local_plot import plot_local_explanation
from aizenx.visualization.counterfactual_plot import plot_counterfactual
from aizenx.visualization.bias_plot import plot_bias
from aizenx.visualization.correlation_heatmap import plot_correlation
from aizenx.visualization.waterfall_plot import plot_waterfall
from aizenx.visualization.partial_dependence import plot_partial_dependence
from aizenx.visualization.decision_boundary import plot_decision_boundary

# analysis modules
from aizenx.analysis.model_summary import summarize_model
from aizenx.analysis.explanation_stability import explanation_stability
from aizenx.analysis.feature_interactions import feature_interactions

from aizenx.data.dataset_profiler import profile_dataset
from aizenx.reports.report_generator import generate_report

from aizenx.explainers.shap_explainer import ShapExplainer
from aizenx.visualization.shap_waterfall import plot_shap_waterfall
from aizenx.explainers.kernel_shap import KernelShapExplainer

from aizenx.visualization.shap_summary import plot_shap_summary

from aizenx.explainers.shap_interactions import ShapInteractionExplainer
from aizenx.explainers.dice_counterfactual import DiceCounterfactual
from aizenx.causal.causal_explainer import CausalExplainer
from aizenx.explainers.graph_explainer import GraphExplainer
from aizenx.visualization.graph_plot import plot_graph


# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target
feature_names = data.feature_names


# Train model
model = RandomForestClassifier()

model.fit(X, y)


# Initialize explainer
explainer = Explainer(model)


print("\n==========================")
print("MODEL SUMMARY")
print("==========================")

summary = summarize_model(model, X, y)
print(summary)


print("\n==========================")
print("DATASET PROFILE")
print("==========================")

profile = profile_dataset(X, feature_names)
print(profile)


print("\n==========================")
print("GLOBAL FEATURE IMPORTANCE")
print("==========================")

importance = explainer.global_importance(X, y)

print(importance)

plot_feature_importance(importance, feature_names)


print("\n==========================")
print("LOCAL EXPLANATION")
print("==========================")

instance = X[0]

local = explainer.explain_instance(instance)

print(local)

plot_local_explanation(local, feature_names)

# SHAP style visualization
plot_waterfall(local, feature_names)


print("\n==========================")
print("COUNTERFACTUAL")
print("==========================")

cf = explainer.generate_counterfactual(instance)

print(cf)

if cf is not None:
    plot_counterfactual(instance, cf, feature_names)


print("\n==========================")
print("BIAS ANALYSIS")
print("==========================")

bias = explainer.detect_bias(X, sensitive_feature=0)

print(bias)

plot_bias(bias["group_prediction_mean"])


print("\n==========================")
print("FEATURE CORRELATION HEATMAP")
print("==========================")

plot_correlation(X, feature_names)


print("\n==========================")
print("PARTIAL DEPENDENCE PLOT")
print("==========================")

plot_partial_dependence(model, X, 0, feature_names[0])


# print("\n==========================")
# print("EXPLANATION STABILITY")
# print("==========================")

# stability = explanation_stability(explainer, instance)

# print("Stability score:", stability)


print("\n==========================")
print("FEATURE INTERACTIONS")
print("==========================")

interactions = feature_interactions(X, feature_names)

print("Top feature interactions:")
print(interactions)

print("\n==========================")
print("DECISION BOUNDARY (first 2 features)")
print("==========================")

# Train a small model using only 2 features for visualization
from sklearn.ensemble import RandomForestClassifier

viz_model = RandomForestClassifier()

viz_model.fit(X[:, :2], y)

plot_decision_boundary(viz_model, X[:, :2], y)


print("\n==========================")
print("EXPLANATION REPORT")
print("==========================")

report = generate_report(summary, dict(zip(feature_names, importance)), bias)

print(report)

print("\n==========================")
print("SHAP EXPLANATION")
print("==========================")

background = X[:100]

shap_explainer = ShapExplainer(model, background)

base_value, shap_values = shap_explainer.explain(instance)

print("Base value:", base_value)

print("SHAP values:", shap_values)

plot_shap_waterfall(base_value, shap_values, feature_names)

print("\n==========================")
print("KERNEL SHAP EXPLANATION")
print("==========================")

background = X[:100]

kernel_shap = KernelShapExplainer(model, background)

base, shap_values = kernel_shap.explain(instance)

print("Base value:", base)

print("SHAP values:", shap_values)

plot_shap_summary(shap_values, feature_names)

print("\n==========================")
print("SHAP INTERACTIONS")
print("==========================")

interaction_exp = ShapInteractionExplainer(model, X[:100])

interactions = interaction_exp.explain(instance)

print(interactions)


print("\n==========================")
print("DICE COUNTERFACTUAL")
print("==========================")

dice = DiceCounterfactual(model)

cf = dice.generate(instance)

print(cf)

print("\n==========================")
print("CAUSAL EXPLANATION")
print("==========================")

causal = CausalExplainer(model)

effect = causal.intervention(instance, 0, instance[0] * 1.2)

print(effect)

print("\n==========================")
print("GRAPH EXPLANATION")
print("==========================")

graph_exp = GraphExplainer(interactions, feature_names)

G = graph_exp.build_graph()

plot_graph(G)