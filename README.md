# AizenX 🚀

![PyPI](https://img.shields.io/pypi/v/aizenx-xai)
![Downloads](https://img.shields.io/pypi/dm/aizenx-xai)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Stars](https://img.shields.io/github/stars/kuppireddybhageerathareddy1110/aizenx)
![Forks](https://img.shields.io/github/forks/kuppireddybhageerathareddy1110/aizenx)
![Issues](https://img.shields.io/github/issues/kuppireddybhageerathareddy1110/aizenx)
![Last Commit](https://img.shields.io/github/last-commit/kuppireddybhageerathareddy1110/aizenx)

AizenX is a simple **Explainable AI (XAI) toolkit** that helps developers understand how machine learning models make decisions.

🌍 **PyPI Package:**  
https://pypi.org/project/aizenx-xai/


AizenX is an **Explainable AI (XAI) toolkit** that helps people understand how machine learning models make decisions.

Machine learning models are often called **"black boxes"** because we can see the input and output, but we don't know **why** the model made a decision.

AizenX opens this black box and explains:

* Why a model made a prediction
* Which features influenced the decision
* How predictions can change
* Whether the model is biased or unfair

This makes AI **more transparent, trustworthy, and easier to debug**.

---

# Why AizenX?

Imagine you build a model that predicts if a patient has cancer.

The model says:

```
Prediction: Cancer
```

But a doctor might ask:

* Why did the model say this?
* Which features were important?
* What would change the prediction?

AizenX helps answer those questions.

---

# Features

AizenX includes tools to explain machine learning models in many ways.

## 1️⃣ Global Feature Importance

Shows which features are **most important for the model overall**.

Example:

```
Feature Importance
-------------------
Age           0.42
Tumor Size    0.30
Texture       0.18
Shape         0.10
```

This tells us **which features the model relies on the most**.

---

## 2️⃣ Local Prediction Explanation

Explains **why the model made a specific prediction** for one data point.

Example:

```
Prediction: Cancer

Feature Contributions
----------------------
Tumor Size        +0.45
Radius            +0.32
Texture           -0.10
```

This shows how each feature influenced the final prediction.

---

## 3️⃣ Counterfactual Explanations

A counterfactual answer tells us:

> "What would need to change for the prediction to change?"

Example:

```
Current prediction: Cancer

Change these features:

Tumor Size: 20 → 12
Texture: 15 → 10

New prediction: No Cancer
```

This helps users understand **how decisions could be different**.

---

## 4️⃣ Bias Detection

AizenX checks if the model is **biased toward certain groups**.

Example:

```
Group A prediction rate: 80%
Group B prediction rate: 45%

Bias detected.
```

This helps ensure **fair AI systems**.

---

## 5️⃣ Visualization Tools

AizenX provides easy-to-understand visualizations such as:

* Feature importance plots
* Explanation waterfall charts
* SHAP summary plots
* Correlation heatmaps
* Decision boundary plots
* Counterfactual comparisons

These visuals make it easier to **see how models behave**.

---
## Installation

Install the latest version from PyPI:

```bash
pip install aizenx-xai
```
---

# Add a PyPI link section

```markdown
## PyPI Package

AizenX is available on Python Package Index:

https://pypi.org/project/aizenx-xai/

You can install it globally using pip.
```

# Quick Example

Here is a simple example showing how to use AizenX.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from aizenx import Explainer

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Train a model
model = RandomForestClassifier()
model.fit(X, y)

# Create explainer
explainer = Explainer(model)

# Global explanation
importance = explainer.global_importance(X, y)

print("Feature Importance:")
print(importance)

# Local explanation
instance = X[0]

explanation = explainer.explain_instance(instance)

print("Local Explanation:")
print(explanation)
```

---

# Visualization Example

You can also visualize explanations.

```python
from aizenx.visualization import plot_feature_importance

plot_feature_importance(importance, feature_names)
```

This will show a chart of the most important features.

---

# Command Line Interface

AizenX includes a CLI tool.

Run:

```bash
aizenx --info
```

Example output:

```
AizenX Explainable AI Toolkit
Version: 0.1.0
```

---

# Project Structure

```
aizenx/
│
├── core/            Main explainer interface
├── explainers/      Explanation algorithms
├── fairness/        Bias and fairness metrics
├── analysis/        Model analysis tools
├── visualization/   Plotting utilities
├── utils/           Helper utilities
├── dashboard/       Interactive dashboard
└── cli.py           Command line interface
```

---

# Who Can Use AizenX?

AizenX can be used by:

* Data Scientists
* Machine Learning Engineers
* Researchers
* Students learning AI
* Developers building ML applications

---

# Why Explainable AI Matters

AI systems are used in important areas like:

* Healthcare
* Finance
* Education
* Hiring
* Self-driving cars

Understanding AI decisions helps ensure:

* Transparency
* Fairness
* Trust
* Safety

AizenX helps make AI systems **easier to understand and improve**.

---

# License

MIT License

---

# Author

Bhageeratha Reddy
<img width="1749" height="959" alt="image" src="https://github.com/user-attachments/assets/ed71d6ab-1828-4b8f-9bf9-19d93db73655" />

---

# GitHub

[https://github.com/kuppireddybhageerathareddy1110/aizenx](https://github.com/kuppireddybhageerathareddy1110/aizenx)

