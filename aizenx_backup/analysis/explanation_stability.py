import numpy as np


def explanation_stability(explainer, instance, noise_level=0.1, trials=30):

    # base explanation
    base_exp = np.array(explainer.explain_instance(instance))

    variations = []

    for _ in range(trials):

        noisy_instance = instance + np.random.normal(0, noise_level, size=len(instance))

        noisy_exp = np.array(explainer.explain_instance(noisy_instance))

        diff = np.linalg.norm(base_exp - noisy_exp)

        variations.append(diff)

    stability = np.mean(variations)

    return stability