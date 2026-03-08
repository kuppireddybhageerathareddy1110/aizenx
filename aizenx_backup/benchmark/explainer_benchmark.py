import time


def benchmark_explainers(explainers, instance):

    results = []

    for name, explainer in explainers.items():

        start = time.time()

        explainer.explain(instance)

        end = time.time()

        results.append({
            "explainer": name,
            "time": end-start
        })

    return results