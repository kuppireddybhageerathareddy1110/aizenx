def generate_report(summary, importance, bias):

    report = {}

    report["model_summary"] = summary

    report["top_features"] = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    report["bias_score"] = bias["disparity"]

    return report