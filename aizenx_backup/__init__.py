from .core.explainer import Explainer

from .analysis.model_summary import summarize_model
from .data.dataset_profiler import profile_dataset
from .reports.report_generator import generate_report
from .causal.causal_explainer import CausalExplainer

__version__ = "0.1.0"

__all__ = [
    "Explainer",
    "summarize_model",
    "profile_dataset",
    "generate_report",
    "CausalExplainer"
]