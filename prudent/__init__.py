"""PRUDENT: re-answer the questions on which a small VLM that reads the full image disagrees with the pruned answer."""
from .config import BENCHMARKS, PLAN, ranking  # noqa: F401
from .reallocate import prudent  # noqa: F401
