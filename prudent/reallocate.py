"""PRUDENT on one benchmark at one average budget K (Section 3.2, Algorithm 1 in Appendix C).

The large VLM answers every question at the fraction b (first pass), at b2 (re-answer, 2K) and at K (one-pass pruning,
used when the checker abstains). With greedy decoding an answer does not depend on whether its question was chosen,
so running the three fractions and choosing afterwards gives the same answers as answering again only the chosen
questions, and the budget accounting is the same (Eq. 1). The keyword arguments give the ablations of Table 5."""
import numpy as np

from .answers import agree
from .config import ELL, TAU, quota  # noqa: F401  (ELL is the default of ell)
from .scores import confidence

ORDERS = ("prudent", "disagreement-random", "confidence", "random")


def prudent(first, re, one, checker, task, K, order="prudent", tau=TAU, ell=ELL, b=None, b2=None, rng=None):
    """first, re, one: per-question frames of the large VLM (ans, score, q and the lp_* columns of first), indexed by
    doc_id; checker: the checker's answers (ans, min), or None for no checker. order: how the checked questions are
    ranked for the quota (PRUDENT: disagreeing first, each group from low to high confidence). tau = None or ell = None
    switch that abstention rule off. b, b2: the fractions of first and re, when they differ from config.PLAN.
    Returns first with the final answers and scores, and the flags reanswered and abstained."""
    n = len(first)
    re, one = re.reindex(first.index), one.reindex(first.index)
    disagrees, abstains = np.zeros(n, bool), np.zeros(n, bool)
    if checker is not None:
        small = checker.ans.reindex(first.index).fillna("")
        disagrees = ~agree(task, small, first.ans)
        if tau is not None:
            least_p = np.exp(checker["min"].astype(float).reindex(first.index).values)
            abstains |= np.nan_to_num(least_p, nan=0) < tau
        abstains |= small.str.split().str.len().values > (10 ** 9 if ell is None else ell)
    conf = np.nan_to_num(confidence(first, task), nan=-1e9)
    rng = rng or np.random.default_rng(0)
    keys = {"prudent": (conf, ~disagrees), "disagreement-random": (rng.random(n), ~disagrees),
            "confidence": (conf,), "random": (rng.random(n),)}
    ranked = np.lexsort(keys[order] + (abstains,))
    chosen = np.zeros(n, bool)
    chosen[ranked[:quota(K, int((~abstains).sum()), b, b2)]] = True
    chosen &= ~abstains
    return first.assign(
        score=np.where(abstains, one.score.values, np.where(chosen, re.score.values, first.score.values)),
        ans=np.where(abstains, one.ans.values, np.where(chosen, re.ans.values, first.ans.values)),
        reanswered=chosen, abstained=abstains)
