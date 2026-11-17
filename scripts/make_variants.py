#!/usr/bin/env python3
"""Two checks of the appendices.
Appendix I: InternVL3-38B's GQA and VQAv2 scored on every question, the stock-phrase questions kept: the lead of
PRUDENT over the best existing method at each budget.
Appendix C: re-answering only the disagreeing questions and leaving the rest of the quota unspent: the share of the
budget used (per question, as Algorithm 1 counts it) and the change of the mean ratio to the full-token score, over the
108 settings of Tables 1 and 2.
Usage: python scripts/make_variants.py --runs data/perq"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prudent import BENCHMARKS, PLAN, prudent, ranking  # noqa: E402
from prudent.answers import agree  # noqa: E402
from prudent.config import LEARNED  # noqa: E402
from prudent.scores import benchmark_score, checker_file, load_run, on, read_checker, scored  # noqa: E402

METHODS = {"IF-Prune": LEARNED, "SGL": "sgl", "FastV": "fastv", "VisionZip": "visionzip", "ToMe": "tome",
           "CDPruner": "cdpruner", "MMTok": "mmtok", "ZOO-Prune": "zoo-prune"}


def stock_phrase_kept(runs):
    m = "InternVL3-38B"
    for t in ("gqa", "vqav2_val_sub"):
        ck = read_checker(checker_file(runs, t))
        for K, (b, b2) in PLAN.items():
            x = prudent(*(load_run(runs, m, t, ranking(t), f) for f in (b, b2, K)), ck, t, K)
            best = max((benchmark_score(t, y.score, y.q), n) for n, r in METHODS.items()
                       if (y := load_run(runs, m, t, r, K)) is not None)
            print(f"{t} {K:.3f}: PRUDENT {benchmark_score(t, x.score, x.q):.2f}, best existing {best[1]} {best[0]:.2f}")


def reanswer_disagreeing_only(runs):
    rows = []
    for m in ("InternVL3-38B", "InternVL3-8B"):
        for t in BENCHMARKS:
            full = load_run(runs, m, t, LEARNED, 1.0)
            keep, ck = scored(full, m, t), read_checker(checker_file(runs, t))
            fs = on(full, keep)
            f = benchmark_score(t, fs.score, fs.q)
            for K, (b, b2) in PLAN.items():
                first, re, one = (load_run(runs, m, t, ranking(t), fr) for fr in (b, b2, K))
                x = prudent(first, re, one, ck, t, K)
                dis = ~agree(t, ck.ans.reindex(first.index).fillna(""), first.ans)
                stop = x.reanswered.values & dis
                y = x.assign(score=np.where(x.reanswered.values & ~stop, first.score.values, x.score.values))
                used = float(np.where(x.abstained.values, K, b + np.where(stop, b2, 0)).mean() / K)
                xs, ys = on(x, keep), on(y, keep)
                rows.append(dict(K=K, used=used, change=100 * (benchmark_score(t, ys.score, ys.q)
                                                             - benchmark_score(t, xs.score, xs.q)) / f))
    r = pd.DataFrame(rows).groupby("K").mean()
    print(r.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/perq")
    a = ap.parse_args()
    stock_phrase_kept(a.runs)
    reanswer_disagreeing_only(a.runs)


if __name__ == "__main__":
    main()
