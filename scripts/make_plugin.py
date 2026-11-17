#!/usr/bin/env python3
"""Appendix L: PRUDENT on the ranking of IF-Prune or SGL (both passes), against the method alone, at 5%, 10% and 20%,
with the paired bootstrap of Appendix K. Figure 5b averages the columns of 10% and 20% on its seven benchmarks.
The generator is seeded 0 and runs over the 16 benchmarks without RefCOCO in the order below, as for the paper.
Usage: python scripts/make_plugin.py --runs data/perq [--out plugin.csv]"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prudent import BENCHMARKS, PLAN, prudent  # noqa: E402
from prudent.config import GROUNDING, LEARNED  # noqa: E402
from prudent.scores import benchmark_score, checker_file, load_run, on, read_checker, scored  # noqa: E402
from prudent.stats import paired  # noqa: E402

FIG5B = ["chartqa", "docvqa_val", "infovqa_val", "textvqa_val", "websrc_val_sub10", "gqa", "mmbench_en_dev"]
METHODS = {"IF-Prune": LEARNED, "SGL": "sgl"}
SPLITS = {**PLAN, 0.2: (0.1, 0.4)}   # 20% with the default split K/2 and 2K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/perq")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng, rows = np.random.default_rng(0), []
    for model in ["InternVL3-38B", "InternVL3-8B"]:
        for task in [t for t in BENCHMARKS if t not in GROUNDING]:
            full = load_run(a.runs, model, task, LEARNED, 1.0)
            if full is None or not os.path.exists(checker_file(a.runs, task)):
                continue
            keep, checker = scored(full, model, task), read_checker(checker_file(a.runs, task))
            for name, rk in METHODS.items():
                for K, (b, b2) in SPLITS.items():
                    first, re, one = (load_run(a.runs, model, task, rk, f) for f in (b, b2, K))
                    if first is None or re is None or one is None:
                        continue
                    x = prudent(first, re, one, checker, task, K)
                    d, lo, hi = paired(task, on(x, keep), on(one, keep), rng)
                    s = lambda y: benchmark_score(task, on(y, keep).score, on(y, keep).q)  # noqa: E731
                    rows.append(dict(model=model, task=task, method=name, K=K, alone=s(one), first=s(x),
                                     sig=int(lo > 0) - int(hi < 0)))
    t = pd.DataFrame(rows)
    if a.out:
        t.to_csv(a.out, index=False)
    t = t[t.task.isin(FIG5B) & t.K.isin([0.05, 0.1, 0.2])]
    t["cell"] = t["alone"].map("{:.1f}".format) + " / " + t["first"].map("{:.1f}".format) + t.sig.map({1: "*", 0: "", -1: "-"})
    pd.set_option("display.width", 250)
    print(t.pivot_table(index=["model", "task"], columns=["method", "K"], values="cell", aggfunc="first").to_string())
    print("significant +/- per method and budget:")
    print(t.groupby(["method", "K"]).sig.agg(lambda v: f"{(v > 0).sum()}+ / {(v < 0).sum()}- of {len(v)}").to_string())


if __name__ == "__main__":
    main()
