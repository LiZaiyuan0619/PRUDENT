#!/usr/bin/env python3
"""The pruning failures of every existing method and how the checker sees them (Figure 3b, Appendix J).
A pruning failure at budget K: the full-token run answers the question correctly and the method's run at K wrongly
(right = score of at least 0.5). For each model, benchmark, budget and method: the share of all questions that are
failures, the share of the failures the checker answers correctly from the full image, the share on which it disagrees
with the pruned answer, and the AUROC with which the questions the checker answers are ranked for these failures by
the large VLM's confidence alone and by the order of PRUDENT (disagreement first, then low confidence).
Usage: python scripts/make_failures.py --runs data/perq [--out failures.csv]"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prudent.answers import agree  # noqa: E402
from prudent.config import BENCHMARKS, ELL, GROUNDING, LEARNED, TAU  # noqa: E402
from prudent.scores import checker_file, confidence, load_run, read_checker  # noqa: E402

METHODS = {"ToMe": "tome", "VisionZip": "visionzip", "ZOO-Prune": "zoo-prune", "FastV": "fastv", "CDPruner": "cdpruner",
           "MMTok": "mmtok", "SGL": "sgl", "IF-Prune": LEARNED}  # Table 1's order


def auroc(score, y):
    """Mann-Whitney AUROC of score for the positives y (ties count half)."""
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(score).rank().values
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/perq")
    ap.add_argument("--models", nargs="+", default=["InternVL3-8B", "InternVL3-38B"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for model in a.models:
        for task in [t for t in BENCHMARKS if t not in GROUNDING]:  # the checker abstains on grounding questions
            full, ck = load_run(a.runs, model, task, LEARNED, 1.0), checker_file(a.runs, task)
            if full is None or not os.path.exists(ck):
                continue
            checker = read_checker(ck)
            for name, spec in METHODS.items():
                for K in (0.025, 0.05, 0.1):
                    run = load_run(a.runs, model, task, spec, K)
                    if run is None:
                        continue
                    idx = run.index.intersection(full.index).intersection(checker.index)
                    fail = (full.score.reindex(idx).values >= 0.5) & (run.score.reindex(idx).values < 0.5)
                    c = checker.reindex(idx)
                    right = c.score.values >= 0.5
                    dis = ~np.asarray(agree(task, c.ans.fillna(""), run.ans.reindex(idx).fillna("")), bool)
                    answers = (c.ans.fillna("").str.split().str.len().values <= ELL) & \
                        (np.nan_to_num(np.exp(c["min"].astype(float).values), nan=0) >= TAU)
                    conf = np.nan_to_num(confidence(run.reindex(idx), task), nan=-1e9)
                    rows.append(dict(model=model, task=task, method=name, K=K, n=len(idx), n_fail=int(fail.sum()),
                                     n_right=int((fail & right).sum()), n_dis=int((fail & dis).sum()),
                                     auroc_confidence=auroc(-conf[answers], fail[answers]),
                                     auroc_prudent=auroc(dis[answers] * 1e12 - conf[answers], fail[answers])))
    d = pd.DataFrame(rows)
    # Figure 3b: both models pooled by questions, then the mean over the benchmarks
    g = d.groupby(["method", "K", "task"])[["n", "n_fail", "n_right", "n_dis"]].sum()
    for c in ("n_fail", "n_right", "n_dis"):
        g[c] = 100 * g[c] / g.n
    m = g.groupby(["method", "K"])[["n_fail", "n_right", "n_dis"]].mean()
    t = pd.DataFrame({"failures (% of questions)": m.n_fail, "checker right (% of failures)": 100 * m.n_right / m.n_fail,
                      "checker disagrees (% of failures)": 100 * m.n_dis / m.n_fail}).unstack("K").round(1)
    pd.set_option("display.width", 220)
    print(t.reindex(list(METHODS)).to_string())
    au = d.groupby("method")[["auroc_confidence", "auroc_prudent"]].median().reindex(list(METHODS)).round(2)
    print("\nAUROC on the questions the checker answers, median over the settings:\n" + au.to_string())
    print(f"\nPRUDENT's order above the confidence alone in {int((d.auroc_prudent > d.auroc_confidence).sum())} "
          f"of {len(d)} settings")
    if a.out:
        d.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
