#!/usr/bin/env python3
"""PRUDENT on a stream of questions (Appendix C). Algorithm 1 sorts all checked questions of a benchmark and
re-answers the first rho|C|. Here the questions arrive one at a time in a random order. A question is re-answered when,
among the checked questions seen so far, it falls in the first rho*m of PRUDENT's order (disagreeing first, low
confidence first), and only while the re-answers so far stay within floor(rho*m). The average budget then never
exceeds K, with no held-out questions and no labels. Each setting is replayed with several random orders. On
InternVL3-38B, GQA and VQAv2 are scored without the questions whose full-token answer is the stock phrase.
Usage: python scripts/make_stream.py --runs data/perq [--orders 5] [--out stream.csv]"""
import argparse
import bisect
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prudent import BENCHMARKS, PLAN, prudent, ranking  # noqa: E402
from prudent.answers import agree  # noqa: E402
from prudent.config import ELL, LEARNED, TAU  # noqa: E402
from prudent.scores import benchmark_score, checker_file, confidence, load_run, on, read_checker, scored  # noqa: E402


def stream(first, re, one, checker, task, K, rng):
    """One random arrival order. Returns the final scores and the realized average budget."""
    b, b2 = PLAN[K]
    rho = (K - b) / b2
    n = len(first)
    re, one = re.reindex(first.index), one.reindex(first.index)
    small = checker.ans.reindex(first.index).fillna("")
    dis = ~agree(task, small, first.ans)
    least_p = np.nan_to_num(np.exp(checker["min"].astype(float).reindex(first.index).values), nan=0)
    abst = (least_p < TAU) | (small.str.split().str.len().values > ELL)
    conf = np.nan_to_num(confidence(first, task), nan=-1e9)
    # PRUDENT's order as one key: disagreeing questions first, then low confidence first
    key = np.where(dis, 0.0, 1.0) * 1e12 + conf
    chosen = np.zeros(n, bool)
    seen = []  # sorted keys of the checked questions seen so far
    n_re = 0
    for i in rng.permutation(n):
        if abst[i]:
            continue
        rank = bisect.bisect_left(seen, key[i])  # questions seen so far that come before it
        bisect.insort(seen, key[i])
        m = len(seen)
        cap = int(rho * m)
        if rank < rho * m and n_re < cap:
            chosen[i] = True
            n_re += 1
    score = np.where(abst, one.score.values, np.where(chosen, re.score.values, first.score.values))
    cost = np.where(abst, K, b + np.where(chosen, b2, 0.0))
    return score, float(cost.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/perq")
    ap.add_argument("--models", nargs="+", default=["InternVL3-8B", "InternVL3-38B"])
    ap.add_argument("--tasks", nargs="+", default=BENCHMARKS)
    ap.add_argument("--orders", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for model in a.models:
        for task in a.tasks:
            full = load_run(a.runs, model, task, LEARNED, 1.0)
            ck = checker_file(a.runs, task)
            if full is None or not os.path.exists(ck):
                continue
            checker, keep = read_checker(ck), scored(full, model, task)
            for K, (b, b2) in PLAN.items():
                rk = ranking(task)
                first, re, one = (load_run(a.runs, model, task, rk, f) for f in (b, b2, K))
                ifp = load_run(a.runs, model, task, LEARNED, K)
                if any(x is None for x in (first, re, one, ifp)):
                    continue
                x = prudent(first, re, one, checker, task, K)
                f, i, x = on(full, keep), on(ifp, keep), on(x, keep)
                row = dict(model=model, task=task, K=K, full=benchmark_score(task, f.score, f.q),
                           ifprune=benchmark_score(task, i.score, i.q), batch=benchmark_score(task, x.score, x.q))
                sel = np.ones(len(first), bool) if keep is None else first.index.isin(keep)
                scores, budgets = [], []
                for seed in range(a.orders):
                    s, c = stream(first, re, one, checker, task, K, np.random.default_rng(seed))
                    scores.append(benchmark_score(task, s[sel], first.q[sel]))
                    budgets.append(c)
                row.update(stream=float(np.mean(scores)), stream_min=float(np.min(scores)),
                           stream_max=float(np.max(scores)), budget=float(np.mean(budgets)) / K)
                rows.append(row)
    d = pd.DataFrame(rows)
    r = d.assign(**{c: 100 * d[c] / d.full for c in ("ifprune", "batch", "stream")})
    avg = r.groupby(["model", "K"])[["ifprune", "batch", "stream"]].mean().round(2)
    print("Avg. (mean ratio to the full-token score):\n" + avg.to_string())
    print(f"budget kept: {100 * d.budget.min():.1f}% to {100 * d.budget.max():.1f}% of K")
    gap = r.stream - r.batch  # in points of the ratio to the full-token score, as MME is out of 2800
    print(f"largest drop from the batch version in one setting: {-min(0, gap.min()):.2f} points of the ratio")
    lead = d.batch > d.ifprune
    print(f"settings where the batch version is above IF-Prune: {int(lead.sum())}, the stream version there too: "
          f"{int((d.stream[lead] > d.ifprune[lead]).sum())}")
    if a.out:
        d.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
