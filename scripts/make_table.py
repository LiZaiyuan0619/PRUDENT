#!/usr/bin/env python3
"""Main tables from the run logs or from the shipped per-question data (--runs data/perq): for each model, benchmark
and budget, the full-token score, the existing methods (IF-Prune, and the others whose runs exist), one-pass pruning
with the guide's ranking, and PRUDENT; the last rows give the mean ratio to the full-token score (Avg. of Tables 1
and 2). --significance adds, for each method, PRUDENT's difference with a star when its paired bootstrap 95% interval
excludes zero (prudent/stats.py). On InternVL3-38B, GQA and VQAv2 are scored without the questions whose full-token
answer is the stock phrase (prudent/scores.py scored()).
Layout read under --runs (written by run_large.sh and run_checker.py):
  <model>/<task>/<ranking>-r<fraction>/   lmms-eval output of the large VLM
  <model>/<task>/<ranking>-r<fraction>.lp.jsonl   its per-question log-probabilities
  checker/<task>.jsonl                     the checker's answers
Usage: python scripts/make_table.py --runs data/perq [--models InternVL3-8B InternVL3-38B] [--significance] [--out table.csv]"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prudent import BENCHMARKS, PLAN, prudent, ranking  # noqa: E402
from prudent.config import LEARNED, RANK_AVG  # noqa: E402
from prudent.stats import paired  # noqa: E402

PUBLISHED = {"IF-Prune": LEARNED, "SGL": "sgl", "FastV": "fastv", "VisionZip": "visionzip", "ToMe": "tome",
             "CDPruner": "cdpruner", "MMTok": "mmtok", "ZOO-Prune": "zoo-prune"}  # run names of run_baseline.sh
from prudent.scores import benchmark_score, checker_file, load_run, on, read_checker, scored  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--models", nargs="+", default=["InternVL3-8B", "InternVL3-38B"])
    ap.add_argument("--tasks", nargs="+", default=BENCHMARKS)
    ap.add_argument("--significance", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows, rng = [], np.random.default_rng(0)
    for model in a.models:
        for task in a.tasks:
            full = load_run(a.runs, model, task, LEARNED, 1.0)
            keep = scored(full, model, task) if full is not None else None
            ck = checker_file(a.runs, task)
            checker = read_checker(ck) if os.path.exists(ck) else None
            for K, (b, b2) in PLAN.items():
                rk = ranking(task)
                first, re, one = (load_run(a.runs, model, task, rk, f) for f in (b, b2, K))
                runs = {name: load_run(a.runs, model, task, spec, K) for name, spec in PUBLISHED.items()}
                runs["guide one pass"] = one
                row = dict(model=model, task=task, K=K)
                for name, df in [("full", full)] + list(runs.items()):
                    df = on(df, keep)
                    row[name] = benchmark_score(task, df.score, df.q) if df is not None else float("nan")
                if all(v is not None for v in (first, re, one, checker)):
                    x = prudent(first, re, one, checker, task, K)  # selects among all questions
                    row["re-answered"] = float(x.reanswered.mean())
                    x = on(x, keep)
                    row["PRUDENT"] = benchmark_score(task, x.score, x.q)
                    if a.significance:  # one generator, in this order (prudent/stats.py)
                        comps = {n: runs[n] for n in PUBLISHED}
                        comps["rank-average one pass"] = load_run(a.runs, model, task, RANK_AVG, K)
                        for name, df in comps.items():
                            if df is not None:
                                d, lo, hi = paired(task, x, df.reindex(x.index), rng)
                                row[f"vs {name}"] = f"{d:+.1f}{'*' if lo > 0 or hi < 0 else ''}"
                rows.append(row)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.assign(K=t.K.map(lambda k: f"{100 * k:g}%")).round(2).to_string(index=False))
    cols = [c for c in list(PUBLISHED) + ["guide one pass", "PRUDENT"] if t[c].notna().any()]
    avg = t.assign(**{c: 100 * t[c] / t["full"] for c in cols}).groupby(["model", "K"])[cols].mean()
    print("\nAvg. (mean ratio to the full-token score):\n" + avg.round(1).to_string())
    if a.out:
        t.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
