#!/usr/bin/env python3
"""Ablations of Table 5 from the same runs as make_table.py: each variant changes one part of PRUDENT, and its change
is reported in points of the ratio to the full-token score, averaged over the models and the benchmarks of a group.
Variants that need more runs are skipped when their runs are missing:
  IF-Prune's ranking in both passes / in the re-answers   run_large.sh with single-ifp32 at the first-pass and
                                                          re-answer fractions (run_all.sh with ABLATIONS=1)
  another checker (--checkers guide=<dir>)                run_checker.sh with CHECKER=... and OUT=<dir's parent>; the
                                                          1B checker of the paper is the guide, InternVL2.5-1B
The random orders depend on the seed (--seed). On InternVL3-38B, GQA and VQAv2 are scored without the questions whose
full-token answer is the stock phrase, as in make_table.py.
Usage: python scripts/make_ablation.py --runs runs [--checkers guide=runs_1b/checker] [--out ablation.csv]"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prudent import BENCHMARKS, PLAN, prudent, ranking  # noqa: E402
from prudent.config import GROUNDING, LEARNED, TEXT_RICH  # noqa: E402
from prudent.scores import benchmark_score, checker_file, load_run, on, read_checker, scored  # noqa: E402

VARIANTS = {"No checker, one pass at K": None, "The checker never abstains": dict(tau=None, ell=None),
            **{f"Abstain below probability {p}": dict(tau=p) for p in (0.3, 0.4, 0.6, 0.7, 0.8)},
            **{f"Abstain above {n} words": dict(ell=n) for n in (5, 20)},
            "Random choice": dict(order="random"), "Disagreement, in random order": dict(order="disagreement-random"),
            "Large VLM's confidence": dict(order="confidence"),
            "IF-Prune's ranking in both passes": dict(ranking=LEARNED),
            "IF-Prune's ranking in the re-answers": dict(re_ranking=LEARNED)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--models", nargs="+", default=["InternVL3-8B", "InternVL3-38B"])
    ap.add_argument("--tasks", nargs="+", default=[t for t in BENCHMARKS if t not in GROUNDING])
    ap.add_argument("--checkers", nargs="*", default=[], help="NAME=DIR with DIR/<task>.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng, rows = np.random.default_rng(a.seed), []
    others = dict(c.split("=", 1) for c in a.checkers)
    for model in a.models:
        for task in a.tasks:
            full = load_run(a.runs, model, task, LEARNED, 1.0)
            ck = checker_file(a.runs, task)
            if full is None or not os.path.exists(ck):
                continue
            keep = scored(full, model, task)

            def score(x):
                x = on(x, keep)
                return benchmark_score(task, x.score, x.q)

            checker, fs = read_checker(ck), score(full)
            for K, (b, b2) in PLAN.items():
                rk = ranking(task)
                first, re, one = (load_run(a.runs, model, task, rk, f) for f in (b, b2, K))
                if first is None or re is None or one is None:
                    continue
                base = score(prudent(first, re, one, checker, task, K))
                row = dict(model=model, task=task, K=K, full=fs, PRUDENT=base)
                for name, kw in VARIANTS.items():
                    kw = dict(kw or {})
                    if name.startswith("No checker"):
                        x = one
                    else:
                        f1, r1, o1 = first, re, one
                        if kw.pop("ranking", None):
                            f1, r1, o1 = (load_run(a.runs, model, task, LEARNED, f) for f in (b, b2, K))
                        if kw.pop("re_ranking", None):
                            r1 = load_run(a.runs, model, task, LEARNED, b2)
                        if f1 is None or r1 is None or o1 is None:
                            continue
                        x = prudent(f1, r1, o1, checker, task, K, rng=rng, **kw)
                    row[name] = score(x)
                for name, d in others.items():
                    if os.path.exists(f"{d}/{task}.jsonl"):
                        x = prudent(first, re, one, read_checker(f"{d}/{task}.jsonl"), task, K)
                        row[f"Checker {name}"] = score(x)
                rows.append(row)
    t = pd.DataFrame(rows)
    if a.out:
        t.to_csv(a.out, index=False)
    t["group"] = np.where(t.task.isin(TEXT_RICH), "text-rich", "general")
    var = [c for c in t.columns if c not in ("model", "task", "K", "full", "PRUDENT", "group")]
    rel = t[["group", "K"]].assign(**{"PRUDENT (ratio to full)": 100 * t.PRUDENT / t.full},
                                   **{v: 100 * (t[v] - t.PRUDENT) / t.full for v in var})
    pd.set_option("display.width", 220)
    print(rel.groupby(["group", "K"]).mean().T.round(1).to_string())


if __name__ == "__main__":
    main()
