"""Paired comparison of two runs on the same questions (Section 4.1): the difference of their benchmark scores with a
bootstrap 95% interval over questions. MMBench resamples questions (all rotated copies together); MME resamples images
within each category and compares the official totals. scripts/make_table.py --significance draws all intervals from
one generator seeded 0, in the order models, benchmarks, budgets, methods, as for the paper."""
import numpy as np
import pandas as pd


def mme_image_scores(score, q):
    """MME per image: 100 * (accuracy + both right) over its two questions, indexed by image, with its category."""
    s = pd.Series(np.asarray(score, float), index=q.index)
    g = s.groupby(q.values)
    img = 100 * (g.mean() + (g.min() == 1))
    return img, img.index.str.split("/").str[0]


def paired(task, x, y, rng, B=2000):
    """x, y: per-question frames (score, q) of two runs. -> (difference x - y in benchmark points, 2.5% and 97.5%
    bootstrap percentiles). A difference is significant when the interval excludes zero."""
    if task == "mme":
        y = y.reindex(x.index)
        ix, cat = mme_image_scores(x.score, x.q)
        iy, _ = mme_image_scores(y.score, y.q)
        d = pd.Series((ix - iy.reindex(ix.index)).values, index=cat)
        bs = np.zeros(B)
        for _, dc in d.groupby(level=0):
            v = dc.values
            bs += v[rng.integers(0, len(v), (B, len(v)))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return float(d.groupby(level=0).mean().sum()), lo, hi
    if task.startswith("mmbench"):
        x, y = x.score.groupby(x.q).min(), y.score.groupby(y.q).min()
    else:
        x, y = x.score, y.score.reindex(x.index)
    d = (x - y).values.astype(float)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return 100 * d.mean(), 100 * lo, 100 * hi
