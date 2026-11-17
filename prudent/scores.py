"""Per-question scores from lmms-eval's sample logs, and each benchmark's score as reported in the paper."""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from .answers import norm
from .config import LETTER_TASKS, STOCK_PHRASE, STOCK_PHRASE_RUNS

# Per-sample metric field written by each lmms-eval task.
METRIC = {"gqa": "exact_match", "textvqa_val": "exact_match", "chartqa": "relaxed_overall", "docvqa_val": "anls",
          "infovqa_val": "anls", "ocrbench": "ocrbench_accuracy", "vqav2_val_sub": "exact_match",
          "ok_vqa_val2014": "exact_match", "ai2d": "exact_match", "scienceqa_img": "exact_match"}
_TOK = re.compile(r"\w+|[^\w\s]")


def websrc_f1(gold, pred):
    """lmms-eval's token F1 for WebSRC: lower-cased token sets (words and single punctuation marks)."""
    a, b = set(_TOK.findall(str(gold).lower().strip())), set(_TOK.findall(str(pred).lower().strip()))
    c = len(a & b)
    if not a or not b or not c:
        return 0.0
    p, r = c / len(b), c / len(a)
    return 2 * p * r / (p + r)


def _mmbench():
    from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator
    return MMBench_Evaluator(sys_prompt="There are several options:")


def read_samples(run_dir, task):
    """One lmms-eval run -> DataFrame indexed by doc_id with the answer (ans), its score and the scoring unit q
    (MMBench groups the option-rotated copies of a question, MME the two questions of an image)."""
    f = glob.glob(f"{run_dir}/**/[0-9]*_samples_*.jsonl", recursive=True)
    if len(f) != 1:
        return None
    mmb = _mmbench() if task.startswith("mmbench") else None
    if task.startswith("refcoco"):
        from lmms_eval.tasks.refcoco.utils_rec import compute_iou
    out = {}
    for line in open(f[0]):
        r = json.loads(line)
        ans, q = norm(r["filtered_resps"][0]), r["doc_id"]
        if task in METRIC:
            v = r[METRIC[task]]
            s = float(v["score"] if isinstance(v, dict) else v)
        elif mmb is not None:
            it = r["gpt_eval_score"]
            pred = mmb.can_infer(str(it["prediction"]), mmb.build_choices(it))
            s, q = float(bool(pred) and pred == it["answer"]), int(it["index"]) % 1_000_000
            if task == "mmbench_en_dev":
                ans = str(pred)
        elif task == "mme":
            v = r.get("mme_perception_score") or r["mme_cognition_score"]
            s, q = float(v["score"]), v["question_id"]
        elif task == "pope":
            s = float(r["pope_accuracy"]["score"])
        elif task.startswith("websrc"):
            v = r["websrc_squad_f1"]
            s = websrc_f1(v["answer"], v["parsed_pred"])
        elif task.startswith("seedbench"):
            v = r["seed_image"]
            s = float(str(v["pred"]).strip()[:1].upper() == str(v["answer"]).strip().upper())
        elif task.startswith("refcoco"):
            v = r["refcoco_IoU"]
            s = float(compute_iou(v["pred"], v["bbox"]) >= 0.5)
        else:
            raise ValueError(f"no reader for task {task}")
        out[r["doc_id"]] = (ans, s, q)
    return pd.DataFrame.from_dict(out, orient="index", columns=["ans", "score", "q"]).sort_index()


def benchmark_score(task, score, q):
    """The score of a benchmark: MMBench counts a question right when all its rotated copies are right; MME is its
    official total (perception + cognition, the sum over categories of the mean image score, at most 2800); the other
    benchmarks average their per-question scores (x100)."""
    s = pd.Series(np.asarray(score, float), index=q.index)
    if task == "mme":
        g = s.groupby(q.values)
        img = 100 * (g.mean() + (g.min() == 1))
        return float(img.groupby(img.index.str.split("/").str[0]).mean().sum())
    return 100 * (s.groupby(q).min().mean() if task.startswith("mmbench") else s.mean())


def scored(full, model, task):
    """The questions that the scores of a model and benchmark count: None for all of them, or, on InternVL3-38B's
    GQA and VQAv2, those whose full-token answer is not the stock phrase (prudent/config.py)."""
    if (model, task) not in STOCK_PHRASE_RUNS:
        return None
    return full.index[full.ans != STOCK_PHRASE]


def on(df, keep):
    """A run restricted to the scored questions (keep from scored())."""
    return df if df is None or keep is None else df.loc[df.index.intersection(keep)]


def read_logprobs(path):
    """Per-question log-probability statistics of the large VLM's answer, written by the pruning model with
    FUSE_LOGPROB=1 (columns lp_first_p, lp_mean, ...); None if the file is missing."""
    if not os.path.exists(path):
        return None
    rows = {int(r["doc_id"]): r["lp"] for r in map(json.loads, open(path)) if "lp" in r}
    return pd.DataFrame.from_dict(rows, orient="index").add_prefix("lp_") if rows else None


def confidence(df, task):
    """The confidence of a first answer: the probability of the first token for option letters, the length-normalized
    log-probability otherwise (Section 3.1)."""
    return df["lp_first_p" if task in LETTER_TASKS else "lp_mean"].astype(float).values


def checker_file(root, task):
    """The checker's answers of a benchmark under a run directory (<task>.jsonl) or a data directory (<task>.parquet)."""
    for ext in ("jsonl", "parquet"):
        if os.path.exists(f"{root}/checker/{task}.{ext}"):
            return f"{root}/checker/{task}.{ext}"
    return f"{root}/checker/{task}.jsonl"


def read_checker(path):
    """The checker's answers: doc_id -> normalized answer (ans) and the log-probability of its least probable token
    (min)."""
    if path.endswith(".parquet"):
        d = pd.read_parquet(path).set_index("doc_id")
    else:
        d = pd.DataFrame([json.loads(line) for line in open(path)]).set_index("doc_id")
    d["ans"] = d.ans.map(norm)
    return d


_PERQ = {}


def _perq(root, model, task):
    """The per-question results of one model and benchmark in a data directory (data/perq/<model>/<task>.parquet,
    one row per run and question), or None when root holds raw runs."""
    key = (root, model, task)
    if key not in _PERQ:
        f = f"{root}/{model}/{task}.parquet"
        _PERQ[key] = pd.read_parquet(f) if os.path.exists(f) else None
    return _PERQ[key]


def load_run(root, model, task, spec, frac):
    """One run of the large VLM under --runs (see scripts/make_table.py), joined with its log-probabilities. root may
    also be the shipped data directory data/perq, which holds the same columns for every run of the paper."""
    perq = _perq(root, model, task)
    if perq is not None:
        d = perq[perq.run == f"{spec}-r{frac:g}"]
        if d.empty:
            return None
        d = d.drop(columns="run").set_index("doc_id").sort_index()
        d.index = d.index.astype(int)
        q = d.q if not d.q.str.fullmatch(r"\d+").all() else d.q.astype(int)  # numeric units as the logs give them
        return d.assign(score=d.score.astype(float), q=q)
    base = f"{root}/{model}/{task}/{spec}-r{frac:g}"
    d = read_samples(base, task)
    if d is None:
        return None
    d.index = d.index.astype(int)
    lp = read_logprobs(base + ".lp.jsonl")
    return d.join(lp) if lp is not None else d
