#!/usr/bin/env python3
"""The existing methods on the other families (fam_opp.py): per question of a shard, the large model's vision pass once,
then each method's kept set at each ratio and the answer through probe_q3 / probe_g4.pruned_answer, the path of
fam_run.py's FastV and SGL answers. Same questions, sharding, prompts, images and answer lengths as fam_run.py
(text-rich benchmarks) and fam_run2.py (general benchmarks); only the large model is loaded.
    python families/fam_run3.py --family qwen3vl --large 8B --task chartqa --shard 0/4 [--limit N] [--methods ...]
-> $PRUDENT_OUT/families/opponents/<family>-<large>/<task>.s<i>of<k>.jsonl: per question n_visual and per
   <method>@<ratio> the answer, lp:<...> (mean log-probability of its tokens) and ms:<...> (selection time)"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

import fam_opp as O
import fam_run as FR
import fam_run2 as FR2  # also sets FR.MAX_NEW for the general tasks

OUT = f"{FR.ROOT}/families/opponents"
RATIOS = {"qwen3vl": "0.05,0.1", "gemma4": "0.2,0.3"}  # the budgets of the cross-family tables


def docs(task):
    """-> (sorted doc ids, doc_id -> PIL image, {doc_id: prompt}) as fam_run.py / fam_run2.py read them."""
    if task in FR2.DATA:
        pr = FR2.prompts(task)
        return sorted(pr), FR2.images(task), pr
    from datasets import load_dataset
    path, name, split = FR.DATA[task]
    ds, pr = load_dataset(path, name, split=split), FR.prompts(task)
    return sorted(pr), (lambda i: ds[i]["image"].convert("RGB")), pr


def setup(family, large):
    """-> (probe module, processor, large model, encode, decode): fam_run.setup without the small model."""
    if family == "qwen3vl":
        import probe_q3 as F
        F.LARGE = sorted(glob.glob(f"{FR.HUB}/models--Qwen--Qwen3-VL-{large}-Instruct/snapshots/*"))[0]
        proc = F.processor()
        big = F.load(F.LARGE)
        enc = (lambda img, p: F.encode(proc, img, p))  # noqa: E731
        dec = (lambda ids: proc.tokenizer.decode(ids, skip_special_tokens=True).strip())  # noqa: E731
    else:
        import probe_g4 as F
        assert large == "31B"
        proc = F.processor()
        big = F.load(F.LARGE, "cuda:0")
        enc = (lambda img, p: F.encode(proc, img, p, True))  # noqa: E731
        dec = (lambda ids: F.clean(proc.tokenizer.decode(ids, skip_special_tokens=True).strip()))  # noqa: E731
    return F, proc, big, enc, dec


def run_doc(F, proc, big, enc, dec, prep, img, prompt, task, eos, methods, ratios, seed):
    P, feats = prep(big, enc(img, prompt).to(big.device))
    n = int(P["vis"].sum())
    res = dict(n_visual=n)
    for meth in methods:
        for r in ratios:
            k = max(1, int(n * r))  # as fam_run.py's cells
            torch.cuda.synchronize()
            t0 = time.time()
            if meth == "rand":
                keep, P2 = np.sort(np.random.default_rng(seed).choice(n, k, replace=False)), P
            else:
                keep, P2 = O.choose(meth, big, proc.tokenizer, P, feats, prompt, k)
            torch.cuda.synchronize()
            ms = 1000 * (time.time() - t0)
            toks, lps = F.pruned_answer(big, P2, keep, FR.MAX_NEW[task], eos)
            c = f"{meth}@{r:g}"
            res[c], res[f"lp:{c}"], res[f"ms:{c}"] = dec(toks), float(np.mean(lps)) if lps else 0.0, round(ms, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen3vl", "gemma4"], required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--task", choices=list(FR.DATA) + list(FR2.DATA), required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--methods", default="", help="default: fam_opp.RULES[family]")
    ap.add_argument("--ratios", default="", help="default: RATIOS[family]")
    ap.add_argument("--base", default="", help="i/k: only fam_run2's shard i of k (its GQA / VQAv2 0/4, POPE 0/3)")
    a = ap.parse_args()
    methods = a.methods.split(",") if a.methods else list(O.RULES[a.family])
    ratios = [float(r) for r in (a.ratios or RATIOS[a.family]).split(",")]
    si, sk = map(int, a.shard.split("/"))
    order, img, pr = docs(a.task)
    if a.base:  # the same docs as the FastV / SGL cells, then --shard within them
        bi, bk = map(int, a.base.split("/"))
        order = [i for p, i in enumerate(order) if p % bk == bi]
    ids = [i for p, i in enumerate(order) if p % sk == si][: a.limit or None]
    out = f"{a.out_dir}/{a.family}-{a.large}/{a.task}.s{si}of{sk}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(line)["doc_id"] for line in open(out) if line.strip()} if os.path.exists(out) else set()
    F, proc, big, enc, dec = setup(a.family, a.large)
    prep = O.q3_prep if a.family == "qwen3vl" else O.g4_prep
    e = big.generation_config.eos_token_id
    eos = set(e if isinstance(e, list) else [e])
    t0, k = time.time(), 0
    with open(out, "a") as f:
        for i in (i for i in ids if i not in done):
            res = run_doc(F, proc, big, enc, dec, prep, img(i), pr[i], a.task, eos, methods, ratios, seed=i)
            f.write(json.dumps(dict(task=a.task, doc_id=i, **res)) + "\n")
            f.flush()
            k += 1
            if k % 50 == 0:
                print(f"{a.task} {k}/{len(ids) - len(done)} docs, {(time.time() - t0) / k:.2f} s/doc", flush=True)
    print(f"[fam_run3] {out}: {len(ids)} docs in the shard, {k} new, {(time.time() - t0) / max(k, 1):.2f} s/doc", flush=True)


if __name__ == "__main__":
    main()
