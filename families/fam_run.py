#!/usr/bin/env python3
"""Qwen3-VL (the 2B guides the 8B or 32B) and Gemma 4 (the E4B guides the 31B) on the five text-rich benchmarks of
Table 4. Per question: the small model's full-view answer, its confidence and SGL map, the large model's full-token answer
and FastV map, and its answers on the top-k visual tokens of each map at 2.5, 5, 10 and 20%. Images come from the HF
datasets of lmms-eval, prompts from data/prompts (the text every model saw). Sharded by question position, resumable.
    python families/fam_run.py --family qwen3vl --large 32B --task chartqa --shard 0/4
-> $PRUDENT_OUT/families/<family>-<large>/<task>.s<i>of<k>.jsonl"""
import argparse
import glob
import json
import os
import time

import numpy as np

HF = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", f"{HF}/hub")
os.environ.setdefault("HF_DATASETS_CACHE", f"{HF}/datasets")
ROOT = os.environ.get("PRUDENT_OUT", "runs")
OUT = f"{ROOT}/families"
HUB = os.environ["HF_HUB_CACHE"]
DATA = {"chartqa": ("lmms-lab/ChartQA", None, "test"), "docvqa_val": ("lmms-lab/DocVQA", "DocVQA", "validation"),
        "infovqa_val": ("lmms-lab/DocVQA", "InfographicVQA", "validation"),
        "textvqa_val": ("lmms-lab/textvqa", None, "validation"), "ocrbench": ("echo840/OCRBench", None, "test")}
MAX_NEW = {"chartqa": 16, "docvqa_val": 32, "infovqa_val": 32, "textvqa_val": 32, "ocrbench": 128}
RATIOS = [0.025, 0.05, 0.1, 0.2]


def prompts(task):
    """{doc_id: prompt}: the text every model saw (data/prompts/<task>.jsonl, from the InternVL3-8B full-token run)."""
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "prompts", f"{task}.jsonl")
    return {r["doc_id"]: r["prompt"] for r in map(json.loads, open(f))}


def setup(family, large):
    """-> (module, processor, small, large model, encode for the small, encode for the large, decode)."""
    if family == "qwen3vl":
        import probe_q3 as F
        F.LARGE = sorted(glob.glob(f"{HUB}/models--Qwen--Qwen3-VL-{large}-Instruct/snapshots/*"))[0]
        proc = F.processor()
        small, big = F.load(F.SMALL), F.load(F.LARGE)
        enc_s = enc_l = (lambda img, p: F.encode(proc, img, p))  # noqa: E731
        dec = (lambda ids: proc.tokenizer.decode(ids, skip_special_tokens=True).strip())  # noqa: E731
    else:
        import probe_g4 as F
        assert large == "31B"
        proc = F.processor()
        small, big = F.load(F.SMALL, "cuda:0"), F.load(F.LARGE, "cuda:0")
        enc_s = (lambda img, p: F.encode(proc, img, p, False))  # noqa: E731
        enc_l = (lambda img, p: F.encode(proc, img, p, True))  # noqa: E731
        dec = (lambda ids: F.clean(proc.tokenizer.decode(ids, skip_special_tokens=True).strip()))  # noqa: E731
    return F, proc, small, big, enc_s, enc_l, dec


def small_pass_lp(F, small, x, max_new):
    """F.small_pass plus the small model's confidence in its own answer (for the abstention of the checker): generate()'s
    own logits are captured by shadowing the instance's generate for this one call; statistics over the answer
    tokens before the first EOS."""
    import torch
    cap, orig = {}, small.generate
    def gen(**kw):  # noqa: E306
        cap["out"] = orig(**kw, output_logits=True)
        return cap["out"]
    small.generate = gen
    try:
        s_ids, sgp = F.small_pass(small, x, max_new)
    finally:
        del small.generate
    e = small.generation_config.eos_token_id
    eos = set(e if isinstance(e, list) else [e])
    lp = torch.stack([torch.log_softmax(z[0].float(), -1) for z in cap["out"].logits])
    n = next((i for i, t in enumerate(s_ids) if t in eos), len(s_ids))
    tok = lp[torch.arange(n, device=lp.device), torch.as_tensor(s_ids[:n], device=lp.device)] if n else lp[:0, 0]
    st = dict(slp_n=n, slp_min=float(tok.min()) if n else 0.0, slp_mean=float(tok.mean()) if n else 0.0,
              slp_first_p=float(lp[0, s_ids[0]].exp()) if s_ids else 0.0, slp_trunc=n == len(s_ids))
    return s_ids, sgp, st


def run_doc(F, small, big, enc_s, enc_l, dec, img, prompt, task, eos, ratios=RATIOS):
    s_ids, sgp, slp = small_pass_lp(F, small, enc_s(img, prompt), MAX_NEW[task])
    P = F.large_prep(big, enc_l(img, prompt).to(big.device))
    n = int(P["vis"].sum())
    assert n == len(sgp), (n, len(sgp))
    full, flp = F.pruned_answer(big, P, None, MAX_NEW[task], eos)
    res = dict(n_visual=n, small=dec(s_ids), full=dec(full), lp_full=float(np.mean(flp)) if flp else 0.0, **slp)
    for name, mp in (("fastv", P["fastv"]), ("sgp_s", sgp)):
        order = np.argsort(-np.asarray(mp, np.float64), kind="stable")
        for ratio in ratios:
            toks, lps = F.pruned_answer(big, P, np.sort(order[:max(1, int(n * ratio))]), MAX_NEW[task], eos)
            res[f"{name}@{ratio:g}"] = dec(toks)
            res[f"lp:{name}@{ratio:g}"] = float(np.mean(lps)) if lps else 0.0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen3vl", "gemma4"], required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--task", choices=list(DATA), required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=OUT, help="where the outputs go")
    ap.add_argument("--ratios", default=",".join(f"{r:g}" for r in RATIOS),
                    help="Gemma 4 uses 0.1,0.2,0.3,0.5")
    a = ap.parse_args()
    from datasets import load_dataset
    si, sk = map(int, a.shard.split("/"))
    path, name, split = DATA[a.task]
    ds, pr = load_dataset(path, name, split=split), prompts(a.task)
    ids = [i for p, i in enumerate(sorted(pr)) if p % sk == si][: a.limit or None]
    out = f"{a.out_dir}/{a.family}-{a.large}/{a.task}.s{si}of{sk}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(line)["doc_id"] for line in open(out) if line.strip()} if os.path.exists(out) else set()
    F, proc, small, big, enc_s, enc_l, dec = setup(a.family, a.large)
    e = big.generation_config.eos_token_id
    eos = set(e if isinstance(e, list) else [e])
    t0, k = time.time(), 0
    with open(out, "a") as f:
        for i in ids:
            if i in done:
                continue
            res = run_doc(F, small, big, enc_s, enc_l, dec, ds[i]["image"].convert("RGB"), pr[i], a.task, eos,
                          [float(r) for r in a.ratios.split(",")])
            f.write(json.dumps(dict(task=a.task, doc_id=i, **res)) + "\n")
            f.flush()
            k += 1
            if k % 50 == 0:
                print(f"{a.task} {k}/{len(ids) - len(done)} docs, {(time.time() - t0) / k:.2f} s/doc", flush=True)
    print(f"[fam_run] {out}: {len(ids)} docs in the shard, {k} new", flush=True)


if __name__ == "__main__":
    main()
