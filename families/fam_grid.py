#!/usr/bin/env python3
"""The budget grid and the checker on the kept tokens on a second family (Appendices G and H, Figure 4d). Per question
of a shard (the questions, prompts, images and answer lengths of fam_run.py / fam_run2.py): Qwen3-VL-2B's full-view
answer, confidence and SGL map; Qwen3-VL-8B's answers on the top-k visual tokens of that map at every ratio of the grid
(the one-pass curve, the first passes b = K/2 and the re-answers 2K for K = 1-20%, b = 1% at K = 2.5%); and the 2B's
own answers on the same kept sets at every first pass b (the checker that reads the kept tokens). Both models share
the visual token grid. Each model's vision pass runs once per question. Greedy, eager attention, as fam_run.
    python families/fam_grid.py --task chartqa --shard 0/2 [--base 0/4] [--limit N] [--out-dir DIR]
-> $PRUDENT_OUT/families/grid/qwen3vl-8B/<task>[.b<j>of<m>].s<i>of<k>.jsonl: per question n_visual, small, slp_*, and for
   each ratio r sgp_s@<r> / lp:sgp_s@<r> (8B) and, for the first passes, pv2B@<r> / lp:pv2B@<r> (2B on the 8B's kept set)"""
import argparse
import json
import os
import time

import numpy as np
import torch

import fam_run as FR
import fam_run3 as FR3  # imports fam_run2 (general tasks' readers, their max_new_tokens)

OUT = f"{FR.ROOT}/families/grid"
GRID = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.075, 0.08, 0.1, 0.12, 0.15, 0.16, 0.2, 0.3,
        0.4]  # the union of b, K and 2K over the budgets of the grid up to 20%
FIRST = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1]  # the first stages b of that plan


@torch.no_grad()
def prep(m, x):
    """probe_q3.large_prep without FastV's map: input embeddings with the image features in place, 3D positions, the
    visual-position mask, DeepStack features (all pruned_answer reads)."""
    mm, ids = m.model, x["input_ids"]
    emb = mm.get_input_embeddings()(ids).clone()
    vo = mm.get_image_features(x["pixel_values"], x["image_grid_thw"], return_dict=True)
    vis = ids == m.config.image_token_id
    emb[vis] = torch.cat(vo.pooler_output, 0).to(emb.dtype)
    pos, _ = mm.get_rope_index(ids, image_grid_thw=x["image_grid_thw"], attention_mask=x["attention_mask"],
                               mm_token_type_ids=x["mm_token_type_ids"])
    return dict(emb=emb, pos=pos, vis=vis[0], ds=[torch.cat(f, 0) for f in vo.deepstack_features])


def eos_of(m):
    e = m.generation_config.eos_token_id
    return set(e if isinstance(e, list) else [e])


def run_doc(F, small, big, enc, dec, img, prompt, task):
    x = enc(img, prompt)
    s_ids, sgp, slp = FR.small_pass_lp(F, small, x, FR.MAX_NEW[task])
    Pb, mx = prep(big, x), FR.MAX_NEW[task]
    n = int(Pb["vis"].sum())
    assert n == len(sgp), (n, len(sgp))
    res = dict(n_visual=n, small=dec(s_ids), **slp)
    order = np.argsort(-np.asarray(sgp, np.float64), kind="stable")  # as fam_run.run_doc's sgp_s cells
    keep = {r: np.sort(order[:max(1, int(n * r))]) for r in GRID}
    for r in GRID:
        toks, lps = F.pruned_answer(big, Pb, keep[r], mx, eos_of(big))
        res[f"sgp_s@{r:g}"], res[f"lp:sgp_s@{r:g}"] = dec(toks), float(np.mean(lps)) if lps else 0.0
    del Pb
    Ps = prep(small, x)
    for r in FIRST:
        toks, lps = F.pruned_answer(small, Ps, keep[r], mx, eos_of(small))
        res[f"pv2B@{r:g}"], res[f"lp:pv2B@{r:g}"] = dec(toks), float(np.mean(lps)) if lps else 0.0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--base", default="", help="i/k: only fam_run2's shard i of k (its GQA / VQAv2 0/4, POPE 0/3)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args()
    si, sk = map(int, a.shard.split("/"))
    order, img, pr = FR3.docs(a.task)
    tag = ""
    if a.base:  # the same docs as the cross-family cells of the general tasks, then --shard within them
        bi, bk = map(int, a.base.split("/"))
        order, tag = [i for p, i in enumerate(order) if p % bk == bi], f".b{bi}of{bk}"
    ids = [i for p, i in enumerate(order) if p % sk == si][: a.limit or None]
    out = f"{a.out_dir}/qwen3vl-8B/{a.task}{tag}.s{si}of{sk}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(line)["doc_id"] for line in open(out) if line.strip()} if os.path.exists(out) else set()
    F, proc, small, big, enc, _, dec = FR.setup("qwen3vl", "8B")
    t0, k = time.time(), 0
    with open(out, "a") as f, torch.no_grad():
        for i in (i for i in ids if i not in done):
            res = run_doc(F, small, big, enc, dec, img(i), pr[i], a.task)
            f.write(json.dumps(dict(task=a.task, doc_id=i, **res)) + "\n")
            f.flush()
            k += 1
            if k % 50 == 0:
                print(f"{a.task} {k}/{len(ids) - len(done)} docs, {(time.time() - t0) / k:.2f} s/doc", flush=True)
    print(f"[fam_grid] {out}: {len(ids)} docs in the shard, {k} new, {(time.time() - t0) / max(k, 1):.2f} s/doc",
          flush=True)


if __name__ == "__main__":
    main()
