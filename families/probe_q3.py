#!/usr/bin/env python3
"""Qwen3-VL building blocks used by fam_run*.py: Qwen3-VL-2B answers with the full view (eager attention -> its SGL map,
as internvl_sgp.py defines it); Qwen3-VL-8B or 32B answers with the full view (its layer-2 last-token attention = FastV's
map) and with only the top-k visual tokens of a map. Pruning is at the LLM input: kept tokens keep their 3D positions and
their DeepStack features. Run alone it answers a few questions per benchmark as a check.
    python families/probe_q3.py [--check] [--tasks chartqa,...] [--limit N]"""
import argparse
import base64
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image

HUB = os.environ.get("HF_HUB_CACHE", os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"))
SMALL, LARGE = (sorted(glob.glob(f"{HUB}/models--Qwen--Qwen3-VL-{s}-Instruct/snapshots/*"))[0] for s in ("2B", "8B"))
PROBE = os.path.join(os.environ.get("PRUDENT_OUT", "runs"), "families", "probe_qwen3vl")
MAX_TOKENS = 1792  # visual-token cap, as InternVL's 6 tiles + thumbnail (7 x 256)
MAX_NEW = {"chartqa": 16, "gqa": 16, "docvqa_val": 32, "textvqa_val": 32}
RATIOS = [0.025, 0.05, 0.1, 0.2]


def internvl_grid(w, h, min_num=1, max_num=6, image_size=448):
    """(cols, rows) of InternVL's dynamic_preprocess (internvl_kl.py, copied: same ratio order and tie rule)."""
    aspect = w / h
    ratios = sorted(set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                        if min_num <= i * j <= max_num), key=lambda x: x[0] * x[1])
    best_diff, best = float("inf"), (1, 1)
    for r in ratios:
        diff = abs(aspect - r[0] / r[1])
        if diff < best_diff:
            best_diff, best = diff, r
        elif diff == best_diff and w * h > 0.5 * image_size * image_size * r[0] * r[1]:
            best = r
    return best


def transfer(imp, num_patches, w, h, gh, gw):
    """An importance map over InternVL's tile tokens (tiles row-major, 16 x 16 tokens each, then the thumbnail)
    -> one score per Qwen merged token (gh x gw, row-major): the tile token under the Qwen token's centre."""
    cols, rows = internvl_grid(w, h)
    assert cols * rows + (cols * rows > 1) == num_patches, (w, h, cols, rows, num_patches)
    y = (np.arange(gh) + 0.5) / gh * rows
    x = (np.arange(gw) + 0.5) / gw * cols
    r, i = np.floor(y).astype(int), np.floor((y % 1) * 16).astype(int)
    c, j = np.floor(x).astype(int), np.floor((x % 1) * 16).astype(int)
    idx = ((r[:, None] * cols + c[None, :]) * 256 + i[:, None] * 16 + j[None, :]).reshape(-1)
    return np.asarray(imp, np.float32)[idx]


def rankavg(*maps):
    """Mean rank (higher = more important) over the maps; ties in the result keep the first map's order."""
    ranks = [np.argsort(np.argsort(m, kind="stable"), kind="stable") for m in maps]
    return np.mean(ranks, axis=0) + 1e-6 * ranks[0] / len(maps[0])


def b64(s):
    return np.frombuffer(base64.b64decode(s), np.float32)


def load(path):
    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0",
                                                       attn_implementation="eager").eval()


def processor():
    """One processor for both models (identical configs), visual tokens capped at MAX_TOKENS."""
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained(SMALL)
    p.image_processor.size = {"shortest_edge": p.image_processor.size["shortest_edge"], "longest_edge": MAX_TOKENS * 32 * 32}
    return p


def encode(proc, img, prompt):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    return proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                    return_tensors="pt").to("cuda:0")


@torch.no_grad()
def small_pass(m, x, max_new):
    """The 2B's greedy answer ids and its SGL map: attention to the visual tokens summed over layers and heads
    from every prompt token after the image (prefill) and every generated token (internvl_sgp.sgp_importance)."""
    out = m.generate(**x, max_new_tokens=max_new, do_sample=False, output_attentions=True, return_dict_in_generate=True)
    ids = x["input_ids"][0]
    vis = (ids == m.config.image_token_id).nonzero().squeeze(1)
    imp = torch.zeros(len(vis), device=ids.device)
    for step, layers in enumerate(out.attentions):
        for a in layers:
            rows = a[0, :, int(vis[-1]) + 1:] if step == 0 else a[0, :, -1:]
            imp += rows[:, :, vis].float().sum(dim=(0, 1))
    return out.sequences[0, len(ids):].tolist(), imp.cpu().numpy()


@torch.no_grad()
def large_prep(m, x):
    """Everything the pruned passes share: input embeddings with the image features in place, 3D positions
    (t, h, w), the visual-position mask, DeepStack features, and FastV's map (layer-2 last-token attention)."""
    mm, ids = m.model, x["input_ids"]
    emb = mm.get_input_embeddings()(ids).clone()
    vo = mm.get_image_features(x["pixel_values"], x["image_grid_thw"], return_dict=True)
    vis = ids == m.config.image_token_id
    emb[vis] = torch.cat(vo.pooler_output, 0).to(emb.dtype)
    pos, _ = mm.get_rope_index(ids, image_grid_thw=x["image_grid_thw"], attention_mask=x["attention_mask"],
                               mm_token_type_ids=x["mm_token_type_ids"])
    att = m(**x, output_attentions=True).attentions[2]
    fastv = att[0, :, -1, vis[0]].float().mean(0).cpu().numpy()
    return dict(emb=emb, pos=pos, vis=vis[0], ds=[torch.cat(f, 0) for f in vo.deepstack_features], fastv=fastv)


@torch.no_grad()
def pruned_answer(m, P, keep, max_new, eos):
    """Greedy answer of the large model seeing all text tokens and only the visual tokens `keep` (indices into
    the visual tokens; None = all). -> (token ids, their log-probs)."""
    from transformers import DynamicCache
    vis, sel, vk = P["vis"], torch.ones_like(P["vis"]), None
    if keep is not None:
        vk = torch.zeros(int(vis.sum()), dtype=torch.bool, device=vis.device)
        vk[torch.as_tensor(keep, device=vis.device)] = True
        sel = sel.clone()
        sel[vis] = vk
    lm, cache = m.model.language_model, DynamicCache(config=m.model.language_model.config)
    ds = P["ds"] if vk is None else [d[vk] for d in P["ds"]]
    h = lm(inputs_embeds=P["emb"][:, sel], position_ids=P["pos"][:, :, sel], past_key_values=cache, use_cache=True,
           visual_pos_masks=vis[sel][None], deepstack_visual_embeds=ds).last_hidden_state[:, -1:]
    nxt, toks, lps = int(P["pos"].max()) + 1, [], []
    for step in range(max_new):
        lp = torch.log_softmax(m.lm_head(h)[0, -1].float(), -1)
        t = int(lp.argmax())
        if t in eos:
            break
        toks.append(t)
        lps.append(float(lp[t]))
        p = torch.full((3, 1, 1), nxt + step, dtype=P["pos"].dtype, device=vis.device)
        h = lm(inputs_embeds=lm.embed_tokens(torch.tensor([[t]], device=vis.device)), position_ids=p,
               past_key_values=cache, use_cache=True).last_hidden_state
    return toks, lps


def run_doc(r, proc, small, large, eos, rng):
    """All answers for one probe doc."""
    task, dec = r["task"], (lambda ids: proc.tokenizer.decode(ids, skip_special_tokens=True).strip())  # noqa: E731
    x = encode(proc, Image.open(r["image"]).convert("RGB"), r["prompt"])
    gh, gw = (int(v) // 2 for v in x["image_grid_thw"][0, 1:])
    s_ids, sgp2b = small_pass(small, x, MAX_NEW[task])
    P = large_prep(large, x)
    n = int(P["vis"].sum())
    assert n == gh * gw == len(sgp2b), (n, gh, gw, len(sgp2b))
    tr = lambda k: transfer(b64(r[k]), r["num_patches"], r["width"], r["height"], gh, gw)  # noqa: E731
    maps = dict(random=rng.random(n), fastv=P["fastv"], sgp2b=sgp2b, ifp_tr=tr("ifp32_b64"), sgp1b_tr=tr("sgp_b64"))
    maps["look_q"] = rankavg(maps["sgp2b"], maps["ifp_tr"])
    maps["look_tr"] = rankavg(maps["sgp1b_tr"], maps["ifp_tr"])
    full, flp = pruned_answer(large, P, None, MAX_NEW[task], eos)
    res = dict(task=task, doc_id=r["doc_id"], n_visual=n, grid=[gh, gw], small=dec(s_ids), full=dec(full),
               lp_full=float(np.mean(flp)) if flp else 0.0)
    for name, mp in maps.items():
        order = np.argsort(-np.asarray(mp, np.float64), kind="stable")
        for ratio in RATIOS:
            toks, lps = pruned_answer(large, P, np.sort(order[:max(1, int(n * ratio))]), MAX_NEW[task], eos)
            res[f"{name}@{ratio:g}"] = dec(toks)
            res[f"lp:{name}@{ratio:g}"] = float(np.mean(lps)) if lps else 0.0
    return res, x, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="chartqa,docvqa_val,textvqa_val,gqa")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="also compare the unpruned path with HF generate()")
    ap.add_argument("--shard", default="0/1", help="i/k: only docs at positions p with p % k == i (own output file)")
    a = ap.parse_args()
    proc, small, large = processor(), load(SMALL), load(LARGE)
    eos = set(large.generation_config.eos_token_id if isinstance(large.generation_config.eos_token_id, list)
              else [large.generation_config.eos_token_id])
    for task in a.tasks.split(","):
        recs = [json.loads(line) for line in open(f"{PROBE}/{task}.jsonl")][: a.limit or None]
        si, sk = map(int, a.shard.split("/"))
        recs = [r for p, r in enumerate(recs) if p % sk == si]
        path = f"{PROBE}/out_{task}{'_check' if a.check else ''}{f'.s{si}of{sk}' if sk > 1 else ''}.jsonl"
        base = f"{PROBE}/out_{task}{'_check' if a.check else ''}.jsonl"  # a shard also skips what the unsharded run did
        done = {json.loads(line)["doc_id"] for q in {path, base} if os.path.exists(q) for line in open(q) if line.strip()}
        rng, t0, k = np.random.default_rng(0), time.time(), 0
        with open(path, "a") as f:
            for r in recs:
                if r["doc_id"] in done:
                    continue
                res, x, P = run_doc(r, proc, small, large, eos, rng)
                if a.check:
                    ref = large.generate(**x, max_new_tokens=MAX_NEW[task], do_sample=False)
                    res["hf_full"] = proc.tokenizer.decode(ref[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                f.write(json.dumps(res) + "\n")
                f.flush()
                k += 1
                if k % 20 == 0 or a.check:
                    print(f"{task} {k} docs, {(time.time() - t0) / k:.2f} s/doc", flush=True)


if __name__ == "__main__":
    main()
