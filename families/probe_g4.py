#!/usr/bin/env python3
"""Gemma 4 building blocks used by fam_run*.py: Gemma-4-E4B answers with the full view (eager attention -> its SGL map);
Gemma-4-31B answers with the full view (its layer-2 last-token attention = FastV's map) and with only the top-k visual
tokens of a map. Both use max_soft_tokens = 1120 (Gemma 4's largest budget). Pruning at the LLM input keeps the kept
tokens' positions; the prefill mask is built as Gemma4Model.forward builds it (vision tokens attend bidirectionally
within the image) on the pruned sequence, with position_ids=None so the gaps are not taken for packed sequences.
    python families/probe_g4.py [--check] [--tasks chartqa,...] [--limit N] [--shard i/k]"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from probe_q3 import MAX_NEW, PROBE, RATIOS

HUB = os.environ.get("HF_HUB_CACHE", os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"))
SMALL, LARGE = (sorted(glob.glob(f"{HUB}/models--google--gemma-4-{s}-it/snapshots/*"))[0] for s in ("E4B", "31B"))
OUT = os.environ.get("G4_OUT", os.path.join(os.environ.get("PRUDENT_OUT", "runs"), "families", "probe_gemma4"))
if os.environ.get("G4_RATIOS"):  # the budgets used on Gemma 4
    RATIOS = [float(r) for r in os.environ["G4_RATIOS"].split(",")]
G4_MAPS = os.environ.get("G4_MAPS", "fastv,sgp_s").split(",")
SOFT_TOKENS = 1120


def load(path, device):
    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map=device,
                                                       attn_implementation="eager").eval()


def processor():
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained(SMALL)
    p.image_processor.max_soft_tokens = SOFT_TOKENS
    return p


EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


def encode(proc, img, prompt, thought_block):
    """Chat prompt; thought_block appends an empty thought block (Gemma 4's no-thinking form): the 31B otherwise
    writes the block itself, eating the answer's token budget, while the E4B answers directly without it (and starts
    reasoning in the answer with it)."""
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if thought_block and not text.endswith(EMPTY_THOUGHT):
        text += EMPTY_THOUGHT
    return proc(text=text, images=[img], return_tensors="pt").to("cuda:0")


def clean(ans):
    return ans[len("thought\n"):].strip() if ans.startswith("thought\n") else ans


class _Stop(Exception):
    pass


@torch.no_grad()
def layer2_last_attention(m, x):
    """The last token's attention at LLM layer 2, mean over heads, over all positions; the forward stops before
    layer 3 (one layer's weights instead of all 60: the 31B's full set does not fit next to other jobs)."""
    cap, layers = {}, m.model.language_model.layers
    h1 = layers[2].self_attn.register_forward_hook(lambda mod, inp, out: cap.__setitem__("w", out[1]))

    def stop(mod, inp):
        raise _Stop()
    h2 = layers[3].register_forward_pre_hook(stop)
    try:
        m(**x)
    except _Stop:
        pass
    finally:
        h1.remove()
        h2.remove()
    return cap["w"][0, :, -1].float().mean(0)


@torch.no_grad()
def large_prep(m, x):
    """Input embeddings with the image features in place (as Gemma4Model.forward), positions, the visual mask, the
    token types (for the vision mask), and FastV's map (layer-2 last-token attention, eager full forward)."""
    mm, ids = m.model, x["input_ids"]
    vis = ids == m.config.image_token_id
    emb = mm.get_input_embeddings()(torch.where(vis, m.config.get_text_config().pad_token_id, ids)).clone()
    feats = mm.get_image_features(x["pixel_values"], x.get("image_position_ids"), return_dict=True)
    emb[vis] = torch.cat(feats.pooler_output, 0).to(emb.dtype)
    fastv = layer2_last_attention(m, x)[vis[0]].cpu().numpy()
    pos = torch.arange(ids.shape[1], device=ids.device)[None]
    return dict(emb=emb, pos=pos, vis=vis[0], mtt=x["mm_token_type_ids"], fastv=fastv)


@torch.no_grad()
def pruned_answer(m, P, keep, max_new, eos):
    """Greedy answer of the large model seeing all text tokens and only the visual tokens `keep` (None = all)."""
    from transformers import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_vision_model, get_block_sequence_ids_for_mask
    vis, sel = P["vis"], torch.ones_like(P["vis"])
    if keep is not None:
        vk = torch.zeros(int(vis.sum()), dtype=torch.bool, device=vis.device)
        vk[torch.as_tensor(keep, device=vis.device)] = True
        sel = sel.clone()
        sel[vis] = vk
    tc, lm = m.config.get_text_config(), m.model.language_model
    cache = DynamicCache(config=tc)
    emb, mtt = P["emb"][:, sel], P["mtt"][:, sel]
    mask = create_masks_for_vision_model(config=tc, inputs_embeds=emb, attention_mask=None, past_key_values=cache,
                                         position_ids=None, block_sequence_ids=get_block_sequence_ids_for_mask(mtt, emb.device))
    h = lm(inputs_embeds=emb, position_ids=P["pos"][:, sel], attention_mask=mask, past_key_values=cache,
           use_cache=True).last_hidden_state[:, -1:]
    cap, nxt, toks, lps = tc.final_logit_softcapping, int(P["pos"].max()) + 1, [], []
    for step in range(max_new):
        logits = m.lm_head(h)[0, -1].float()
        if cap:
            logits = torch.tanh(logits / cap) * cap
        lp = torch.log_softmax(logits, -1)
        t = int(lp.argmax())
        if t in eos:
            break
        toks.append(t)
        lps.append(float(lp[t]))
        h = lm(inputs_embeds=lm.embed_tokens(torch.tensor([[t]], device=vis.device)),
               position_ids=torch.tensor([[nxt + step]], device=vis.device), past_key_values=cache,
               use_cache=True).last_hidden_state
    return toks, lps


@torch.no_grad()
def small_pass(m, x, max_new):
    """probe_q3.small_pass for Gemma: in decode steps the sliding-window layers' caches hold only the last window,
    so their attention rows are shorter than the sequence; align the visual columns to each row's key range
    (columns that fell out of the window add nothing at that step)."""
    out = m.generate(**x, max_new_tokens=max_new, do_sample=False, output_attentions=True, return_dict_in_generate=True)
    ids = x["input_ids"][0]
    L = len(ids)
    vis = (ids == m.config.image_token_id).nonzero().squeeze(1)
    imp = torch.zeros(len(vis), device=ids.device)
    for step, layers in enumerate(out.attentions):
        total = L if step == 0 else L + step
        for a in layers:
            rows = a[0, :, int(vis[-1]) + 1:] if step == 0 else a[0, :, -1:]
            off = total - a.shape[-1]
            ok = vis - off >= 0
            imp[ok] += rows[:, :, vis[ok] - off].float().sum(dim=(0, 1))
    return out.sequences[0, L:].tolist(), imp.cpu().numpy()


def run_doc(r, proc, small, large, eos, rng):
    task, dec = r["task"], (lambda ids: clean(proc.tokenizer.decode(ids, skip_special_tokens=True).strip()))  # noqa: E731
    img = Image.open(r["image"]).convert("RGB")
    s_ids, sgp = small_pass(small, encode(proc, img, r["prompt"], False), MAX_NEW[task])
    x = encode(proc, img, r["prompt"], True).to(large.device)  # the large model may sit on another GPU
    P = large_prep(large, x)
    n = int(P["vis"].sum())
    assert n == len(sgp), (n, len(sgp))
    full, flp = pruned_answer(large, P, None, MAX_NEW[task], eos)
    res = dict(task=task, doc_id=r["doc_id"], n_visual=n, small=dec(s_ids), full=dec(full),
               lp_full=float(np.mean(flp)) if flp else 0.0)
    for name, mp in ((k, v) for k, v in dict(fastv=P["fastv"], sgp_s=sgp).items() if k in G4_MAPS):
        order = np.argsort(-np.asarray(mp, np.float64), kind="stable")
        for ratio in RATIOS:
            toks, lps = pruned_answer(large, P, np.sort(order[:max(1, int(n * ratio))]), MAX_NEW[task], eos)
            res[f"{name}@{ratio:g}"] = dec(toks)
            res[f"lp:{name}@{ratio:g}"] = float(np.mean(lps)) if lps else 0.0
    return res, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="chartqa,docvqa_val,textvqa_val,gqa")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="also compare the unpruned path with HF generate()")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    two = torch.cuda.device_count() > 1  # E4B on cuda:0, 31B on cuda:1 when two GPUs are visible
    proc, small, large = processor(), load(SMALL, "cuda:0"), load(LARGE, "cuda:1" if two else "cuda:0")
    e = large.generation_config.eos_token_id
    eos = set(e if isinstance(e, list) else [e])
    si, sk = map(int, a.shard.split("/"))
    for task in a.tasks.split(","):
        recs = [json.loads(line) for line in open(f"{PROBE}/{task}.jsonl")][: a.limit or None]
        recs = [r for p, r in enumerate(recs) if p % sk == si]
        path = f"{OUT}/out_{task}{'_check' if a.check else ''}{f'.s{si}of{sk}' if sk > 1 else ''}.jsonl"
        done = {json.loads(line)["doc_id"] for line in open(path) if line.strip()} if os.path.exists(path) else set()
        rng, t0, k = np.random.default_rng(0), time.time(), 0
        with open(path, "a") as f:
            for r in recs:
                if r["doc_id"] in done:
                    continue
                res, x = run_doc(r, proc, small, large, eos, rng)
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
