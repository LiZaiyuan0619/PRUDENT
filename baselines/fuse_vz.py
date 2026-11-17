#!/usr/bin/env python3
"""VisionZip and ToMe as existing baselines on InternVL. Run through baselines/entry.py (scripts/run_baseline.sh):
    python baselines/entry.py --model internvl_fuse --model_args ...   with FUSE_SPEC=vzip | tome
Both decide the kept set and the merged features before the large model's LLM and hand them to the unchanged
pruning path of internvl_kl (visual_token_importance with their k tokens on top, visual_features with the merged
features), as FUSE_MERGE does. This file replaces internvl_fuse.parse_spec and InternVL2_Fuse.generate_until in-process
(other rules go to the original), so internvl_fuse itself is unchanged.
  vzip  VisionZip (Yang et al., CVPR'25; official code dvlab-research/VisionZip clip_encoder.py) on the large
        model's InternViT at its feature layer (the last): dominant = the tokens the [CLS] token attends to most
        (summed over heads); of the k kept, 54/64 dominant, the rest contextual: the other tokens split uniformly
        into targets and to-merge, each merged into the target of largest key cosine; contextual = target + mean of
        its merged tokens. InternVL's 2x2 pixel shuffle groups 4 patches into one LLM token: attention summed and
        keys concatenated by that same grouping; merging happens before the projector (mlp1), as VisionZip merges
        CLIP's hidden states before LLaVA's projector.
  tome  ToMe (Bolya et al., ICLR'23) at the LLM input: bipartite soft matching (alternate A / B in token order,
        cosine of the same pooled keys), the r most similar A tokens merged into their B match by size-weighted
        mean, repeated until k tokens remain; before the projector.
k = int(n * ratio) as the pruning layer counts (internvl_kl's modeling_qwen2 / internlm2: topk(int(n * keep))).
"""
import os
import re
import sys

import torch


def vzip_select(cls_att, keys, x, k, dom_share=54 / 64):
    """cls_att [n] (pooled [CLS] attention), keys [n, d] (pooled keys), x [n, c] (pre-projector features), k kept
    -> (score [n] with exactly the k kept on top, x with the contextual tokens written at their targets)."""
    n = len(cls_att)
    kd = min(k, max(1, round(k * dom_share)))
    m = k - kd
    dom = cls_att.float().topk(kd).indices
    score = torch.zeros(n, device=x.device)
    score[dom] = 2.0
    if m == 0:
        return score, x
    rest = torch.ones(n, dtype=torch.bool, device=x.device)
    rest[dom] = False
    rest = rest.nonzero().squeeze(1)  # the non-dominant tokens in token order
    step = max(1, len(rest) // m)
    t_local = torch.arange(0, len(rest), step, device=x.device)[:m]
    is_t = torch.zeros(len(rest), dtype=torch.bool, device=x.device)
    is_t[t_local] = True
    tgt, mrg = rest[is_t], rest[~is_t]
    kn = keys.float() / keys.float().norm(dim=-1, keepdim=True)
    assign = (kn[mrg] @ kn[tgt].T).argmax(1)  # each to-merge token -> its most similar target
    xs = x.float()
    agg = torch.zeros(len(tgt), x.shape[1], device=x.device).index_add_(0, assign, xs[mrg])
    cnt = torch.zeros(len(tgt), device=x.device).index_add_(0, assign, torch.ones(len(mrg), device=x.device))
    out = x.clone()
    out[tgt] = (xs[tgt] + agg / cnt.clamp(min=1).unsqueeze(1)).to(x.dtype)
    score[tgt] = 1.0
    return score, out


def tome_select(keys, x, k):
    """ToMe's bipartite soft matching repeated until k tokens remain. keys [n, d] (the metric), x [n, c]
    -> (score [n] with the k survivors on top, x with the survivors' size-weighted means at their positions)."""
    n, dev = len(x), x.device
    pos, size = torch.arange(n, device=dev), torch.ones(n, device=dev)
    feat = x.float().clone()
    met = keys.float() / keys.float().norm(dim=-1, keepdim=True)
    while len(pos) > k:
        a, b = torch.arange(0, len(pos), 2, device=dev), torch.arange(1, len(pos), 2, device=dev)
        r = min(len(a), len(pos) - k)
        best, j = (met[a] @ met[b].T).max(1)
        order = best.argsort(descending=True)
        ma, ka = a[order[:r]], a[order[r:]]  # A tokens merged / kept
        bi, sa = j[order[:r]], size[a[order[:r]]]
        sb = size[b].clone().index_add_(0, bi, sa)
        fb = (feat[b] * size[b, None]).index_add_(0, bi, feat[ma] * sa[:, None]) / sb[:, None]
        mb = (met[b] * size[b, None]).index_add_(0, bi, met[ma] * sa[:, None]) / sb[:, None]
        mb = mb / mb.norm(dim=-1, keepdim=True)
        p2 = torch.cat([pos[ka], pos[b]])
        o = p2.argsort()
        pos, size = p2[o], torch.cat([size[ka], sb])[o]
        feat, met = torch.cat([feat[ka], fb])[o], torch.cat([met[ka], mb])[o]
    score = torch.zeros(n, device=dev)
    score[pos] = 1.0
    out = x.clone()
    out[pos] = feat.to(x.dtype)
    return score, out


def vit_stats(m, pixel_values):
    """One pass of the large model's InternViT -> (pre-projector features x [P, 256, 4C] as extract_feature makes
    them, pooled [CLS] attention [P * 256], pooled keys [P * 256, 4C]) at the feature layer (select_layer -1)."""
    assert m.select_layer == -1, m.select_layer
    att, cap = m.vision_model.encoder.layers[-1].attn, {}
    hook = att.register_forward_pre_hook(lambda mod, args: cap.__setitem__("x", args[0]))
    try:
        vit = m.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True).last_hidden_state
    finally:
        hook.remove()
    B, N, C = cap["x"].shape
    H = att.num_heads
    qkv = att.qkv(cap["x"]).reshape(B, N, 3, H, C // H)
    q, kk = qkv[:, :, 0], qkv[:, :, 1]
    if att.qk_normalization:
        q = att.q_norm(q.flatten(-2, -1)).view(B, N, H, C // H)
        kk = att.k_norm(kk.flatten(-2, -1)).view(B, N, H, C // H)
    a = torch.einsum("bhd,bnhd->bhn", q[:, 0].float() * att.scale, kk.float()).softmax(-1).sum(1)[:, 1:]
    side = int((N - 1) ** 0.5)

    def ps(t):  # the projector's 2x2 grouping, exactly as extract_feature applies it
        t = m.pixel_shuffle(t.reshape(B, side, side, -1).contiguous(), scale_factor=m.downsample_ratio)
        return t.reshape(B, -1, t.shape[-1])
    return ps(vit[:, 1:]), ps(a[..., None]).sum(-1).reshape(-1), ps(kk[:, 1:].flatten(-2, -1)).flatten(0, 1)


sys.path.insert(0, os.path.join(os.environ["IFPRUNE_REPO"],
                                "lmms-eval"))
from lmms_eval.models import internvl_fuse as F  # noqa: E402  (patched in-process, never edited)

_parse, _generate = F.parse_spec, F.InternVL2_Fuse.generate_until


def parse_spec(spec):
    return (spec, "", None, None) if spec in ("vzip", "tome") else _parse(spec)


def generate_until(self, requests):
    if self.rule not in ("vzip", "tome"):
        return _generate(self, requests)
    from tqdm import tqdm
    res, m = [], self.large_model
    pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
    for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
        gen_kwargs = {k: gen_kwargs.get(k, v) for k, v in F.DEFAULT_GEN_KWARGS.items()}
        visuals = self.flatten([doc_to_visual(self.task_dict[task][split][doc_id])])
        assert self.modality == "image" and len(visuals) == 1, "one image per sample"
        pixel_values = torch.cat([F.load_image(v).to(torch.bfloat16).cuda() for v in visuals], dim=0)
        ratio = self.ratio_override or gen_kwargs.get("large_model_prune_ratio", 0.05)
        with torch.no_grad():
            x, cls_att, keys = vit_stats(m, pixel_values)
            xf = x.reshape(-1, x.shape[-1])
            n, k = xf.shape[0], int(xf.shape[0] * ratio)
            score, xm = vzip_select(cls_att, keys, xf, k) if self.rule == "vzip" else tome_select(keys, xf, k)
            feat = m.mlp1(xm.reshape(x.shape))
        generation_config = dict(
            num_beams=gen_kwargs.get("num_beams", 1), max_new_tokens=gen_kwargs.get("max_new_tokens", 1024),
            min_new_tokens=1, do_sample=gen_kwargs.get("do_sample", False),
            return_dict_in_generate=False, output_scores=False, output_attentions=False,
            large_model_prune_layer=gen_kwargs.get("large_model_prune_layer", 0.05),
            large_model_prune_ratio=ratio, visual_token_importance=score, visual_features=feat)
        res.append(m.chat(tokenizer=self.large_model_tokenizer, pixel_values=pixel_values, question="<image>\n" + contexts,
                          generation_config=generation_config, large_model=True))
        if self.dump:
            rec = dict(task=task, doc_id=doc_id, n_visual=n, num_patches=int(pixel_values.shape[0]), kept=k,
                       overlap=None, spec=self.spec, base=self.base)
            if self.logprob:
                rec["lp"] = F.fuse_logprob.stats(m.language_model, self.eos_id)
            with open(self.dump, "a") as f:
                f.write(__import__("json").dumps(rec) + "\n")
        pbar.update(1)
    pbar.close()
    return res


F.parse_spec, F.InternVL2_Fuse.generate_until = parse_spec, generate_until

if __name__ == "__main__":
    if os.environ.get("LARGE_SPLIT_CKPT"):  # the 78B over two GPUs
        import fuse_mg  # noqa: F401
    from lmms_eval.__main__ import cli_evaluate
    cli_evaluate()
