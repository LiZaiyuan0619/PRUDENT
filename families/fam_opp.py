#!/usr/bin/env python3
"""The existing methods of the InternVL tables on the other families (Table 4 and Appendix L):
VisionZip, ToMe, CDPruner, MMTok and ZOO-Prune on Qwen3-VL (8B / 32B) and Gemma 4 31B. Each method selects k =
max(1, int(n * ratio)) visual tokens from the large model's own vision features before its LLM, as on InternVL
(baselines/fuse_vz.py, fuse_new.py); the answer then goes through probe_q3 / probe_g4.pruned_answer, the same path as
the FastV / SGL cells of fam_run.py (kept tokens keep their positions; Qwen's DeepStack features follow the tokens).
The selection functions are copied unchanged from fuse_vz.py / fuse_new.py (those import lmms-eval at module level).
Per family, what each method reads:
  vzip   dominant tokens = the most attended at the vision tower's last layer. Neither tower has a [CLS] token, so the
         attention each patch receives, averaged over the query patches and summed over heads, stands in for the
         [CLS] row (our adaptation); pooled by the tower's own patch grouping (Qwen: the merger's 2x2 groups, summed;
         Gemma: the pooler's 3x3 groups, summed). Merge metric = the last layer's keys before RoPE (Qwen: the 4 patch
         keys concatenated, as the merger concatenates patches; Gemma: averaged, as its pooler averages patches).
  tome   the same keys as metric; tokens merged before the projector.
  cdp    projector outputs and the mean LLM input embedding of the prompt (sign -1 as the official code).
  mmtok  projector outputs, the LLM input embeddings of "Question: <prompt>", and the vision tower's last hidden states
         averaged over each token's patches (the official Qwen2.5-VL path's image_features).
  zoo    Qwen only: |J u| over the merger's Linear-GELU-Linear after its LayerNorm (as InternVL's mlp1[1:]). Gemma 4's
         projector is an RMSNorm without scale and one Linear: J u is the same for every token (or, through the norm,
         ~1/rms of the token), so ZOO-Prune is not defined there and is not run.
Merging (vzip's contextual tokens, tome) happens before the projector, as VisionZip merges CLIP's hidden states before
LLaVA's projector; on Qwen the DeepStack mergers' inputs are merged with the same assignment (concatenated along the
feature axis, so one call merges all levels)."""
import copy
import os

import numpy as np
import torch
import torch.nn.functional as Fn

RULES = {"qwen3vl": ("vzip", "tome", "cdp", "mmtok", "zoo"), "gemma4": ("vzip", "tome", "cdp", "mmtok")}


# ---- copied unchanged from baselines/fuse_vz.py (vzip_select, tome_select) and fuse_new.py (cdp_select,
# _greedy_cover, mmtok_select, sads_select) ----


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


def cdp_select(feat, temb, k, sign=1.0):
    """CDPruner: feat [n, D] projector outputs, temb [T, D] instruction embeddings -> k indices (conditional DPP)."""
    f = Fn.normalize(feat.float(), dim=-1)
    r = sign * (f @ Fn.normalize(temb.float().mean(0), dim=-1))
    r = (r - r.min() + 1e-6) / (r.max() - r.min())
    ker = r[:, None] * (f @ f.T) * r[None, :]
    n, dev = len(f), f.device
    cis, di2s = torch.zeros(k, n, device=dev), ker.diagonal().clone()
    sel = torch.empty(k, dtype=torch.long, device=dev)
    for i in range(k):
        j = torch.argmax(di2s)
        sel[i] = j
        eis = (ker[j] - cis[:i, j] @ cis[:i]) / di2s[j].clamp(min=1e-12).sqrt()  # clamp: an exhausted kernel
        cis[i] = eis
        di2s -= eis.square()
        di2s[j] = -float("inf")
    return sel


@torch.jit.script
def _greedy_cover(comb: torch.Tensor, k: int) -> torch.Tensor:
    """MMTok's greedy_merged_jit_kernel (semantic_selector.py) without padding exclusion: comb [m + n, n]."""
    best = torch.zeros(comb.shape[0], device=comb.device, dtype=comb.dtype)
    mask = torch.zeros(comb.shape[1], device=comb.device, dtype=comb.dtype)
    sel = torch.zeros(k, dtype=torch.long, device=comb.device)
    for i in range(k):
        delta = (comb - best.unsqueeze(1)).clamp_min_(0).sum(dim=0) + mask
        j = torch.argmax(delta)
        sel[i] = j
        mask[j] = float("-inf")
        best = torch.maximum(best, comb[:, j])
    return sel


def mmtok_select(feat, pooled, temb, k, alpha=0.5, tv=0.01, vv=0.2):
    """MMTok: feat [n, D] projector outputs, pooled [n, C] vision features, temb [T, D] text-token embeddings."""
    x, xc, z = (Fn.normalize(t.float(), dim=-1) for t in (feat, pooled, temb))
    P = torch.softmax((z @ x.T) / tv, dim=1) / len(z)
    Q = torch.softmax((xc @ xc.T) / vv, dim=1) / len(xc)
    return _greedy_cover(torch.cat([P, alpha * Q], 0), k)


def sads_select(feat, sens, k):
    """ZOO-Prune's SADS (select_tokens_sads): feat [n, D] projector outputs, sens [n]."""
    f = Fn.normalize(feat.float(), dim=-1)
    w = (sens - sens.min()) / (sens.max() - sens.min() + 1e-8) if sens.max() > sens.min() else torch.ones_like(sens)
    sel, free, mind = torch.empty(k, dtype=torch.long, device=f.device), torch.ones(len(f), dtype=torch.bool, device=f.device), None
    for i in range(k):
        j = torch.argmax((w if mind is None else mind * w).masked_fill(~free, float("-inf")))
        sel[i], free[j] = j, False
        d = 1.0 - f @ f[j]
        mind = d if mind is None else torch.minimum(mind, d)
    return sel


# ---- ZOO-Prune's sensitivity: fuse_new.zoo_sensitivity (stage "mlp") with the MLP and its input passed in ----
_F32 = {}


def zoo_mlp(merger):
    """An fp32 copy of Qwen's merger Linear-GELU-Linear, cached per merger (the MLP the official code perturbs)."""
    if id(merger) not in _F32:
        _F32[id(merger)] = copy.deepcopy(torch.nn.Sequential(merger.linear_fc1, merger.act_fn, merger.linear_fc2)).float()
    return _F32[id(merger)]


def zoo_sens(mlp, h, m_dirs=64, chunk=8, seed=0):
    """Mean over m random unit directions u (shared by all tokens) of |J(h_i) u|; h [n, d] = the MLP's input (the
    projector's input after its norm). J u by forward-mode differentiation, as fuse_new.py."""
    h = h.float()
    g = torch.Generator(device=h.device).manual_seed(seed)
    u = torch.randn(m_dirs, h.shape[-1], generator=g, device=h.device)
    u = u / (u.norm(dim=-1, keepdim=True) + 1e-12)
    s, n = torch.zeros(len(h), device=h.device), len(h)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True  # as fuse_new.py: only for these products (~1e-3 relative)
    try:
        for c in range(0, m_dirs, chunk):
            uc = u[c:c + chunk]
            _, ju = torch.func.jvp(mlp, (h.repeat(len(uc), 1),), (uc.repeat_interleave(n, 0),))
            s += ju.norm(dim=-1).view(len(uc), n).sum(0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return s / m_dirs


# ---- Qwen3-VL ----
@torch.no_grad()
def q3_prep(m, x):
    """One pass of the large model's vision tower with hooks -> (P as probe_q3.large_prep without FastV's map, feats):
    att [n] (attention each patch receives at the last layer, summed per merged token), keys [n, 4C] (the last layer's
    keys before RoPE, the token's 4 patches concatenated), X [n, 4 x 4C] (the merger's and the three DeepStack mergers'
    inputs, grouped as the merger groups them), feat [n, D] (projector outputs), pooled [n, C] (MMTok's vision
    features), project (X -> main features, DeepStack features), zoo (-> ZOO-Prune's sensitivity [n])."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
    mm, ids = m.model, x["input_ids"]
    vt, cap = mm.visual, {}
    assert x["image_grid_thw"].shape[0] == 1, "one image per sample"
    am = vt.blocks[-1].attn
    hooks = [am.register_forward_pre_hook(lambda mod, a, kw: cap.__setitem__("attn", (a[0], kw["position_embeddings"])),
                                          with_kwargs=True),
             vt.merger.register_forward_pre_hook(lambda mod, a: cap.__setitem__("last", a[0]))]
    hooks += [mg.register_forward_pre_hook(lambda mod, a, i=i: cap.__setitem__(i, a[0]))
              for i, mg in enumerate(vt.deepstack_merger_list)]
    try:
        vo = mm.get_image_features(x["pixel_values"], x["image_grid_thw"], return_dict=True)
    finally:
        for h in hooks:
            h.remove()
    vis = ids == m.config.image_token_id
    emb = mm.get_input_embeddings()(ids).clone()
    feat = torch.cat(vo.pooler_output, 0)
    emb[vis] = feat.to(emb.dtype)
    pos, _ = mm.get_rope_index(ids, image_grid_thw=x["image_grid_thw"], attention_mask=x["attention_mask"],
                               mm_token_type_ids=x["mm_token_type_ids"])
    P = dict(emb=emb, pos=pos, vis=vis[0], ds=[torch.cat(f, 0) for f in vo.deepstack_features])
    h, (cos, sin) = cap["attn"]
    N, C, H = h.shape[0], h.shape[1], am.num_heads
    n = N // 4
    assert n * 4 == N and n == len(feat) == int(vis.sum()), (N, len(feat), int(vis.sum()))
    q, k, _ = am.qkv(h).reshape(N, 3, H, -1).permute(1, 0, 2, 3).unbind(0)
    qr, kr = apply_rotary_pos_emb_vision(q, k, cos, sin)
    recv = torch.zeros(N, device=h.device)
    for j in range(H):  # one head at a time: [N, N] in fp32, N <= 4 x 1792 patches
        recv += ((qr[:, j].float() @ kr[:, j].float().T) * am.scaling).softmax(-1).mean(0)
    last = cap["last"]
    X = torch.cat([t.reshape(n, 4 * C) for t in [last] + [cap[i] for i in range(len(vt.deepstack_merger_list))]], -1)

    def project(Xm):
        parts = Xm.split(4 * C, -1)
        return vt.merger(parts[0].reshape(-1, C)), [mg(p.reshape(-1, C)) for mg, p in zip(vt.deepstack_merger_list, parts[1:])]

    feats = dict(att=recv.view(n, 4).sum(1), keys=k.reshape(N, C).reshape(n, 4 * C), X=X, feat=feat,
                 pooled=last.reshape(n, 4, C).float().mean(1), project=project,
                 zoo=lambda: zoo_sens(zoo_mlp(vt.merger), vt.merger.norm(last).reshape(n, 4 * C)))
    return P, feats


# ---- Gemma 4 ----
@torch.no_grad()
def g4_prep(m, x):
    """As q3_prep for Gemma 4 (P as probe_g4.large_prep without FastV's map). Visual tokens = the pooler's soft tokens
    (3x3 patches averaged). feats: att [n] (attention each patch receives at the last layer, summed per soft token),
    keys [n, H x d] (the last layer's keys before RoPE, averaged per soft token), X [n, C] (the soft tokens the
    projector embeds), feat [n, D], pooled [n, C] (the encoder's last hidden states averaged per soft token), project."""
    mm, ids = m.model, x["input_ids"]
    vt, cap = mm.vision_tower, {}
    sa = vt.encoder.layers[-1].self_attn
    hooks = [sa.register_forward_pre_hook(lambda mod, a, kw: cap.__setitem__("h", kw["hidden_states"]), with_kwargs=True),
             sa.register_forward_hook(lambda mod, a, o: cap.__setitem__("w", o[1])),
             vt.pooler.register_forward_pre_hook(lambda mod, a, kw: cap.__setitem__("pool", kw), with_kwargs=True)]
    try:
        vo = mm.get_image_features(x["pixel_values"], x.get("image_position_ids"), return_dict=True)
    finally:
        for h in hooks:
            h.remove()
    vis = ids == m.config.image_token_id
    emb = mm.get_input_embeddings()(torch.where(vis, m.config.get_text_config().pad_token_id, ids)).clone()
    feat = torch.cat(vo.pooler_output, 0)
    emb[vis] = feat.to(emb.dtype)
    P = dict(emb=emb, pos=torch.arange(ids.shape[1], device=ids.device)[None], vis=vis[0], mtt=x["mm_token_type_ids"])
    kw = cap["pool"]  # the pooler's inputs: the encoder's last hidden states and the patch positions
    hid, pp, pad, L = kw["hidden_states"][0], kw["pixel_position_ids"][0], kw["padding_positions"][0], kw["output_length"]
    N, dev = hid.shape[0], hid.device
    kk = int((N // L) ** 0.5)
    cp = pp.clamp(min=0)
    kid = torch.div(cp, kk, rounding_mode="floor")
    kid = kid[:, 0] + (cp[:, 0].max() + 1) // kk * kid[:, 1]  # each patch's soft-token slot (_avg_pool_by_positions)
    slot = torch.zeros(L, dtype=torch.bool, device=dev)
    slot[kid] = True  # the pooler's mask: the slots some patch falls in
    ok = ~pad
    cnt = torch.zeros(L, device=dev).index_add_(0, kid[ok], torch.ones(int(ok.sum()), device=dev))
    assert int(slot.sum()) == len(feat) == int(vis.sum()), (int(slot.sum()), len(feat), int(vis.sum()))

    def per_slot(v, mean):
        s = torch.zeros(L, v.shape[-1], device=dev).index_add_(0, kid[ok], v[ok].float())
        return (s / cnt.clamp(min=1)[:, None] if mean else s)[slot]

    w, recv = cap["w"][0], torch.zeros(N, device=dev)  # [H, N, N] after the softmax (padded keys get 0)
    for j in range(w.shape[0]):
        recv += w[j][ok].float().mean(0)
    hs = cap["h"][0]  # the last layer's attention input (after its RMSNorm)
    kpre = sa.k_norm(sa.k_proj(hs).view(N, -1, sa.head_dim)).reshape(N, -1)
    feats = dict(att=per_slot(recv[:, None], False)[:, 0], keys=per_slot(kpre, True), X=vo.last_hidden_state, feat=feat,
                 pooled=per_slot(hid, True), project=lambda Xm: (mm.embed_vision(inputs_embeds=Xm), None), zoo=None)
    return P, feats


# ---- one method's selection ----
def text_emb(m, tok, text):
    """LLM input embeddings [T, D] of a text (no special tokens), as fuse_new.text_emb."""
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(m.device)
    return m.model.get_input_embeddings()(ids)[0]


@torch.no_grad()
def choose(rule, m, tok, P, feats, prompt, k):
    """-> (the kept visual tokens' indices, sorted numpy; P with the merged features written in for vzip / tome)."""
    if rule in ("vzip", "tome"):
        if rule == "vzip":
            score, Xm = vzip_select(feats["att"], feats["keys"], feats["X"], k)
        else:
            score, Xm = tome_select(feats["keys"], feats["X"], k)
        keep = score.topk(k).indices
        main, ds = feats["project"](Xm)
        emb = P["emb"].clone()
        emb[0, P["vis"]] = main.to(emb.dtype)
        P = dict(P, emb=emb) if ds is None else dict(P, emb=emb, ds=[d.to(P["ds"][0].dtype) for d in ds])
    elif rule == "cdp":
        keep = cdp_select(feats["feat"], text_emb(m, tok, prompt), k, float(os.environ.get("CDP_SIGN", "-1")))
    elif rule == "mmtok":
        keep = mmtok_select(feats["feat"], feats["pooled"], text_emb(m, tok, "Question: " + " ".join(prompt.split())), k)
    elif rule == "zoo":
        if "_sens" not in feats:  # the same for every ratio of a doc
            feats["_sens"] = feats["zoo"]()
        keep = sads_select(feats["feat"], feats["_sens"], k)
    else:
        raise ValueError(rule)
    keep = keep.cpu().numpy()
    assert len(keep) == k == len(set(keep.tolist())), (rule, len(keep), k)
    return np.sort(keep), P
