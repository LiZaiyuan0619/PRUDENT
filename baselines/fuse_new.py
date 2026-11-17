#!/usr/bin/env python3
"""CDPruner, MMTok and ZOO-Prune as existing baselines on InternVL3. Run through baselines/entry.py:
    python baselines/entry.py --model internvl_fuse --model_args ...   with FUSE_SPEC=cdp | mmtok | zoo
Each selects k = int(n * ratio) tokens before the large model's LLM from InternVL's own features and hands them to the
unchanged pruning path of internvl_kl (visual_token_importance with the k on top), as fuse_vz.py does. Each port
follows the official code named below; internvl_fuse is patched in-process on top of fuse_vz's patch, never edited.
  cdp    CDPruner (Zhang et al., NeurIPS'25; CDPruner/llava/model/llava_arch.py encode_images): conditional DPP with
         kernel diag(r) S diag(r), S = cosine of the projector outputs, r = min-max normalized cosine between each
         projector output and the mean LLM input embedding of the instruction (the paper's recipe for MLLMs without a
         paired text encoder, Sec. 3.3; the official code only has LLaVA's CLIP path); its greedy MAP inference.
         Relevance sign as the official code ((-relevance).mean, CDP_SIGN=-1).
  mmtok  MMTok (Dong et al., ICLR'26; MMTok/mmtok, Qwen2.5-VL path): greedy maximum coverage of the text tokens (LLM
         input embeddings of "Question: <prompt>", softmax temperature 0.01 over cosine with the projector outputs)
         and of the vision tokens (ViT features averaged over each 2x2 group, temperature 0.2), alpha 0.5.
  zoo    ZOO-Prune (Kim et al., CVPR'26; ZOO-Prune/LLaVA/llava/model/llava_arch.py): sensitivity = mean over m = 64
         random unit directions u of |J u| at the projector's MLP (InternVL's mlp1 after its input LayerNorm = the
         Linear-GELU-Linear the official code perturbs in LLaVA; ZOO_STAGE=full includes the LayerNorm, which turns the
         score into ~1/std of each token); J u by forward-mode differentiation, the h -> 0 limit of the official
         central difference (h = 0.01 vanishes in bf16); then SADS: the most sensitive first, then argmax of min
         cosine distance to the chosen (projector outputs) x min-max normalized sensitivity.
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fuse_vz as VZ  # noqa: E402  (importing it patches internvl_fuse for vzip / tome; we chain on top)
from fuse_vz import F  # noqa: E402  (lmms_eval.models.internvl_fuse)

RULES = ("cdp", "mmtok", "zoo")


def feats(m, pixel_values):
    """One pass of the large model's InternViT -> x [n, 4C] (pre-projector, pixel-shuffled as extract_feature makes
    it), feat [P, 256, D] = mlp1(x) (what the LLM gets), pooled [n, C] (the 4 patches of each 2x2 group averaged)."""
    assert m.select_layer == -1, m.select_layer
    vit = m.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True).last_hidden_state[:, 1:]
    B, N, C = vit.shape
    side = int(N ** 0.5)
    x = m.pixel_shuffle(vit.reshape(B, side, side, C), scale_factor=m.downsample_ratio)
    x = x.reshape(B, -1, x.shape[-1])  # 4C channels = 4 contiguous patch vectors (pixel_shuffle's two views)
    pooled = x.reshape(B, x.shape[1], 4, C).float().mean(2)
    return x.reshape(-1, x.shape[-1]), m.mlp1(x), pooled.reshape(-1, C)


def text_emb(m, tok, text):
    """LLM input embeddings [T, D] of a text (no special tokens)."""
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(m.language_model.device)
    return m.language_model.get_input_embeddings()(ids)[0]


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


_F32 = {}


def zoo_sensitivity(mlp1, x, m_dirs=64, chunk=8, seed=0, stage=None):
    """ZOO-Prune's sensitivity: mean over m random unit directions u (shared by all tokens) of |J(x_i) u| on an fp32
    copy of the projector. stage "mlp" (default): InternVL's mlp1 minus its input LayerNorm, i.e. Linear-GELU-Linear
    as LLaVA's mm_projector the official code perturbs; "full": all of mlp1 -- through the LayerNorm the sensitivity
    is ~1/std of each token and the text regions are dropped first."""
    stage = stage or os.environ.get("ZOO_STAGE", "mlp")
    key = (id(mlp1), stage)
    if key not in _F32:
        import copy
        _F32[key] = copy.deepcopy(mlp1 if stage == "full" else mlp1[1:]).float()
    f32 = _F32[key]
    x = (x if stage == "full" else mlp1[0](x)).float()
    g = torch.Generator(device=x.device).manual_seed(seed)
    u = torch.randn(m_dirs, x.shape[-1], generator=g, device=x.device)
    u = u / (u.norm(dim=-1, keepdim=True) + 1e-12)
    s, n = torch.zeros(len(x), device=x.device), len(x)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True  # only for these products (~1e-3 relative)
    try:
        for c in range(0, m_dirs, chunk):
            uc = u[c:c + chunk]
            _, ju = torch.func.jvp(f32, (x.repeat(len(uc), 1),), (uc.repeat_interleave(n, 0),))
            s += ju.norm(dim=-1).view(len(uc), n).sum(0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return s / m_dirs


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


def select(rule, m, tok, pixel_values, contexts, ratio):
    """-> (score [n] with the k kept on top, feat [P, 256, D], n, k)."""
    x, feat, pooled = feats(m, pixel_values)
    fl = feat.reshape(-1, feat.shape[-1])
    n = fl.shape[0]
    k = int(n * ratio)  # as the pruning layer counts (topk(int(n * keep)))
    if rule == "cdp":
        sel = cdp_select(fl, text_emb(m, tok, contexts), k, float(os.environ.get("CDP_SIGN", "-1")))
    elif rule == "mmtok":
        sel = mmtok_select(fl, pooled, text_emb(m, tok, "Question: " + " ".join(contexts.split())), k)
    else:
        sel = sads_select(fl, zoo_sensitivity(m.mlp1, x), k)
    score = torch.zeros(n, device=fl.device)
    score[sel] = 1.0
    return score, feat, n, k


_parse, _generate = F.parse_spec, F.InternVL2_Fuse.generate_until


def parse_spec(spec):
    return (spec, "", None, None) if spec in RULES else _parse(spec)


def generate_until(self, requests):
    if self.rule not in RULES:
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
            torch.cuda.synchronize()
            t0 = time.time()
            score, feat, n, k = select(self.rule, m, self.large_model_tokenizer, pixel_values, contexts, ratio)
            torch.cuda.synchronize()
            sel_ms = 1000 * (time.time() - t0)
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
                       overlap=None, spec=self.spec, base=self.base, sel_ms=round(sel_ms, 2))
            if self.logprob:
                rec["lp"] = F.fuse_logprob.stats(m.language_model, self.eos_id)
            with open(self.dump, "a") as f:
                f.write(json.dumps(rec) + "\n")
        pbar.update(1)
    pbar.close()
    return res


F.parse_spec, F.InternVL2_Fuse.generate_until = parse_spec, generate_until

if __name__ == "__main__":
    from lmms_eval.__main__ import cli_evaluate
    cli_evaluate()
