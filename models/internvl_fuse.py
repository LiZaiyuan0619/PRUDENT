"""Prune the large VLM with IF-Prune's map fused with a second map read from a dump.

IF-Prune's map is computed here exactly as internvl_kl does; the second map comes from an internvl_sgp or
internvl_obs dump, so nothing else is rerun. The rules are in fuse_rules.py.

FUSE_SPEC = <rule>[:<observer>][@<ratio>]. The paper uses
  rankavg:sgp     the guide's ranking, the average of the ranks of IF-Prune's map and SGL's map
  single:ifp32    IF-Prune's map alone, in float32 (multiple-choice and yes-or-no questions)
The other rules (ifp, interleave, wrank<w>, tiebreak, entw, union, tie*, thumb*) are kept for completeness.
SGP_MAPS = the internvl_sgp dump (observer "sgp"); OBS_MAPS = internvl_obs dump(s), comma-separated.
FUSE_BASE = ifp (IF-Prune's bf16 map as computed here) or ifp32 (its float32 recompute from the OBS_MAPS dump, as in
the paper), the map that every rule fuses with.
FUSE_DUMP_PATH = per question: tokens kept and the overlap of the two maps' top-5% sets; with FUSE_LOGPROB=1 also the
log-probability statistics of the answer, which give the confidence of the first answer.
FUSE_MERGE = none (default) | disagree | all | passthrough: optional merging of dropped tokens, not used in the paper.
"""
import base64
import json
import os
import re

import numpy as np
import torch
from tqdm import tqdm

from lmms_eval.api.registry import register_model
from lmms_eval.models import fuse_logprob
from lmms_eval.models import fuse_rules as fr
from lmms_eval.models.internvl_kl import DEFAULT_GEN_KWARGS, IMG_CONTEXT_TOKEN, InternVL2_KL, load_image, preprocess_internvl2_5


def parse_spec(spec):
    m = re.fullmatch(r"(ifp|single|interleave|rankavg|wrank|tiebreak|entw|union|tie|thumb)(asc|desc|rand\d+|[0-9.]+)?"
                     r"(?::([a-z0-9_]+))?(?:@([0-9.]+))?", spec)
    assert m, f"bad FUSE_SPEC {spec!r}"
    rule, arg, obs, ratio = m.groups()
    return rule, arg or "", obs, float(ratio) if ratio else None


def load_observer(obs):
    """-> {doc_id: float32 map} for one observer."""
    maps = {}
    if obs == "sgpans":  # the answer tokens' part of SGP: SGP's map minus its prompt-only part (base_prompt)
        s, p = load_observer("sgp"), load_observer("base_prompt")
        return {d: np.clip(s[d] - p[d], 0, None).astype(np.float32) for d in s}
    if obs == "sgp":
        for line in open(os.environ["SGP_MAPS"]):
            r = json.loads(line)
            maps[r["doc_id"]] = np.frombuffer(base64.b64decode(r["sgp_importance_f32_b64"]), dtype=np.float32).copy()
        return maps
    for path in os.environ["OBS_MAPS"].split(","):  # several files when the dump was sharded
        for line in open(path):
            r = json.loads(line)
            i = r["maps"].index(obs)
            blk = np.frombuffer(base64.b64decode(r["block_f32_b64"]), dtype=np.float32).reshape(len(r["maps"]), r["n_visual"])
            assert r["doc_id"] not in maps, ("doc in two dump files", r["doc_id"])
            maps[r["doc_id"]] = blk[i].copy()
    return maps


def apply_rule(rule, arg, a, b, ratio, num_patches):
    if rule == "single":
        return fr.single(b, ratio)
    if rule == "interleave":
        return fr.interleave(a, b, ratio)
    if rule == "rankavg":
        return fr.wrank(a, b, ratio, 0.5)
    if rule == "wrank":
        return fr.wrank(a, b, ratio, float(arg))
    if rule == "tiebreak":
        return fr.tiebreak(a, b, ratio)
    if rule == "entw":
        return fr.entw(a, b, ratio)
    if rule == "union":
        return fr.union(a, b, ratio)
    if rule == "tie":
        return fr.tie_noise(a, ratio, arg, seed=int(arg[4:]) if arg.startswith("rand") else 0)
    if rule == "thumb":
        return fr.thumb_quota(a, ratio, num_patches, float(arg))
    raise ValueError(rule)


@register_model("internvl_fuse")
class InternVL2_Fuse(InternVL2_KL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spec = os.environ["FUSE_SPEC"]
        self.rule, self.arg, self.obs, self.ratio_override = parse_spec(self.spec)
        self.obs_maps = load_observer(self.obs) if self.obs else None
        self.dump = os.environ.get("FUSE_DUMP_PATH")
        self.base = os.environ.get("FUSE_BASE", "ifp")
        assert self.base in ("ifp", "ifp32") and not (self.base == "ifp32" and self.rule == "ifp"), self.base
        self.base_maps = load_observer("ifp32") if self.base == "ifp32" else None
        self.merge = os.environ.get("FUSE_MERGE", "none")
        # FUSE_LOGPROB=1 adds the answer log-probs to the dump (fuse_logprob.py); greedy answers are unchanged
        self.logprob = os.environ.get("FUSE_LOGPROB") == "1"
        if self.logprob:
            fuse_logprob.install(self.large_model.language_model)
            self.eos_id = self.large_model_tokenizer.convert_tokens_to_ids("<|im_end|>")
        assert self.merge in ("none", "disagree", "all", "passthrough"), self.merge

    def generate_until(self, requests):
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        device = self._device
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            gen_kwargs = {k: gen_kwargs.get(k, v) for k, v in DEFAULT_GEN_KWARGS.items()}  # as internvl_kl
            visuals = self.flatten([doc_to_visual(self.task_dict[task][split][doc_id])])
            assert self.modality == "image" and len(visuals) == 1, "fusion study: one image per sample"
            pixel_values = torch.cat([load_image(v).to(torch.bfloat16).cuda() for v in visuals], dim=0)
            P = pixel_values.shape[0]
            contexts = "<image>\n" + contexts
            ret = preprocess_internvl2_5("internvl2_5", [[{"from": "human", "value": contexts}]], self.small_model_tokenizer,
                                         [256 * P], group_by_length=True, num_image=1)
            position_ids = ret["attention_mask"].long().cumsum(-1) - 1
            position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
            self.small_model.img_context_token_id = self.small_model_tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            imp = self.small_model(input_ids=ret["input_ids"].to(device), labels=ret["labels"].to(device),
                                   attention_mask=ret["attention_mask"].to(device), position_ids=position_ids.to(device),
                                   pixel_values=pixel_values.to(device),
                                   image_flags=torch.tensor([1] * P, dtype=torch.long).to(device),
                                   infer_entropy_only=True).importance
            ratio = self.ratio_override or gen_kwargs.get("large_model_prune_ratio", 0.05)
            a = imp.detach().float().view(-1)
            if self.base_maps is not None:
                a = torch.from_numpy(self.base_maps[doc_id]).to(a.device)
                assert len(a) == imp.numel(), (doc_id, len(a), imp.numel())
            n = len(a)
            if self.rule == "ifp":  # exactly internvl_kl: the bf16 map as is
                score, keep, b = imp, ratio, None
            else:
                b = None
                if self.obs_maps is not None:
                    b = torch.from_numpy(self.obs_maps[doc_id]).to(a.device)
                    assert len(b) == n, (doc_id, len(b), n)
                score, keep = apply_rule(self.rule, self.arg, a, b, ratio, P)
            if self.dump:
                k5 = int(n * 0.05)
                ov = (len(fr.topk_set(a, k5) & fr.topk_set(b, k5)) / k5) if b is not None else None
                rec = dict(task=task, doc_id=doc_id, n_visual=n, num_patches=int(P),
                           kept=int(n * keep), overlap=ov, spec=self.spec, base=self.base)
            generation_config = dict(
                num_beams=gen_kwargs.get("num_beams", 1), max_new_tokens=gen_kwargs.get("max_new_tokens", 1024),
                min_new_tokens=1, do_sample=gen_kwargs.get("do_sample", False),
                return_dict_in_generate=False, output_scores=False, output_attentions=False,
                large_model_prune_layer=gen_kwargs.get("large_model_prune_layer", 0.05),
                large_model_prune_ratio=keep, visual_token_importance=score)
            if self.merge != "none":
                with torch.no_grad():
                    vit = self.large_model.extract_feature(pixel_values)
                feat = vit.reshape(-1, vit.shape[-1])
                n_merged = 0
                if self.merge != "passthrough":
                    s32 = score.float().view(-1)
                    feat, n_merged = fr.merge_plan(s32, keep, a, b if b is not None else a, feat, self.merge)
                generation_config["visual_features"] = feat.reshape(vit.shape)
                if self.dump:
                    with open(self.dump + ".merge", "a") as f:
                        f.write(json.dumps(dict(doc_id=doc_id, merged=n_merged)) + "\n")
            res.append(self.large_model.chat(tokenizer=self.large_model_tokenizer, pixel_values=pixel_values,
                                             question=contexts, generation_config=generation_config, large_model=True))
            if self.dump:
                if self.logprob:
                    rec["lp"] = fuse_logprob.stats(self.large_model.language_model, self.eos_id)
                with open(self.dump, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            pbar.update(1)
        pbar.close()
        return res
