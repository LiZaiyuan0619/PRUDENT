"""Importance maps of the guide, computed once per question and dumped, so that internvl_fuse reads them
instead of recomputing them.

PRUDENT uses ifp32: IF-Prune's map recomputed in float32 from the same hidden states (its head and KL in float32; the
released map is bf16 and has many ties at small budgets). The other observers are cheaper alternatives kept for
completeness: ifp_post and ifp_img2q (attention of IF-Prune's small model), base_prompt (InternVL2.5-1B on the prompt
only), vit_cls, llm_l1, llm_l2 and llm_l2_last (attention inside the large VLM), txt_sim (similarity to the question)
and div (farthest-point coverage).

OBS_DUMP_PATH=<file.jsonl> (required): per question the maps (float32, base64, one [n_maps, n_visual] block), their
names, a checksum of IF-Prune's bf16 map and the time each observer took.
OBS_SHARD=<i>/<n>: only questions with doc_id % n == i (the others get "" and no record), so that a large benchmark
can run as several jobs.
OBS_RUN_LARGE=1 (default): also answer with the large VLM pruned by IF-Prune's map exactly as internvl_kl does;
0 returns "" answers.
"""
import base64
import copy
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval.api.registry import register_model
from lmms_eval.models.internvl_kl import (DEFAULT_GEN_KWARGS, IMG_CONTEXT_TOKEN, InternVL2_KL, InternVLChatConfig,
                                          InternVLChatModel, load_image, preprocess_internvl2_5)
from internvl.conversation import get_conv_template  # noqa: E402  (repo path added by internvl_kl)
from internvl.model.internlm2.modeling_internlm2 import InternLM2Attention  # noqa: E402
from internvl.model.qwen2.modeling_qwen2 import Qwen2Attention  # noqa: E402  (InternVL3's LLM)
from internvl.model.qwen2.modeling_qwen2 import QWEN2_ATTENTION_CLASSES  # noqa: E402

MAPS = ["ifp32", "ifp_post", "ifp_img2q", "base_prompt", "vit_cls", "llm_l1", "llm_l2", "llm_l2_last",
        "txt_sim", "div"]
BASE_1B = "OpenGVLab/InternVL2_5-1B"


def to_eager_qwen2(lm):
    """Switch a (peft-wrapped) repo Qwen2 LLM to eager attention so it can return the weights."""
    n = 0
    for mod in lm.modules():
        if type(mod).__name__ == "Qwen2FlashAttention2":
            mod.__class__ = QWEN2_ATTENTION_CLASSES["eager"]
            n += 1
        if type(mod).__name__ == "Qwen2Model":
            mod.config._attn_implementation = "eager"
            mod._attn_implementation = "eager"
    assert n > 0, "no flash attention module found"
    return n


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def pool_like_pixel_shuffle(model, per_patch):
    """[P, 1024] patch values -> [P * 256] in the order extract_feature lays out the visual tokens."""
    P, m = per_patch.shape
    h = int(m ** 0.5)
    x = model.pixel_shuffle(per_patch.reshape(P, h, h, 1), scale_factor=model.downsample_ratio)
    return x.reshape(P, -1, x.shape[-1]).sum(-1).reshape(-1)


def farthest_point_order(feat, m):
    """Greedy max-min order over L2-normalised features; the first m by greedy, the rest by their final
    distance to the chosen set. -> score (higher = earlier)."""
    f = torch.nn.functional.normalize(feat.float(), dim=-1)
    n = len(f)
    start = int((1 - f @ f.mean(0)).argmax())  # the token farthest from the mean
    chosen = [start]
    dmin = 1 - f @ f[start]
    dmin[start] = -1
    for _ in range(min(m, n) - 1):
        j = int(dmin.argmax())
        chosen.append(j)
        dmin = torch.minimum(dmin, 1 - f @ f[j])
        dmin[torch.tensor(chosen, device=f.device)] = -1
    score = dmin.clone()  # rest: by distance to the chosen set, all below the chosen ones
    score[torch.tensor(chosen, device=f.device)] = torch.arange(len(chosen), 0, -1, device=f.device, dtype=score.dtype) + 10.0
    return score


@register_model("internvl_obs")
class InternVL2_Obs(InternVL2_KL):
    def __init__(self, sgp_small: str = BASE_1B, **kwargs):
        super().__init__(**kwargs)
        self.dump = os.environ["OBS_DUMP_PATH"]
        self.run_large = os.environ.get("OBS_RUN_LARGE", "1") == "1"
        sh = os.environ.get("OBS_SHARD", "0/1").split("/")
        self.shard, self.n_shards = int(sh[0]), int(sh[1])
        self.check = os.environ.get("OBS_CHECK") == "1"  # smoke tests: eager layers 0-2 vs the flash forward
        # a second copy of IF-Prune's small model whose LLM returns attention weights; its ViT stays flash so
        # the visual features equal the production path
        cfg = InternVLChatConfig.from_json_file(f"{self.path}/config.json")
        cfg.llm_config.attn_implementation = "flash_attention_2"
        cfg.vision_config.use_flash_attn = True
        self.small_eager = InternVLChatModel.from_pretrained(
            self.path, config=cfg, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map=self.device_map).eval()
        to_eager_qwen2(self.small_eager.language_model)
        self.head32 = copy.deepcopy(self.small_model.mean_logvar_lgkld).float()
        # SGP's small model (plain InternVL2.5-1B), eager LLM as in internvl_sgp
        self.base_tok = AutoTokenizer.from_pretrained(sgp_small, trust_remote_code=True, use_fast=False)
        self.base = AutoModel.from_pretrained(sgp_small, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                              use_flash_attn=False, trust_remote_code=True).eval().to(self._device)
        assert self.base.language_model.config._attn_implementation == "eager"

    @torch.no_grad()
    def ifp_part(self, pv, contexts, P, ms):
        """-> (IF-Prune's bf16 map exactly as internvl_kl computes it, {ifp32, ifp_post, ifp_img2q})."""
        dev, tok = self._device, self.small_model_tokenizer
        ret = preprocess_internvl2_5("internvl2_5", [[{"from": "human", "value": contexts}]], tok,
                                     [256 * P], group_by_length=True, num_image=1)
        pos = ret["attention_mask"].long().cumsum(-1) - 1
        pos.masked_fill_(ret["attention_mask"] == 0, 1)
        img_id = tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.small_model.img_context_token_id = img_id
        ids = ret["input_ids"].to(dev)
        cap = {}
        hook = self.small_model.mean_logvar_lgkld.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0]))
        t0 = sync()
        imp = self.small_model(input_ids=ids, labels=ret["labels"].to(dev), attention_mask=ret["attention_mask"].to(dev),
                               position_ids=pos.to(dev), pixel_values=pv,
                               image_flags=torch.ones(P, dtype=torch.long, device=dev), infer_entropy_only=True).importance
        ms["ifp"] = 1000 * (sync() - t0)
        hook.remove()
        # the same head and KL in float32, from the same (bf16) hidden states
        t0 = sync()
        m, x = self.small_model, cap["x"].float()
        mw, lv = self.head32(x).split([x.shape[-1], 1], dim=-1)
        mu, plv = m.prior_mu.float(), m.prior_logvar.float()
        tm = torch.sigmoid(mw) * (x - mu) + mu
        out = {"ifp32": (plv / 2 - lv / 2 + (lv.exp() + (tm - mu) ** 2) / (2 * plv.exp()) - 0.5).mean(-1)}
        ms["ifp32"] = 1000 * (sync() - t0)
        # attention of the same small model on the same input
        se = self.small_eager
        se.img_context_token_id = img_id
        t0 = sync()
        emb = se.language_model.get_input_embeddings()(ids)
        sel = ids[0] == img_id
        emb[0, sel] = se.extract_feature(pv).reshape(-1, emb.shape[-1]).to(emb.dtype)
        att = se.language_model(inputs_embeds=emb, attention_mask=ret["attention_mask"].to(dev), position_ids=pos.to(dev),
                                output_attentions=True, return_dict=True).attentions
        cols = sel.nonzero().view(-1)
        first, after = int(cols[0]), int(cols[-1]) + 1
        starts = (ids[0, :first] == tok.convert_tokens_to_ids("<|im_start|>")).nonzero().view(-1)
        q = slice(int(starts[-1]) + 1, first - 1)  # "user\n<question>\n", up to '<img>'
        post = img2q = 0.0
        for a in att:  # [1, heads, N, N]
            post = post + a[0, :, after:][:, :, cols].float().sum((0, 1))
            img2q = img2q + a[0][:, cols][:, :, q].float().sum((0, 2))
        out["ifp_post"], out["ifp_img2q"] = post, img2q
        ms["ifp_attn"] = 1000 * (sync() - t0)
        return imp, out

    @torch.no_grad()
    def base_part(self, pv, contexts, P, ms):
        """SGP's prefill term without generation: plain 1B, prompt as internvl_sgp builds it."""
        m, tok, dev = self.base, self.base_tok, self._device
        conv = __import__(type(m).__module__.rsplit(".", 1)[0] + ".conversation", fromlist=["get_conv_template"])
        template = conv.get_conv_template(m.template)
        template.system_message = m.system_message
        template.append_message(template.roles[0], contexts)
        template.append_message(template.roles[1], None)
        query = template.get_prompt().replace("<image>", "<img>" + IMG_CONTEXT_TOKEN * m.num_image_token * P + "</img>", 1)
        ids = tok(query, return_tensors="pt")["input_ids"].to(dev)
        m.img_context_token_id = tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        sel = ids[0] == m.img_context_token_id
        cols = sel.nonzero().view(-1)
        assert int(cols[-1] - cols[0]) + 1 == len(cols) == 256 * P
        t0 = sync()
        emb = m.language_model.get_input_embeddings()(ids)
        emb[0, sel] = m.extract_feature(pv).reshape(-1, emb.shape[-1]).to(emb.dtype)
        att = m.language_model(inputs_embeds=emb, output_attentions=True, return_dict=True).attentions
        after = int(cols[-1]) + 1
        bp = 0.0
        for a in att:
            bp = bp + a[0, :, after:][:, :, cols].float().sum((0, 1))
        ms["base_prompt"] = 1000 * (sync() - t0)
        return {"base_prompt": bp}

    @torch.no_grad()
    def large_part(self, pv, contexts, P, ms, check=False):
        """The 8B's own signals: ViT [CLS] attention, LLM layers 1-2 attention, embedding similarity, coverage."""
        L, tok, dev = self.large_model, self.large_model_tokenizer, self._device
        attn = L.vision_model.encoder.layers[-1].attn
        cap = {}
        attn.use_flash_attn = False  # the last ViT layer only, so its attention probabilities pass attn_drop
        hook = attn.attn_drop.register_forward_hook(lambda m, i, o: cap.__setitem__("a", o))
        t0 = sync()
        try:
            vit = L.extract_feature(pv)  # [P, 256, C]
        finally:
            hook.remove()
            attn.use_flash_attn = True
        out = {"vit_cls": pool_like_pixel_shuffle(L, cap["a"][:, :, 0, 1:].float().mean(1))}
        ms["vit_cls"] = 1000 * (sync() - t0)
        # the prompt exactly as chat() builds it
        template = get_conv_template(L.template)
        template.system_message = L.system_message
        template.append_message(template.roles[0], contexts)
        template.append_message(template.roles[1], None)
        query = template.get_prompt().replace("<image>", "<img>" + IMG_CONTEXT_TOKEN * L.num_image_token * P + "</img>", 1)
        ids = tok(query, return_tensors="pt")["input_ids"].to(dev)
        sel = ids[0] == tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        cols = sel.nonzero().view(-1)
        assert int(cols[-1] - cols[0]) + 1 == len(cols) == L.num_image_token * P
        lm = L.language_model
        emb = lm.get_input_embeddings()(ids)
        C = emb.shape[-1]
        emb[0, sel] = vit.reshape(-1, C).to(emb.dtype)
        N, after = ids.shape[1], int(cols[-1]) + 1
        t0 = sync()
        pos = torch.arange(N, device=dev)[None]
        mask = torch.full((N, N), torch.finfo(emb.dtype).min, device=dev, dtype=emb.dtype).triu(1)[None, None]
        h = emb
        for li in range(3):
            layer = lm.model.layers[li]
            if hasattr(layer, "attention"):  # InternLM2 (InternVL2 / 2.5)
                ao, w, _ = InternLM2Attention.forward(layer.attention, hidden_states=layer.attention_norm(h), attention_mask=mask,
                                                      position_ids=pos, output_attentions=True, use_cache=False,
                                                      prunded_sequence_length=0)
            else:  # Qwen2 (InternVL3; IF-Prune's internvl/model/qwen2, the same eager signature)
                ao, w, _ = Qwen2Attention.forward(layer.self_attn, hidden_states=layer.input_layernorm(h), attention_mask=mask,
                                                  position_ids=pos, output_attentions=True, use_cache=False,
                                                  prunded_sequence_length=0)
            if li >= 1:
                out[f"llm_l{li}"] = w[0, :, after:][:, :, cols].float().mean((0, 1))
            if li == 2:
                out["llm_l2_last"] = w[0, :, -1, cols].float().mean(0)
            h = h + ao
            h = h + (layer.feed_forward(layer.ffn_norm(h)) if hasattr(layer, "attention") else layer.mlp(layer.post_attention_layernorm(h)))
        ms["llm_attn"] = 1000 * (sync() - t0)
        if check:  # the eager re-run of layers 0-2 must agree with the model's own (flash) forward
            ref = lm.model(inputs_embeds=emb, output_hidden_states=True, return_dict=True).hidden_states[3]
            out["_check_l3_maxdiff"] = float((ref.float() - h.float()).abs().max())
        t0 = sync()
        im_end = (ids[0, after:] == tok.convert_tokens_to_ids("<|im_end|>")).nonzero().view(-1)
        q_emb = emb[0, after + 1:after + int(im_end[0])].float()  # the question, between '</img>' and '<|im_end|>'
        v = vit.reshape(-1, C).float()
        out["txt_sim"] = (torch.nn.functional.normalize(v, dim=-1)
                          @ torch.nn.functional.normalize(q_emb, dim=-1).T).max(-1).values
        ms["txt_sim"] = 1000 * (sync() - t0)
        t0 = sync()
        out["div"] = farthest_point_order(v, m=int(0.25 * len(v)))
        ms["div"] = 1000 * (sync() - t0)
        return out

    def generate_until(self, requests):
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if doc_id % self.n_shards != self.shard:
                res.append("")
                pbar.update(1)
                continue
            gen_kwargs = {k: gen_kwargs.get(k, v) for k, v in DEFAULT_GEN_KWARGS.items()}  # as internvl_kl
            visuals = self.flatten([doc_to_visual(self.task_dict[task][split][doc_id])])
            assert self.modality == "image" and len(visuals) == 1, "observer dump: one image per sample"
            pv = torch.cat([load_image(v).to(torch.bfloat16).cuda() for v in visuals], dim=0)
            P = pv.shape[0]
            contexts = "<image>\n" + contexts
            ms = {}
            imp, maps = self.ifp_part(pv, contexts, P, ms)
            maps.update(self.base_part(pv, contexts, P, ms))
            maps.update(self.large_part(pv, contexts, P, ms, check=self.check))
            n = imp.numel()
            block = torch.stack([maps[k].float().reshape(-1) for k in MAPS]).cpu().numpy().astype(np.float32)
            assert block.shape == (len(MAPS), n), (block.shape, n)
            k5 = int(n * 0.05)
            with open(self.dump, "a") as f:
                f.write(json.dumps(dict(
                    task=task, doc_id=doc_id, num_patches=int(P), n_visual=int(n), maps=MAPS,
                    ifprune_sum=float(imp.float().sum()),
                    ifp32_top5_overlap=len(set(imp.float().view(-1).topk(k5).indices.tolist())
                                           & set(maps["ifp32"].topk(k5).indices.tolist())) / k5,
                    ms={k: round(v, 2) for k, v in ms.items()},
                    check_l3_maxdiff=maps.get("_check_l3_maxdiff"),
                    block_f32_b64=base64.b64encode(block.tobytes()).decode("ascii"))) + "\n")
            if self.run_large:  # IF-Prune exactly as internvl_kl: its bf16 map, same generation config
                generation_config = dict(
                    num_beams=gen_kwargs.get("num_beams", 1), max_new_tokens=gen_kwargs.get("max_new_tokens", 1024),
                    min_new_tokens=1, do_sample=gen_kwargs.get("do_sample", False),
                    return_dict_in_generate=False, output_scores=False, output_attentions=False,
                    large_model_prune_layer=gen_kwargs.get("large_model_prune_layer", 0.05),
                    large_model_prune_ratio=gen_kwargs.get("large_model_prune_ratio", 0.05),
                    visual_token_importance=imp)
                res.append(self.large_model.chat(tokenizer=self.large_model_tokenizer, pixel_values=pv,
                                                 question=contexts, generation_config=generation_config, large_model=True))
            else:
                res.append("")
            pbar.update(1)
        pbar.close()
        return res
