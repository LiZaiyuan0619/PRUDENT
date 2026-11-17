"""SGL's ranking as a second importance estimate (SGL, arXiv 2412.03324).

A pre-trained small VLM answers the question, and the importance of a visual token is the attention it receives,
summed over all layers and heads, from every prompt token after the image (prefill) and from every generated token,
as get_attention_rank() in the SGL code. The small model is the pre-trained InternVL2.5-1B, the base of IF-Prune's
estimator, with eager attention so that the weights are returned, and greedy decoding.

Per question this model computes both maps on the same tiles, IF-Prune's (as internvl_kl) and SGL's, and prunes the
large VLM with SGL's map through the path internvl_kl uses for IF-Prune's (same layer, ratio and decoding).

    --model internvl_sgp --model_args pretrained=<IF-Prune 1B>,large_checkpoint=<large VLM>,sgp_small=<InternVL2.5-1B>
    SGP_DUMP_PATH=<file.jsonl>   per question: SGL's map (float32, base64), the small model's answer, the overlap of
                                 the two maps' top-5% sets and a checksum of IF-Prune's map
"""
import base64
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval.api.registry import register_model
from lmms_eval.models.internvl_kl import DEFAULT_GEN_KWARGS, IMG_CONTEXT_TOKEN, InternVL2_KL, load_image, preprocess_internvl2_5

_SGP_DUMP = os.environ.get("SGP_DUMP_PATH")


def sgp_importance(attentions, visual_cols, after_image):
    """attentions: generate()'s per-step tuples of per-layer [1, heads, q, k] weights.
    -> [n_visual] summed attention to the visual columns (SGL get_attention_rank)."""
    imp = 0.0
    for step, layers in enumerate(attentions):
        for a in layers:  # one layer at a time: the prefill weights are [heads, ~2k, ~2k] per layer
            rows = a[0, :, after_image:] if step == 0 else a[0, :, 0:1]
            imp = imp + rows[:, :, visual_cols].float().sum(dim=(0, 1))
    return imp


@register_model("internvl_sgp")
class InternVL2_SGP(InternVL2_KL):
    def __init__(self, sgp_small: str = "OpenGVLab/InternVL2_5-1B", **kwargs):
        super().__init__(**kwargs)
        self.sgp_tokenizer = AutoTokenizer.from_pretrained(sgp_small, trust_remote_code=True, use_fast=False)
        self.sgp_model = AutoModel.from_pretrained(
            sgp_small, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=False,
            trust_remote_code=True).eval().to(self._device)
        assert self.sgp_model.language_model.config._attn_implementation == "eager"

    @torch.no_grad()
    def sgp(self, pixel_values, question, max_new_tokens):
        """-> (importance [n_visual], answer) from the plain small model, as its chat() builds the prompt."""
        m, tok = self.sgp_model, self.sgp_tokenizer
        conv = __import__(type(m).__module__.rsplit(".", 1)[0] + ".conversation", fromlist=["get_conv_template"])
        template = conv.get_conv_template(m.template)
        template.system_message = m.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        n_patches = pixel_values.shape[0]
        query = template.get_prompt().replace(
            "<image>", "<img>" + IMG_CONTEXT_TOKEN * m.num_image_token * n_patches + "</img>", 1)
        enc = tok(query, return_tensors="pt")
        input_ids = enc["input_ids"].to(self._device)
        m.img_context_token_id = tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        visual_cols = (input_ids[0] == m.img_context_token_id).nonzero().view(-1)
        assert len(visual_cols) == m.num_image_token * n_patches
        assert int(visual_cols[-1] - visual_cols[0]) + 1 == len(visual_cols), "image tokens not contiguous"
        out = m.generate(pixel_values=pixel_values, input_ids=input_ids, attention_mask=enc["attention_mask"].to(self._device),
                         max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                         eos_token_id=tok.convert_tokens_to_ids(template.sep.strip()),
                         output_attentions=True, return_dict_in_generate=True)
        imp = sgp_importance(out.attentions, visual_cols, int(visual_cols[-1]) + 1)
        answer = tok.batch_decode(out.sequences, skip_special_tokens=True)[0].split(template.sep.strip())[0].strip()
        return imp, answer

    def generate_until(self, requests):
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        device = self._device
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            gen_kwargs = {k: gen_kwargs.get(k, v) for k, v in DEFAULT_GEN_KWARGS.items()}  # as internvl_kl
            visuals = self.flatten([doc_to_visual(self.task_dict[task][split][doc_id])])
            assert self.modality == "image" and len(visuals) == 1, "SGP diagnosis: one image per sample"
            pixel_values = torch.cat([load_image(v).to(torch.bfloat16).cuda() for v in visuals], dim=0)
            num_patches = pixel_values.shape[0]
            contexts = "<image>\n" + contexts
            # IF-Prune's map, exactly as internvl_kl (kept for the overlap and as a consistency check)
            ret = preprocess_internvl2_5("internvl2_5", [[{"from": "human", "value": contexts}]], self.small_model_tokenizer,
                                         [256 * num_patches], group_by_length=True, num_image=1)
            position_ids = ret["attention_mask"].long().cumsum(-1) - 1
            position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
            self.small_model.img_context_token_id = self.small_model_tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            kl = self.small_model(input_ids=ret["input_ids"].to(device), labels=ret["labels"].to(device),
                                  attention_mask=ret["attention_mask"].to(device), position_ids=position_ids.to(device),
                                  pixel_values=pixel_values.to(device),
                                  image_flags=torch.tensor([1] * num_patches, dtype=torch.long).to(device),
                                  infer_entropy_only=True).importance
            imp, answer = self.sgp(pixel_values, contexts, gen_kwargs["max_new_tokens"])
            assert imp.numel() == kl.numel(), (imp.numel(), kl.numel())
            if _SGP_DUMP:
                k = max(1, int(imp.numel() * 0.05))
                a, b = set(imp.topk(k).indices.tolist()), set(kl.view(-1).topk(k).indices.tolist())
                with open(_SGP_DUMP, "a") as f:
                    f.write(json.dumps(dict(
                        task=task, doc_id=doc_id, num_patches=int(num_patches), n_visual=int(imp.numel()),
                        sgp_answer=answer, top5_overlap=len(a & b) / k,
                        ifprune_sum=float(kl.detach().float().sum()),
                        sgp_importance_f32_b64=base64.b64encode(imp.float().cpu().numpy().astype(np.float32).tobytes()).decode("ascii"),
                    )) + "\n")
            generation_config = dict(
                num_beams=gen_kwargs.get("num_beams", 1), max_new_tokens=gen_kwargs.get("max_new_tokens", 1024),
                min_new_tokens=1, do_sample=gen_kwargs.get("do_sample", False),
                return_dict_in_generate=False, output_scores=False, output_attentions=False,
                large_model_prune_layer=gen_kwargs.get("large_model_prune_layer", 0.05),
                large_model_prune_ratio=gen_kwargs.get("large_model_prune_ratio", 0.05),
                visual_token_importance=imp.to(kl.dtype))
            res.append(self.large_model.chat(tokenizer=self.large_model_tokenizer, pixel_values=pixel_values,
                                             question=contexts, generation_config=generation_config, large_model=True))
            pbar.update(1)
        pbar.close()
        return res
