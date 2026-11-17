"""The checker of PRUDENT: a small VLM answers each question from the full image, without pruning.

Its answer and the log-probabilities of its answer tokens are appended to CHECKER_DUMP_PATH, one record per question
(doc_id, ans, n, sum, mean, min, first_p, first_margin, eos_p, truncated); `min` is the log-probability of the least
probable token, which decides when the checker abstains. Greedy decoding, the model's own chat template and the
task's max_new_tokens, with the tiles of internvl_kl.load_image.

    CHECKER_DUMP_PATH=<file.jsonl> python -m lmms_eval --model internvl_checker \\
        --model_args pretrained=OpenGVLab/InternVL3-2B --tasks <task> --batch_size 1 --log_samples --output_path <dir>
"""
import json
import os
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.internvl_kl import DEFAULT_GEN_KWARGS, IMG_CONTEXT_TOKEN, load_image


def gen_stats(seq, logits, eos):
    """seq [T] generated ids, logits: T tensors [1, vocab] (generate's output_logits) -> statistics over the answer
    tokens before the first EOS; eos_p = P(EOS) right after them, None when the answer hit max_new_tokens."""
    lp = torch.stack([torch.log_softmax(x[0].float(), -1) for x in logits])
    n = int((seq != eos).long().cumprod(0).sum())
    tok = lp[torch.arange(n, device=lp.device), seq[:n]]
    p0 = lp[0].exp()
    top2 = torch.topk(p0, 2).values
    truncated = n == len(seq)
    return dict(n=n, sum=float(tok.sum()), mean=float(tok.mean()) if n else 0.0, min=float(tok.min()) if n else 0.0,
                first_p=float(p0[seq[0]]), first_margin=float(top2[0] - top2[1]),
                eos_p=None if truncated else float(lp[n, eos].exp()), truncated=truncated)


@register_model("internvl_checker")
class InternVLChecker(lmms):
    def __init__(self, pretrained: str = "OpenGVLab/InternVL3-2B", batch_size: str = "1", **kwargs):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(pretrained, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                               use_flash_attn=False, trust_remote_code=True).eval().cuda()
        self.conv = __import__(type(self.model).__module__.rsplit(".", 1)[0] + ".conversation",
                               fromlist=["get_conv_template"])
        self.model.img_context_token_id = self.tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.dump = os.environ["CHECKER_DUMP_PATH"]
        self._rank, self._world_size = 0, 1

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def generate_until(self, requests) -> List[str]:
        res, m, tok = [], self.model, self.tok
        with open(self.dump, "a") as out, torch.no_grad():
            for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in tqdm([r.args for r in requests]):
                vis = doc_to_visual(self.task_dict[task][split][doc_id])
                pv = load_image(vis[0] if isinstance(vis, list) else vis).to(torch.bfloat16).cuda()
                t = self.conv.get_conv_template(m.template)
                t.system_message = m.system_message
                t.append_message(t.roles[0], "<image>\n" + contexts)
                t.append_message(t.roles[1], None)
                q = t.get_prompt().replace(
                    "<image>", "<img>" + IMG_CONTEXT_TOKEN * m.num_image_token * pv.shape[0] + "</img>", 1)
                enc = tok(q, return_tensors="pt")
                eos = tok.convert_tokens_to_ids(t.sep.strip())
                mnt = gen_kwargs.get("max_new_tokens", DEFAULT_GEN_KWARGS["max_new_tokens"])
                g = m.generate(pixel_values=pv, input_ids=enc["input_ids"].cuda(),
                               attention_mask=enc["attention_mask"].cuda(), max_new_tokens=mnt, do_sample=False,
                               num_beams=1, eos_token_id=eos, return_dict_in_generate=True, output_logits=True)
                ans = tok.batch_decode(g.sequences, skip_special_tokens=True)[0].split(t.sep.strip())[0].strip()
                out.write(json.dumps(dict(doc_id=doc_id, ans=ans, **gen_stats(g.sequences[0], g.logits, eos))) + "\n")
                out.flush()
                res.append(ans)
        return res

    def loglikelihood(self, requests):
        raise NotImplementedError

    def generate_until_multi_round(self, requests):
        raise NotImplementedError
