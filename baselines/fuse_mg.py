#!/usr/bin/env python3
"""Two GPUs for one large model (InternVL3-78B: 147 GB in bf16). internvl_kl loads the large checkpoint
with device_map = the process's one GPU; importing this module makes the checkpoint named by LARGE_SPLIT_CKPT
(only that one) load split over the visible GPUs, by internvl_kl.split_model's recipe with the layer count read
from its config: GPU 0 holds the ViT, the projector, the embeddings, the final norm / head, the last LLM layer and
about half a GPU of layers; the other layers spread evenly. The small models stay on GPU 0. Run as a script it is
lmms-eval's command line with the patch applied (baselines/entry.py imports it when LARGE_SPLIT_CKPT is set)."""
import json
import math
import os
import sys

import torch

sys.path.append(os.path.join(os.environ["IFPRUNE_REPO"],
                             "internvl_kl"))
from internvl.model.internvl_chat import InternVLChatModel  # noqa: E402  (the class internvl_kl uses)


def split_map(ckpt, ngpu):
    n = json.load(open(f"{ckpt}/config.json"))["llm_config"]["num_hidden_layers"]
    per = math.ceil(n / (ngpu - 0.5))
    counts = [per] * ngpu
    counts[0] = math.ceil(per * 0.5)
    if os.environ.get("LARGE_SPLIT_GPU0_LAYERS"):  # to even out the free memory of the GPUs
        counts[0] = int(os.environ["LARGE_SPLIT_GPU0_LAYERS"])
        rest = math.ceil((n - counts[0]) / max(1, ngpu - 1))
        counts[1:] = [rest] * (ngpu - 1)
    dm, c = {}, 0
    for g, k in enumerate(counts):
        for _ in range(k):
            if c < n:
                dm[f"language_model.model.layers.{c}"] = g
                c += 1
    for name in ("vision_model", "mlp1", "language_model.model.tok_embeddings", "language_model.model.embed_tokens",
                 "language_model.output", "language_model.model.norm", "language_model.lm_head",
                 f"language_model.model.layers.{n - 1}",
                 "prior_mu", "prior_logvar", "mean_logvar_lgkld"):  # IF-Prune's KL head lives on the class too
        dm[name] = 0
    return dm


_orig = InternVLChatModel.from_pretrained.__func__


def _from_pretrained(cls, path, *args, **kwargs):
    target = os.environ.get("LARGE_SPLIT_CKPT")
    split = target and torch.cuda.device_count() > 1 and os.path.realpath(str(path)) == os.path.realpath(target)
    if split:
        kwargs["device_map"] = split_map(path, torch.cuda.device_count())
    model = _orig(cls, path, *args, **kwargs)
    if split:  # internvl_kl's single-device branch calls large_model.to(device): keep the split instead
        model.to = lambda *a, **k: model
        print(f"[fuse_mg] {path} split over {torch.cuda.device_count()} GPUs", flush=True)
    return model


InternVLChatModel.from_pretrained = classmethod(_from_pretrained)

if __name__ == "__main__":
    from lmms_eval.__main__ import cli_evaluate
    cli_evaluate()
