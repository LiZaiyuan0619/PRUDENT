#!/usr/bin/env python3
"""CPU checks of the token selection of the existing methods in baselines/ (no model weights, no GPU): on random
features each method keeps exactly k tokens, returns the shapes the pruning path expects, and is deterministic.
Needs the lmms-eval of setup.sh:  IFPRUNE_REPO=third_party/IF-Prune python tests/test_baselines_cpu.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines"))
import fuse_new as NEW  # noqa: E402
import fuse_vz as VZ  # noqa: E402

torch.manual_seed(0)
n, c, d, t, k = 1024, 64, 48, 12, 51  # visual tokens, pre-projector channels, LLM width, text tokens, kept (5%)
x, keys, att = torch.randn(n, c), torch.randn(n, c), torch.rand(n)
feat, pooled, temb = torch.randn(n, d), torch.randn(n, c // 4), torch.randn(t, d)


def kept(score):
    return int((score > 0).sum())


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    assert ok, name


s, out = VZ.vzip_select(att, keys, x, k)
check("VisionZip keeps k tokens", kept(s) == k, f"({kept(s)} of {n})")
check("VisionZip keeps the [CLS]-dominant tokens", bool((s[att.topk(round(k * 54 / 64)).indices] == 2).all()))
check("VisionZip writes merged features in place", out.shape == x.shape and not torch.equal(out, x))
s, out = VZ.tome_select(keys, x, k)
check("ToMe keeps k tokens", kept(s) == k and out.shape == x.shape, f"({kept(s)})")
sel = NEW.cdp_select(feat, temb, k, -1.0)
check("CDPruner selects k distinct tokens", len(set(sel.tolist())) == k)
check("CDPruner is deterministic", torch.equal(sel, NEW.cdp_select(feat, temb, k, -1.0)))
sel = NEW.mmtok_select(feat, pooled, temb, k)
check("MMTok selects k distinct tokens", len(set(sel.tolist())) == k)
mlp1 = torch.nn.Sequential(torch.nn.LayerNorm(c), torch.nn.Linear(c, d), torch.nn.GELU(), torch.nn.Linear(d, d))
sens = NEW.zoo_sensitivity(mlp1, x, m_dirs=16, chunk=4)
check("ZOO-Prune sensitivity, one value per token", sens.shape == (n,) and bool(torch.isfinite(sens).all()))
check("ZOO-Prune sensitivity is deterministic", torch.allclose(sens, NEW.zoo_sensitivity(mlp1, x, m_dirs=16, chunk=4)))
sel = NEW.sads_select(feat, sens, k)
check("ZOO-Prune SADS selects k distinct tokens", len(set(sel.tolist())) == k)
check("ZOO-Prune SADS starts from the most sensitive token", int(sel[0]) == int(sens.argmax()))
print("all checks passed")
