#!/usr/bin/env python3
"""Qwen3-VL-4B as the checker on Qwen3-VL-8B and 32B (Appendix L, the 2B keeps guiding): its full-view answers and token
statistics on the five text-rich benchmarks, with fam_run's images, prompts and answer lengths. Greedy, resumable,
sharded by question position.
    python families/fam_checker.py --task chartqa [--shard 0/2] [--limit N]
-> $PRUDENT_OUT/families/qwen3vl-4Bcheck/<task>.s<i>of<k>.jsonl: doc_id, small (the 4B's answer), slp_n / min / mean / first_p / trunc"""
import argparse
import glob
import json
import os
import time

import fam_run as FR  # noqa: E402  (offline HF environment, DATA, MAX_NEW, prompts)

CKPT = sorted(glob.glob(f"{FR.HUB}/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*"))[0]


def main():
    import torch
    from datasets import load_dataset

    import probe_q3 as F
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(FR.DATA), required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=f"{FR.OUT}/qwen3vl-4Bcheck")
    a = ap.parse_args()
    si, sk = map(int, a.shard.split("/"))
    path, name, split = FR.DATA[a.task]
    ds, pr = load_dataset(path, name, split=split), FR.prompts(a.task)
    ids = [i for p, i in enumerate(sorted(pr)) if p % sk == si][: a.limit or None]
    out = f"{a.out_dir}/{a.task}.s{si}of{sk}.jsonl"
    os.makedirs(a.out_dir, exist_ok=True)
    done = {json.loads(l)["doc_id"] for l in open(out) if l.strip()} if os.path.exists(out) else set()
    proc, m = F.processor(), F.load(CKPT)
    e = m.generation_config.eos_token_id
    eos = set(e if isinstance(e, list) else [e])
    t0, k = time.time(), 0
    with open(out, "a") as f, torch.no_grad():
        for i in ids:
            if i in done:
                continue
            x = F.encode(proc, ds[i]["image"].convert("RGB"), pr[i])
            g = m.generate(**x, max_new_tokens=FR.MAX_NEW[a.task], do_sample=False, output_logits=True,
                           return_dict_in_generate=True)
            s = g.sequences[0, x["input_ids"].shape[1]:].tolist()
            lp = torch.stack([torch.log_softmax(z[0].float(), -1) for z in g.logits])
            n = next((j for j, t in enumerate(s) if t in eos), len(s))
            tok = lp[torch.arange(n, device=lp.device), torch.as_tensor(s[:n], device=lp.device)]
            f.write(json.dumps(dict(task=a.task, doc_id=i, small=proc.tokenizer.decode(s, skip_special_tokens=True).strip(),
                                    slp_n=n, slp_min=float(tok.min()) if n else 0.0, slp_mean=float(tok.mean()) if n else 0.0,
                                    slp_first_p=float(lp[0, s[0]].exp()) if s else 0.0, slp_trunc=n == len(s))) + "\n")
            f.flush()
            k += 1
            if k % 200 == 0:
                print(f"{a.task} {k}/{len(ids) - len(done)}, {(time.time() - t0) / k:.2f} s/doc", flush=True)
    print(f"[fam_checker] {out}: {len(ids)} docs in the shard, {k} new", flush=True)


if __name__ == "__main__":
    main()
