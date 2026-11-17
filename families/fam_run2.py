#!/usr/bin/env python3
"""fam_run.py on the general benchmarks of the cross-family tables: GQA, the VQAv2 subset, OK-VQA, POPE and MMBench (en
dev), for Qwen3-VL (2B -> 8B / 32B) and Gemma 4 (E4B -> 31B). Same per-question work (fam_run.run_doc); prompts from
data/prompts; images from the HF datasets (GQA's from its images config by imageId; the VQAv2 subset from its four
validation shards in the HF cache).
    python families/fam_run2.py --family qwen3vl --large 8B --task gqa --shard 0/4 [--limit N] [--out-dir ...]
-> $PRUDENT_OUT/families/<family>-<large>/<task>.s<i>of<k>.jsonl (scored with lmms-eval's process_results)"""
import argparse
import glob
import json
import os
import time

import fam_run as FR

VQAV2 = sorted(glob.glob(f"{FR.HF}/hub/datasets--lmms-lab--VQAv2/snapshots/*/data/validation-0000[0-3]-of-00068.parquet"))
DATA = {"gqa": ("lmms-lab/GQA", "testdev_balanced_instructions", "testdev"), "vqav2_val_sub": ("parquet", None, "train"),
        "ok_vqa_val2014": ("lmms-lab/OK-VQA", None, "val2014"), "pope": ("lmms-lab/POPE", None, "test"),
        "mmbench_en_dev": ("lmms-lab/MMBench", "en", "dev")}
FR.MAX_NEW.update({t: 16 for t in DATA})


def prompts(task):
    """{doc_id: prompt}: the text every model saw (data/prompts/<task>.jsonl, from the InternVL3-8B full-token run)."""
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "prompts", f"{task}.jsonl")
    return {r["doc_id"]: r["prompt"] for r in map(json.loads, open(f))}


def images(task):
    """-> callable doc_id -> PIL image, in lmms-eval's doc order."""
    from datasets import load_dataset
    path, name, split = DATA[task]
    ds = load_dataset("parquet", data_files=VQAV2, split="train") if task == "vqav2_val_sub" else load_dataset(path, name, split=split)
    if task != "gqa":
        return lambda i: ds[i]["image"].convert("RGB")
    imgs = {r["id"]: r["image"] for r in load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")}
    return lambda i: imgs[ds[i]["imageId"]].convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen3vl", "gemma4"], required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--task", choices=list(DATA), required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=FR.OUT)
    ap.add_argument("--ratios", default=",".join(f"{r:g}" for r in FR.RATIOS))
    a = ap.parse_args()
    si, sk = map(int, a.shard.split("/"))
    img, pr = images(a.task), prompts(a.task)
    ids = [i for p, i in enumerate(sorted(pr)) if p % sk == si][: a.limit or None]
    out = f"{a.out_dir}/{a.family}-{a.large}/{a.task}.s{si}of{sk}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(line)["doc_id"] for line in open(out) if line.strip()} if os.path.exists(out) else set()
    F, proc, small, big, enc_s, enc_l, dec = FR.setup(a.family, a.large)
    e = big.generation_config.eos_token_id
    eos = set(e if isinstance(e, list) else [e])
    t0, k = time.time(), 0
    with open(out, "a") as f:
        for i in (i for i in ids if i not in done):
            res = FR.run_doc(F, small, big, enc_s, enc_l, dec, img(i), pr[i], a.task, eos, [float(r) for r in a.ratios.split(",")])
            f.write(json.dumps(dict(task=a.task, doc_id=i, **res)) + "\n")
            f.flush()
            k += 1
            if k % 50 == 0:
                print(f"{a.task} {k}/{len(ids) - len(done)} docs, {(time.time() - t0) / k:.2f} s/doc", flush=True)
    print(f"[fam_run2] {out}: {len(ids)} docs in the shard, {k} new", flush=True)


if __name__ == "__main__":
    main()
