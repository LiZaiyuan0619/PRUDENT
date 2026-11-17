#!/bin/bash
# Stage 1: the large VLM answers one benchmark from a fraction of its visual tokens, kept by a ranking:
#   rankavg-sgp   the guide's ranking (average of the ranks of SGL's and IF-Prune's maps)
#   single-ifp32  IF-Prune's map alone (closed-form questions, the IF-Prune baseline, and fraction 1 = full tokens)
# It also logs the log-probabilities of every answer (the confidence of a first answer).
#   bash scripts/run_large.sh <model> <task> <ranking> <fraction>     e.g. InternVL3-8B chartqa rankavg-sgp 0.025
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODEL=$1; TASK=$2; SPEC=$3; F=$(frac "$4")
LARGE=${LARGE_CKPT:-OpenGVLab/$MODEL}
RUN=$OUT/$MODEL/$TASK/$SPEC-r$F
mkdir -p "$(dirname "$RUN")"; rm -rf "$RUN" "$RUN.lp.jsonl"; cd "$IFPRUNE_REPO/lmms-eval"
FUSE_SPEC=${SPEC/-/:} FUSE_BASE=ifp32 FUSE_LOGPROB=1 FUSE_DUMP_PATH=$RUN.lp.jsonl \
SGP_MAPS=$OUT/maps/sgp_$TASK.jsonl OBS_MAPS=$OUT/maps/obs_$TASK.jsonl \
ENTRY=(-m lmms_eval); [ -n "${LARGE_SPLIT_CKPT:-}" ] && ENTRY=("$(cd "$(dirname "$0")/.." && pwd)/baselines/entry.py")  # 78B on two GPUs
python "${ENTRY[@]}" --model internvl_fuse --model_args pretrained=$IFP_1B,large_checkpoint=$LARGE \
  --tasks "$TASK" --batch_size 1 --gen_kwargs large_model_prune_ratio=$F,large_model_prune_layer=$LAYER \
  --log_samples --output_path "$RUN" ${LIMIT:+--limit $LIMIT}
