#!/bin/bash
# Stage 0: the guide's two maps for one benchmark, dumped once (they do not depend on the large VLM):
# SGL's attention map of the pre-trained 1B (internvl_sgp) and IF-Prune's map in float32 (internvl_obs).
#   bash scripts/run_guide.sh <task> [large VLM checkpoint used only to finish the lmms-eval pass]
set -euo pipefail
source "$(dirname "$0")/common.sh"
TASK=$1; LARGE=${2:-${LARGE_CKPT:-OpenGVLab/InternVL3-8B}}
mkdir -p "$OUT/maps"; cd "$IFPRUNE_REPO/lmms-eval"
rm -f "$OUT/maps/sgp_$TASK.jsonl" "$OUT/maps/obs_$TASK.jsonl"
SGP_DUMP_PATH=$OUT/maps/sgp_$TASK.jsonl python -m lmms_eval --model internvl_sgp \
  --model_args pretrained=$IFP_1B,large_checkpoint=$LARGE,sgp_small=$SMALL --tasks "$TASK" --batch_size 1 \
  --gen_kwargs large_model_prune_ratio=0.05,large_model_prune_layer=$LAYER --log_samples --output_path "$OUT/maps/sgp_$TASK" ${LIMIT:+--limit $LIMIT}
OBS_DUMP_PATH=$OUT/maps/obs_$TASK.jsonl OBS_RUN_LARGE=0 python -m lmms_eval --model internvl_obs \
  --model_args pretrained=$IFP_1B,large_checkpoint=$LARGE,sgp_small=$SMALL --tasks "$TASK" --batch_size 1 \
  --gen_kwargs large_model_prune_ratio=0.05,large_model_prune_layer=$LAYER --log_samples --output_path "$OUT/maps/obs_$TASK" ${LIMIT:+--limit $LIMIT}
