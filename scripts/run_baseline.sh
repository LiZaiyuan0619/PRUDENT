#!/bin/bash
# A existing pruning method on one benchmark at one fraction of the visual tokens, on the same large VLM, lmms-eval
# pipeline and answer logging as PRUDENT (Section 4.1). Each method keeps its own ranking:
#   ifprune    IF-Prune's released importance estimator (its map in float32)
#   sgl        SGL, the attention of the pre-trained InternVL2.5-1B while it answers
#   fastv      FastV, the attention of the large VLM's second LLM layer from the last prompt token
#   visionzip, tome                  baselines/fuse_vz.py
#   cdpruner, mmtok, zoo-prune       baselines/fuse_new.py
#   bash scripts/run_baseline.sh <model> <task> <method> <fraction>      e.g. InternVL3-8B chartqa visionzip 0.05
# IF-Prune, SGL and FastV need the maps of scripts/run_guide.sh (FastV also the large VLM's own map, made here).
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODEL=$1; TASK=$2; METHOD=$3; F=$(frac "$4")
LARGE=${LARGE_CKPT:-OpenGVLab/$MODEL}
PKG=$(cd "$(dirname "$0")/.." && pwd)
case $METHOD in
  ifprune) SPEC=single:ifp32 ;;  sgl) SPEC=single:sgp ;;  fastv) SPEC=single:llm_l2_last ;;
  visionzip) SPEC=vzip ;;  tome) SPEC=tome ;;  cdpruner) SPEC=cdp ;;  mmtok) SPEC=mmtok ;;  zoo-prune) SPEC=zoo ;;
  *) echo "unknown method: $METHOD" >&2; exit 1 ;;
esac
OBS=$OUT/maps/obs_$TASK.jsonl
cd "$IFPRUNE_REPO/lmms-eval"
if [ "$METHOD" = fastv ]; then  # the large VLM's own attention: maps of this model
  OBS=$OUT/maps/obs_${MODEL}_$TASK.jsonl
  [ -f "$OBS" ] || OBS_DUMP_PATH=$OBS OBS_RUN_LARGE=0 python -m lmms_eval --model internvl_obs \
    --model_args pretrained=$IFP_1B,large_checkpoint=$LARGE,sgp_small=$SMALL --tasks "$TASK" --batch_size 1 \
    --gen_kwargs large_model_prune_ratio=0.05,large_model_prune_layer=$LAYER \
    --log_samples --output_path "$OUT/maps/obs_${MODEL}_$TASK" ${LIMIT:+--limit $LIMIT}
fi
RUN=$OUT/$MODEL/$TASK/$METHOD-r$F
mkdir -p "$(dirname "$RUN")"; rm -rf "$RUN" "$RUN.lp.jsonl"
FUSE_SPEC=$SPEC FUSE_BASE=ifp32 FUSE_LOGPROB=1 FUSE_DUMP_PATH=$RUN.lp.jsonl \
SGP_MAPS=$OUT/maps/sgp_$TASK.jsonl OBS_MAPS=$OBS \
python "$PKG/baselines/entry.py" --model internvl_fuse --model_args pretrained=$IFP_1B,large_checkpoint=$LARGE \
  --tasks "$TASK" --batch_size 1 --gen_kwargs large_model_prune_ratio=$F,large_model_prune_layer=$LAYER \
  --log_samples --output_path "$RUN" ${LIMIT:+--limit $LIMIT}
