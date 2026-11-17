#!/bin/bash
# Stage 2: the checker answers one benchmark from the full image and logs the log-probability of its tokens.
#   bash scripts/run_checker.sh <task>          (for the checker ablation, CHECKER=OpenGVLab/InternVL2_5-1B, the
#                                               guide, or CHECKER=OpenGVLab/InternVL3-8B)
set -euo pipefail
source "$(dirname "$0")/common.sh"
TASK=$1
mkdir -p "$OUT/checker"; rm -f "$OUT/checker/$TASK.jsonl"; cd "$IFPRUNE_REPO/lmms-eval"
CHECKER_DUMP_PATH=$OUT/checker/$TASK.jsonl python -m lmms_eval --model internvl_checker \
  --model_args pretrained=$CHECKER --tasks "$TASK" --batch_size 1 \
  --log_samples --output_path "$OUT/checker/$TASK.lmms" ${LIMIT:+--limit $LIMIT}
