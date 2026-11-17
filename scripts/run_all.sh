#!/bin/bash
# Everything one benchmark needs for one large VLM at the budgets 2.5%, 5% and 10%: the guide's maps, the checker,
# full tokens, IF-Prune at K, and the large VLM at the first-pass, re-answer and one-pass fractions of each budget.
#   bash scripts/run_all.sh <model> <task>        e.g. InternVL3-8B chartqa;  LIMIT=50 for a quick check
set -euo pipefail
D=$(dirname "$0")
MODEL=$1; TASK=$2
RANK=$(cd "$D/.." && python -c "from prudent.config import ranking; print(ranking('$TASK'))")
[ -f "${OUT:-runs}/maps/obs_$TASK.jsonl" ] || bash "$D/run_guide.sh" "$TASK"
[ -f "${OUT:-runs}/checker/$TASK.jsonl" ] || bash "$D/run_checker.sh" "$TASK"
for F in 1 0.025 0.05 0.1; do bash "$D/run_large.sh" "$MODEL" "$TASK" single-ifp32 $F; done
for F in 0.01 0.025 0.05 0.1 0.2; do
  [ "$RANK" = single-ifp32 ] && [ -d "${OUT:-runs}/$MODEL/$TASK/single-ifp32-r$F" ] && continue
  bash "$D/run_large.sh" "$MODEL" "$TASK" "$RANK" $F
done
# ABLATIONS=1: also IF-Prune's ranking at the first-pass and re-answer fractions (Table 5, "IF-Prune's ranking ...")
if [ "${ABLATIONS:-0}" = 1 ] && [ "$RANK" != single-ifp32 ]; then
  for F in 0.01 0.2; do bash "$D/run_large.sh" "$MODEL" "$TASK" single-ifp32 $F; done
fi
