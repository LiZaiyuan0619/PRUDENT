#!/bin/bash
# Fetch IF-Prune at the commit we used (PRUDENT runs on its lmms-eval and InternVL code), apply our changes to its
# lmms-eval and add our model files. Then install IF-Prune's environment (its INSTALLATION.md) and requirements.txt.
#   bash setup.sh [target_dir]      (default: third_party/IF-Prune)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DST=${1:-$HERE/third_party/IF-Prune}
git clone https://github.com/snap-research/EVLM-IF-Prune.git "$DST"
git -C "$DST" checkout d17c2af
(cd "$DST" && patch -p1 < "$HERE/patches/lmms-eval.patch")
cp "$HERE"/models/*.py "$DST/lmms-eval/lmms_eval/models/"
cp -r "$HERE"/tasks/* "$DST/lmms-eval/lmms_eval/tasks/"   # RefCOCO with InternVL's grounding prompt, SEED-Bench images
echo "done; before running the scripts:"
echo "  export IFPRUNE_REPO=$DST PYTHONPATH=$DST/lmms-eval:\${PYTHONPATH:-}"
