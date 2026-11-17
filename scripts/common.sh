# Shared settings of the run scripts (sourced). Override any of them in the environment.
: "${IFPRUNE_REPO:?run setup.sh and export IFPRUNE_REPO}"
export PYTHONPATH=$IFPRUNE_REPO/lmms-eval:${PYTHONPATH:-}
export LMMS_EVAL_NO_API=1            # MMBench answers are matched offline; no API calls
OUT=$(realpath -m "${OUT:-runs}")    # where every stage writes (read by make_table.py)
IFP_1B=${IFP_1B:-ZachSun/internvl2_5_1b_pruning_kl_lora32}   # IF-Prune's released importance estimator
SMALL=${SMALL:-OpenGVLab/InternVL2_5-1B}                      # the pre-trained small VLM of SGL's ranking
CHECKER=${CHECKER:-OpenGVLab/InternVL3-2B}
LAYER=0                              # the large VLM prunes its visual tokens before its first LLM layer
frac() { python -c "print(f'{float(\"$1\"):g}')"; }
