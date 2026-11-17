#!/usr/bin/env python3
"""lmms-eval's command line with the existing methods of this folder registered in internvl_fuse.
FUSE_SPEC = vzip | tome (fuse_vz.py) or cdp | mmtok | zoo (fuse_new.py) select the tokens as those methods do, and
single:<map> runs as before (IF-Prune, SGL, FastV). With LARGE_SPLIT_CKPT set to the large checkpoint, the large VLM
is split over the visible GPUs (fuse_mg.py, for InternVL3-78B). scripts/run_baseline.sh calls this file."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("LARGE_SPLIT_CKPT"):
    import fuse_mg  # noqa: E402,F401
import fuse_new  # noqa: E402,F401  (imports fuse_vz; both patch internvl_fuse in-process)
from lmms_eval.__main__ import cli_evaluate  # noqa: E402

if __name__ == "__main__":
    cli_evaluate()
