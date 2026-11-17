"""Benchmarks, budgets and hyperparameters of PRUDENT as used in the paper."""

# The 18 benchmarks of the main tables (lmms-eval task names), by group.
TEXT_RICH = ["chartqa", "docvqa_val", "infovqa_val", "textvqa_val", "ocrbench", "websrc_val_sub10"]
GENERAL = ["gqa", "vqav2_val_sub", "ok_vqa_val2014", "mmbench_en_dev", "mmbench_cn_dev", "mme", "pope",
           "seedbench_img_dis", "ai2d", "scienceqa_img"]
GROUNDING = ["refcoco_bbox_rec_testA_dis", "refcoco_bbox_rec_testB_dis"]
BENCHMARKS = TEXT_RICH + GENERAL + GROUNDING

# Answers that are option letters: agreement compares the letters, confidence is the first-token probability.
LETTER_TASKS = {"mmbench_en_dev", "mmbench_cn_dev", "mmstar", "realworldqa", "ai2d", "scienceqa_img",
                "seedbench_img_dis", "mmmu_val_single_dis"}
# Multiple-choice and yes-or-no questions: the guide ranks by the learned estimate alone (Appendix D).
CLOSED_FORM = LETTER_TASKS | {"mme", "pope"}

# Rankings of the guide: the average of the ranks of its attention (SGL) and of the learned estimate (IF-Prune),
# or the learned estimate alone on closed-form questions.
RANK_AVG, LEARNED = "rankavg-sgp", "single-ifp32"


def ranking(task):
    return LEARNED if task in CLOSED_FORM else RANK_AVG


# Average budget K -> (first pass b, re-answer b2). b = K/2 and b2 = 2K, except at 2.5% (Appendix C).
PLAN = {0.025: (0.01, 0.05), 0.05: (0.025, 0.1), 0.1: (0.05, 0.2)}
TAU = 0.5   # the checker abstains when the least probable token of its answer is below this probability,
ELL = 10    # or when its answer is longer than this many words

# With full tokens, InternVL3-38B answers part of GQA and VQAv2 with a stock phrase (Appendix I). Its scores on these
# two leave out those questions, while PRUDENT still selects among all of them.
STOCK_PHRASE = "answering does not require reading text in the image"
STOCK_PHRASE_RUNS = {("InternVL3-38B", "gqa"), ("InternVL3-38B", "vqav2_val_sub")}


def quota(K, n_checked, b=None, b2=None):
    """Questions re-answered at b2, chosen among the checked ones, so that the average budget stays K."""
    b0, b20 = PLAN.get(K, (K / 2, 2 * K))
    b, b2 = b or b0, b2 or b20
    return int((K - b) / b2 * n_checked)
