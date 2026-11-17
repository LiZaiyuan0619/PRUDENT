"""When two answers count as equal (Appendix B): the disagreement of the checker with the first answer."""
import re

import numpy as np

from .config import LETTER_TASKS


def norm(a):
    return " ".join(str(a).strip().lower().rstrip(".").split())


_LETTER = re.compile(r"^\(?([a-fA-F])(?:[\).:,]|\s|$)")


def option_letter(s):
    """'B', 'b.', '(b) cat', 'B: cat' -> 'B'; an answer that does not start with an option letter -> None."""
    m = _LETTER.match(str(s).strip())
    return m.group(1).upper() if m else None


_QUOTED = re.compile(r'[\"“”]\s*([^\"“”]+?)\s*(?:[\"“”]|$)')  # double quotes only: apostrophes are not quotes


def core(s):
    """The content of an answer: the text inside its first double quotes when it has any (OCR answers often wrap the
    word in a sentence), else the whole answer; normalized, trailing punctuation dropped."""
    s = str(s)
    m = _QUOTED.search(s)
    return norm((m.group(1) if m else s).strip(" .,;:!?"))


_BOX = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def parse_box(s):
    """First [x1, y1, x2, y2] in an answer; InternVL's 0-1000 boxes rescaled to 0-1 (a box already in 0-1 is kept).
    The same as tasks/refcoco/utils_dis.py, so that the tables need no lmms-eval."""
    m = _BOX.search(str(s))
    if not m:
        return [0.0, 0.0, 0.0, 0.0]
    b = [float(m.group(i)) for i in range(1, 5)]
    return [v / 1000.0 for v in b] if max(b) > 1.0 else b


def compute_iou(box1, box2):
    """Intersection over union of two [x_min, y_min, x_max, y_max] boxes, as lmms-eval's RefCOCO scorer computes it."""
    x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1])
    x_right, y_bottom = min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / (box1_area + box2_area - intersection_area)

def agree(task, checker_ans, first_ans):
    """Per question, whether the checker's answer matches the first answer under the answer format of the task.
    Option letters compare the letters (falling back to the normalized strings when either has no letter), boxes
    of RefCOCO match at IoU >= 0.5, open answers compare their cores."""
    if task.startswith("refcoco"):
        return np.array([compute_iou(parse_box(a), parse_box(b)) >= 0.5 for a, b in zip(checker_ans, first_ans)])
    if task in LETTER_TASKS:
        a, b = checker_ans.map(option_letter), first_ans.map(option_letter)
        both = a.notna() & b.notna()
        return np.where(both, a == b, checker_ans.map(norm) == first_ans.map(norm))
    return (checker_ans.map(core) == first_ans.map(core)).values
