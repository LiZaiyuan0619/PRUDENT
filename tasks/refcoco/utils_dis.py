"""RefCOCO referring-expression comprehension with InternVL's own grounding prompt and box format
(<ref>..</ref> in, <box>[[x1, y1, x2, y2]]</box> out, coordinates 0-1000, as InternVL's
evaluate_grounding.py); the box is rescaled to 0-1 and scored with utils_rec's metrics (IoU, ACC@k,
center). Same rows as refcoco_bbox_rec_*."""
import re

from lmms_eval.tasks.refcoco.utils_rec import COCO_REC_METRICS

_BOX = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def refcoco_dis_doc_to_text(doc):
    return f"Please provide the bounding box coordinate of the region this sentence describes: <ref>{doc['answer']}</ref>"


def parse_box(s):
    """First [x1, y1, x2, y2] in the answer; InternVL's 0-1000 boxes rescaled to 0-1 (a box already in 0-1 is kept)."""
    m = _BOX.search(str(s))
    if not m:
        return [0.0, 0.0, 0.0, 0.0]
    b = [float(m.group(i)) for i in range(1, 5)]
    return [v / 1000.0 for v in b] if max(b) > 1.0 else b


def refcoco_dis_process_result(doc, result):
    pred = parse_box(result[0] if len(result) > 0 else "")
    data = {"answer": doc["answer"], "pred": pred, "ann_id": doc["question_id"], "bbox": doc["bbox"]}
    return {f"refcoco_{metric}": data for metric in COCO_REC_METRICS}
