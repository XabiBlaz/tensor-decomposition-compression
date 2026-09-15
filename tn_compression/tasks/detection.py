"""TorchVision image-list contract and standard COCO box evaluation."""

import contextlib
import io

import torch

from .vision import evaluation_mode


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def validate_predictions(predictions, image_count):
    if not isinstance(predictions, list) or len(predictions) != image_count:
        raise ValueError("Detection output must contain one prediction dictionary per image.")
    for prediction in predictions:
        if not {"boxes", "labels", "scores"} <= prediction.keys():
            raise ValueError("Detector output requires boxes, labels and scores.")
        boxes, labels, scores = (prediction[key] for key in ("boxes", "labels", "scores"))
        if boxes.shape != (len(labels), 4) or scores.shape != labels.shape:
            raise ValueError("Invalid detector box/label/score shapes.")
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
            raise ValueError("Non-finite detector predictions.")
        if ((boxes[:, 2:] - boxes[:, :2]) < 0).any():
            raise ValueError("Detector boxes must use ordered xyxy coordinates.")


@torch.no_grad()
def evaluate_detection(model, batches, coco, *, device="cpu", category_mapping=None):
    from pycocotools.cocoeval import COCOeval
    from pycocotools.coco import COCO
    rows, image_ids = [], []
    valid_categories = set(coco.getCatIds())
    with evaluation_mode(model):
        for images, targets in batches:
            predictions = model([image.to(device) for image in images])
            validate_predictions(predictions, len(images))
            for prediction, target in zip(predictions, targets):
                image_id = int(target["image_id"])
                image_ids.append(image_id)
                boxes = prediction["boxes"].detach().cpu()
                for box, label, score in zip(boxes, prediction["labels"].tolist(), prediction["scores"].tolist()):
                    category = category_mapping[label] if category_mapping is not None else label
                    if category not in valid_categories:
                        raise ValueError(f"Model label {label} has no COCO category mapping.")
                    x1, y1, x2, y2 = box.tolist()
                    rows.append({"image_id": image_id, "category_id": category,
                                 "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    if not image_ids or len(image_ids) != len(set(image_ids)):
        raise ValueError("COCO evaluation needs nonempty, unique image IDs.")
    with contextlib.redirect_stdout(io.StringIO()):
        if rows:
            predictions_coco = coco.loadRes(rows)
        else:
            predictions_coco = COCO()
            predictions_coco.dataset = {"images": coco.dataset["images"],
                                        "categories": coco.dataset["categories"], "annotations": []}
            predictions_coco.createIndex()
        evaluator = COCOeval(coco, predictions_coco, "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {"task": "detection", "ap": float(evaluator.stats[0]), "ap50": float(evaluator.stats[1]),
            "image_ids": sorted(image_ids), "examples": len(image_ids), "predictions": len(rows),
            "max_detections": list(evaluator.params.maxDets), "subset_only": True}
