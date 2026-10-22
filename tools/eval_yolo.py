"""
Person-detection accuracy and speed of a YOLO ONNX model, through the same
preprocessing / decoding / NMS the drone uses (ml.models.YOLODetector).

Labels are YOLO format (one .txt per image: "class cx cy w h", normalised),
e.g. coco128, VisDrone or SARD converted for Ultralytics.

usage:
    ~/venv/bin/python tools/eval_yolo.py --images DIR --labels DIR [--model PATH] [--cls 0]
                                          [--conf 0.35] [--threads 1] [--every 1 --offset 0]
--every/--offset evaluate every Nth image (e.g. to leave out calibration images).
"""
import argparse, glob, os, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import logging
logging.disable(logging.WARNING)
import cv2
import numpy as np
from config import Config
from ml.models import YOLODetector, _iou_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model", default=None, help="default: config.py yolo_model_path")
    ap.add_argument("--cls", type=int, default=0, help="label class id of 'person'")
    ap.add_argument("--conf", type=float, default=None, help="default: config.py yolo_conf_threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU for a detection to count as a hit")
    ap.add_argument("--threads", type=int, default=None, help="default: config.py yolo_intra_threads")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()

    cfg = Config()
    if a.model: cfg.yolo_model_path = a.model
    if a.conf is not None: cfg.yolo_conf_threshold = a.conf
    if a.threads: cfg.yolo_intra_threads = a.threads
    det = YOLODetector(cfg)
    if not det.load():
        sys.exit(f"could not load {cfg.yolo_model_path}")

    paths = sorted(p for p in glob.glob(os.path.join(a.images, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
    paths = paths[a.offset::a.every]
    tp = fp = fn = 0
    ms = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = []
        lab = os.path.join(a.labels, os.path.splitext(os.path.basename(p))[0] + ".txt")
        if os.path.exists(lab):
            for line in open(lab):
                v = line.split()
                if len(v) >= 5 and int(float(v[0])) == a.cls:
                    cx, cy, bw, bh = (float(x) for x in v[1:5])
                    gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        t0 = time.perf_counter()
        dets = det.detect(img)
        ms.append((time.perf_counter() - t0) * 1000)
        dets.sort(key=lambda d: -d.confidence)
        used = set()
        if gt and dets:
            iou = _iou_matrix(np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], np.float32),
                              np.array(gt, np.float32))
            for i in range(len(dets)):
                cand = [(iou[i, j], j) for j in range(len(gt)) if j not in used and iou[i, j] >= a.iou]
                if cand:
                    used.add(max(cand)[1]); tp += 1
                else:
                    fp += 1
        else:
            fp += len(dets)
        fn += len(gt) - len(used)

    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    print(f"{os.path.basename(cfg.yolo_model_path)}: {len(paths)} images, {tp + fn} people | "
          f"recall {rec:.3f}  precision {prec:.3f}  F1 {2 * prec * rec / max(1e-9, prec + rec):.3f} "
          f"(TP {tp} FP {fp} FN {fn}) | detect() median {np.median(ms):.0f} ms "
          f"({cfg.yolo_intra_threads} thread{'s' if cfg.yolo_intra_threads > 1 else ''})")


if __name__ == "__main__":
    main()
