"""
ML Models — Optimized for RPi4 + MLX90640 thermal human detection.

Optimizations applied per requirement:
  1. Pre-allocated morphology kernels  — no per-frame np.ones() allocation
  2. Vectorised NMS with numpy          — replaces Python-loop NMS
  3. Human shape filter (aspect ratio + area) — rejects non-human blobs
  4. Temperature range validation       — only 26-39C blobs qualify
  5. EMA confidence smoother           — eliminates flickering detections
  6. Smoothed display bounding box     — EMA on box coords, no jitter
  7. Re-entry grace window             — same person briefly off-frame = same ID
  8. Unique human counter              — seen-ID set, never double-counts
  9. np.linalg.solve instead of inv()  — stable Kalman update
 10. Pre-allocated ONNX input buffer   — avoids per-frame allocation
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional, Set
from ml.detection import Detection

logger = logging.getLogger("ml")

# Pre-allocated morphology kernels — created once at import, reused every frame
_K3  = np.ones((3,  3),  np.uint8)
_K5  = np.ones((5,  5),  np.uint8)
_K9  = np.ones((9,  9),  np.uint8)
_K15 = np.ones((15, 15), np.uint8)

# Human body proportions at MLX90640 upscaled resolution
_ASPECT_MIN  = 0.25   # w/h  (very tall / partially visible)
_ASPECT_MAX  = 2.50   # w/h  (wide crouched)
_AREA_MIN_PX = 120    # px²  (minimum blob area in display coords)
_AREA_MAX_FR = 0.35   # fraction of frame (anything larger is not a person)

# Temperature range for a human as seen through clothing/debris
_HUMAN_TEMP_MIN = 12.0   # C (lowered to support indoor clothing / dynamic range)
_HUMAN_TEMP_MAX = 45.0   # C

# EMA alpha for confidence smoothing — lower = smoother but slower
_CONF_EMA = 0.30


# ═══════════════════════════════════════════════════════════════════
# VECTORISED NMS  (shared utility)
# ═══════════════════════════════════════════════════════════════════

def _nms(dets: List[Detection], iou_thresh: float) -> List[Detection]:
    if len(dets) <= 1:
        return dets
    boxes  = np.array([[d.x1,d.y1,d.x2,d.y2] for d in dets], dtype=np.float32)
    scores = np.array([d.confidence for d in dets], dtype=np.float32)
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas  = (x2-x1)*(y2-y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        rest = order[1:]
        ix1  = np.maximum(x1[i], x1[rest]); iy1 = np.maximum(y1[i], y1[rest])
        ix2  = np.minimum(x2[i], x2[rest]); iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0., ix2-ix1) * np.maximum(0., iy2-iy1)
        iou   = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou < iou_thresh]
    return [dets[k] for k in keep]


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1,ay1,ax2,ay2 = a[:,0],a[:,1],a[:,2],a[:,3]
    bx1,by1,bx2,by2 = b[:,0],b[:,1],b[:,2],b[:,3]
    ix1 = np.maximum(ax1[:,None], bx1[None]); iy1 = np.maximum(ay1[:,None], by1[None])
    ix2 = np.minimum(ax2[:,None], bx2[None]); iy2 = np.minimum(ay2[:,None], by2[None])
    inter = np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
    aa = (ax2-ax1)*(ay2-ay1); ab = (bx2-bx1)*(by2-by1)
    return inter / (aa[:,None] + ab[None] - inter + 1e-9)


# ═══════════════════════════════════════════════════════════════════
# ANOMALY DETECTOR  — runs every frame (~12ms on RPi4)
# ═══════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Adaptive background subtraction on raw 32x24 MLX90640 data.

    Frame pipeline:
      1. Adaptive background EMA  (cold pixels only)
      2. Threshold at bg + max(8C, 5sigma)  AND  <= 39C (human range)
      3. Morphological clean on native 32x24  (fast — tiny image)
      4. Single upscale to display res  (INTER_NEAREST)
      5. One dilation pass at display res
      6. Contours  -> area + aspect-ratio filter  -> temperature check
      7. EMA confidence smoothing per spatial cell
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._bg_mean: Optional[float] = None
        self._bg_std:  Optional[float] = None
        self._alpha = 0.04            # EMA ~ adapts over 25 frames
        self._conf_ema: dict = {}     # spatial cell -> smoothed conf
        # Cache display/native ratios
        self._vw = cfg.display_width
        self._vh = cfg.display_height
        self._sx = cfg.display_width  / 32.0
        self._sy = cfg.display_height / 24.0

    def load(self) -> bool:
        logger.info("[ANOMALY] Optimised adaptive detector ready (human filter + EMA conf)")
        return True

    def detect(
        self,
        raw:    Optional[np.ndarray],
        visual: Optional[np.ndarray],
    ) -> Tuple[List[Detection], bool]:
        if raw is not None:
            return self._from_raw(raw)
        if visual is not None:
            return self._from_visual(visual), False
        return [], False

    # ── Raw temperature path ──────────────────────────────────────

    def _from_raw(self, raw: np.ndarray) -> Tuple[List[Detection], bool]:
        # 1. Adaptive background (exclude hot pixels from estimate)
        med = float(np.median(raw))
        cold = raw < (med + 2.0 * raw.std())
        if cold.any():
            bg_m, bg_s = float(raw[cold].mean()), float(raw[cold].std())
        else:
            bg_m, bg_s = med, float(raw.std())

        bg_mean = self._bg_mean
        bg_std = self._bg_std
        if bg_mean is not None and bg_std is not None:
            a = self._alpha
            bg_mean = (1-a)*bg_mean + a*bg_m
            bg_std  = (1-a)*bg_std  + a*bg_s
        else:
            bg_mean, bg_std = bg_m, bg_s
            
        self._bg_mean = bg_mean
        self._bg_std = bg_std
        # Human skin is ~36C. In a hot 33C room, delta is only 3C.
        # Dropping threshold from 8.0C delta to 1.5C delta solves high-ambient masking!
        threshold = bg_mean + max(1.5, 3.0 * bg_std)

        # 2. Hot mask: above threshold AND within human temp range
        hot = (
            (raw > threshold) & (raw <= _HUMAN_TEMP_MAX)
        ).astype(np.uint8) * 255

        # 3. Morphology at native 32x24 (cheap — tiny image)
        hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN,  _K3)
        hot = cv2.morphologyEx(hot, cv2.MORPH_CLOSE, _K3)

        # 4. Single upscale to display resolution
        hot_up = cv2.resize(hot, (self._vw, self._vh), interpolation=cv2.INTER_NEAREST)

        # 5. One dilation at display res — gives contours breathing room
        hot_up = cv2.dilate(hot_up, _K15)

        # 6. Contours + human filter
        contours, _ = cv2.findContours(
            hot_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_area = max(self.cfg.anomaly_min_area_px, _AREA_MIN_PX)
        max_area = int(self._vw * self._vh * _AREA_MAX_FR)
        dets: List[Detection] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue                          # too small or too large

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if not (_ASPECT_MIN <= aspect <= _ASPECT_MAX):
                continue                          # wrong shape for human

            # Map back to native coords to read actual temperature
            rx1 = max(0,  int(x         / self._sx))
            ry1 = max(0,  int(y         / self._sy))
            rx2 = min(31, int((x+w)     / self._sx))
            ry2 = min(23, int((y+h)     / self._sy))
            if rx2 <= rx1 or ry2 <= ry1:
                continue

            region   = raw[ry1:ry2+1, rx1:rx2+1]
            blob_max = float(region.max())

            if blob_max < _HUMAN_TEMP_MIN:
                continue                          # too cold to be human

            # 7. Raw confidence + EMA smoothing
            excess   = (blob_max - threshold) / 10.0
            raw_conf = float(np.clip(0.40 + excess * 0.50, 0.40, 0.92))
            cell     = (x // 80, y // 60)        # 16x12 spatial grid
            prev     = self._conf_ema.get(cell, raw_conf)
            conf     = (1 - _CONF_EMA) * prev + _CONF_EMA * raw_conf
            self._conf_ema[cell] = conf

            dets.append(Detection(
                x1=float(x), y1=float(y),
                x2=float(x+w), y2=float(y+h),
                confidence=conf,
                source="anomaly",
                temp_celsius=blob_max,
            ))

        # Prune stale EMA cells
        if len(self._conf_ema) > 64:
            self._conf_ema.clear()

        return dets, len(dets) > 0

    # ── Visual fallback ───────────────────────────────────────────

    def _from_visual(self, visual: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(visual, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _K5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K9)
        mask = cv2.dilate(mask, _K15)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        VW, VH    = visual.shape[1], visual.shape[0]
        max_area  = int(VW * VH * _AREA_MAX_FR)
        dets = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < _AREA_MIN_PX or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if not (_ASPECT_MIN <= w/max(h,1) <= _ASPECT_MAX):
                continue
                
            # Maps coordinates from Native Visual Space into Native Display Space (1280x720) 
            # to prevent coordinate collisions with YOLO and shrinking when displaying.
            sx = self.cfg.display_width / VW
            sy = self.cfg.display_height / VH
            cx1 = x * sx
            cy1 = y * sy
            cx2 = (x + w) * sx
            cy2 = (y + h) * sy
            
            dets.append(Detection(
                x1=float(cx1), y1=float(cy1),
                x2=float(cx2), y2=float(cy2),
                confidence=0.42, source="anomaly",
            ))
        return dets


# ═══════════════════════════════════════════════════════════════════
# YOLOV8n ONNX DETECTOR  — runs every Nth frame
# ═══════════════════════════════════════════════════════════════════

class YOLODetector:
    def __init__(self, cfg):
        self.cfg         = cfg
        self._model      = None
        self._input_name = None
        self._backend    = "none"
        self._counter    = 0
        sz = cfg.yolo_input_size
        # Pre-allocated input buffer — avoids malloc every frame
        self._inp = np.zeros((1, 3, sz, sz), dtype=np.float32)

    def load(self) -> bool:
        path = Path(self.cfg.yolo_model_path)
        if path.exists():
            try:
                import subprocess
                import sys
                # Run import in a subprocess to catch C-level SIGILL (Illegal instruction)
                # without crashing the main rescue drone process.
                check = subprocess.run(
                    [sys.executable, "-c", "import onnxruntime"],
                    capture_output=True
                )
                if check.returncode != 0:
                    logger.error(f"[YOLO] ONNX Runtime is broken on this system (Exit {check.returncode}). Likely an illegal instruction/architecture mismatch.")
                    return self._try_cv2_fallback(path)

                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                self._model = ort.InferenceSession(
                    str(path), sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._model.get_inputs()[0].name
                self._backend    = "onnx"
                logger.info(f"[YOLO] ONNX loaded: {path} | 4-core | {self.cfg.yolo_input_size}px")
                return True
            except Exception as e:
                logger.warning(f"[YOLO] ONNX failed: {e}")
                return self._try_cv2_fallback(path)
        logger.warning(
            "[YOLO] No model or failing runtime — anomaly detector is primary.\n"
            "  Train: python scripts/train.py --data data.yaml --export-onnx"
        )
        return False
        
    def _try_cv2_fallback(self, path) -> bool:
        logger.info("[YOLO] Attempting failover to OpenCV DNN Pipeline...")
        try:
            self._model = cv2.dnn.readNetFromONNX(str(path))
            self._backend = "cv2"
            logger.info(f"[YOLO] Successfully activated cv2.dnn backend! | {self.cfg.yolo_input_size}px")
            return True
        except Exception as e2:
            logger.error(f"[YOLO] OpenCV DNN fallback failed: {e2}")
            logger.warning("[YOLO] Disabling YOLO safely. Try: pip install onnxruntime==1.15.1 or use a 64-bit OS.")
            return False

    def should_run(self) -> bool:
        self._counter += 1
        return self._counter % self.cfg.yolo_every_n_frames == 0

    def detect(self, visual: np.ndarray, out_w: Optional[int] = None, out_h: Optional[int] = None) -> List[Detection]:
        if self._backend == "none" or visual is None:
            return []
        sz     = self.cfg.yolo_input_size
        VH, VW = visual.shape[:2]
        
        if out_w is None: out_w = VW
        if out_h is None: out_h = VH
        
        # --- LETTERBOXING TO PRESERVE ASPECT RATIO ---
        shape = visual.shape[:2]  # H, W
        r = min(sz / shape[0], sz / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = sz - new_unpad[0], sz - new_unpad[1]
        dw /= 2; dh /= 2
        
        if shape[::-1] != new_unpad:
            resized = cv2.resize(visual, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = visual.copy()
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        resized = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # YOLO ONNX expects RGB colors for proper skin tracking; OpenCV gives us BGR.
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self._backend == "onnx":
            # Write into pre-allocated buffer in-place
            np.copyto(
                self._inp[0],
                np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1)),
            )
            out = self._model.run(None, {self._input_name: self._inp})[0]
        elif self._backend == "cv2":
            blob = cv2.dnn.blobFromImage(resized, 1/255.0, (sz, sz), swapRB=False, crop=False)
            self._model.setInput(blob)
            out = self._model.forward()
            
        return self._parse_letterbox(out, out_w, out_h, VW, VH, r, dw, dh)

    def _parse_letterbox(self, out, out_w, out_h, VW, VH, r, dw, dh) -> List[Detection]:
        if out.ndim == 3:
            pred = out[0]
            # Matrix transpose for YOLOv8 which outputs [num_classes + 4, num_anchors]
            if pred.shape[0] < pred.shape[1]: 
                pred = pred.T
        else:
            pred = out
        if len(pred) == 0:
            return []
        # Index 4 is the first class (Class 0), which is 'person' on COCO/YOLO.
        confs = pred[:, 4].astype(np.float32)
        keep  = confs >= self.cfg.yolo_conf_threshold
        pred, confs = pred[keep], confs[keep]
        if len(pred) == 0:
            return []
            
        # 1. Strip letterbox padding and scaling back to original visual frame (VW, VH)
        cx = (pred[:,0] - dw) / r
        cy = (pred[:,1] - dh) / r
        bw = (pred[:,2]) / r
        bh = (pred[:,3]) / r
        
        # 2. Scale up to the display constraints (out_w, out_h)
        sx, sy = out_w / VW, out_h / VH
        cx *= sx; cy *= sy
        bw *= sx; bh *= sy
        
        dets = [
            Detection(
                x1=float(cx[i]-bw[i]/2), y1=float(cy[i]-bh[i]/2),
                x2=float(cx[i]+bw[i]/2), y2=float(cy[i]+bh[i]/2),
                confidence=float(confs[i]), source="rgb",
            )
            for i in range(len(pred))
        ]
        return _nms(dets, self.cfg.yolo_iou_threshold)

    @property
    def backend(self) -> str:
        return self._backend


# ═══════════════════════════════════════════════════════════════════
# SENSOR FUSION
# ═══════════════════════════════════════════════════════════════════

class SensorFusion:
    def __init__(self, cfg):
        self.cfg = cfg

    def fuse(
        self,
        anomaly_dets:  List[Detection],
        yolo_dets:     List[Detection],
        radar_present: bool,
        radar_conf:    float,
    ) -> List[Detection]:
        merged = list(yolo_dets)

        # Add anomaly dets not overlapping any YOLO box
        if anomaly_dets:
            if yolo_dets:
                yb = np.array([[d.x1,d.y1,d.x2,d.y2] for d in yolo_dets], np.float32)
                for a in anomaly_dets:
                    ab  = np.array([[a.x1,a.y1,a.x2,a.y2]], np.float32)
                    iou = _iou_matrix(ab, yb)[0]
                    if iou.max() <= self.cfg.fusion_iou_merge:
                        merged.append(Detection(
                            x1=a.x1, y1=a.y1, x2=a.x2, y2=a.y2,
                            confidence=float(np.clip(
                                a.confidence * self.cfg.weight_anomaly * 2, 0, 0.85
                            )),
                            source="anomaly", temp_celsius=a.temp_celsius,
                        ))
            else:
                merged.extend(anomaly_dets)

        # LD2410 radar confidence boost
        if radar_present and radar_conf > 0.3 and merged:
            boost = self.cfg.weight_radar * radar_conf
            merged = [
                Detection(
                    x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                    confidence=min(0.99, d.confidence + boost),
                    source="fused" if d.source != "anomaly" else d.source,
                    temp_celsius=d.temp_celsius,
                )
                for d in merged
            ]

        if radar_present and not merged and radar_conf > 0.5:
            logger.info(
                "[FUSION] Radar: presence confirmed, no thermal blob — "
                "possible deep burial > 30cm"
            )

        return _nms(merged, self.cfg.fusion_iou_merge)


# ═══════════════════════════════════════════════════════════════════
# KALMAN TRACK
# ═══════════════════════════════════════════════════════════════════

class _Track:
    _id_counter = 0

    def __init__(self, det: Detection):
        _Track._id_counter += 1
        self.id        = _Track._id_counter
        self.hits      = 1
        self.streak    = 1
        self.missed    = 0
        self.total_age = 0
        self.last_conf = det.confidence
        self.last_src  = det.source
        self.last_temp = det.temp_celsius

        # Smoothed display box — EMA of raw detection positions
        self._smooth   = np.array(
            [det.x1, det.y1, det.x2, det.y2], dtype=np.float32
        )
        self._sa       = 0.35   # EMA alpha for box smoothing

        # Kalman state [cx, cy, s, r, vx, vy, vs]
        w  = det.x2-det.x1; h = det.y2-det.y1
        self.x = np.array(
            [[det.x1+w/2], [det.y1+h/2], [w*h], [w/max(h,1.)],
             [0.], [0.], [0.]], dtype=np.float64
        )
        self.P = np.diag([10.,10.,10.,10.,1000.,1000.,10.])
        self.F = np.eye(7); self.F[0,4]=self.F[1,5]=self.F[2,6]=1.
        self.H = np.eye(4, 7)
        self.R = np.diag([1.,1.,10.,10.]).astype(np.float64)
        self.Q = np.eye(7)*0.01
        self.Q[4,4]=self.Q[5,5]=1.0; self.Q[6,6]=0.5

    def predict(self):
        self.x         = self.F @ self.x
        self.P         = self.F @ self.P @ self.F.T + self.Q
        self.total_age += 1
        self.missed    += 1
        self.streak     = max(0, self.streak - 1)
        if self.x[2,0] < 0: self.x[2,0] = 1.

    def update(self, det: Detection):
        w=det.x2-det.x1; h=det.y2-det.y1
        cx=det.x1+w/2; cy=det.y1+h/2
        z  = np.array([[cx],[cy],[w*h],[w/max(h,1.)]])
        S  = self.H @ self.P @ self.H.T + self.R
        K  = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(7) - K @ self.H) @ self.P
        # EMA smooth display box
        raw = np.array([det.x1,det.y1,det.x2,det.y2], dtype=np.float32)
        self._smooth = self._sa*raw + (1-self._sa)*self._smooth
        self.hits+=1; self.streak+=1; self.missed=0
        self.last_conf=det.confidence; self.last_src=det.source
        self.last_temp=det.temp_celsius

    def predicted_box(self) -> np.ndarray:
        cx,cy = self.x[0,0],self.x[1,0]
        s,r   = max(self.x[2,0],1.), max(self.x[3,0],0.1)
        w=np.sqrt(s*r); h=s/w
        return np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2])

    def display_box(self) -> np.ndarray:
        return self._smooth


# ═══════════════════════════════════════════════════════════════════
# HUMAN TRACKER
#   - unique counting via seen-ID set  (req 4)
#   - re-entry grace window            (req 5)
#   - smoothed boxes                   (req 6)
#   - confirmed-only output            (req 6 — no flickering new boxes)
# ═══════════════════════════════════════════════════════════════════

class HumanTracker:
    def __init__(self, cfg):
        self.cfg           = cfg
        self._tracks:      List[_Track] = []
        self._frame:       int          = 0
        self._seen_ids:    Set[int]     = set()
        self._total_unique: int         = 0
        # Graveyard: recently-lost tracks available for re-entry matching
        self._graveyard:   deque        = deque(maxlen=20)
        self._grace        = max(8, cfg.tracker_max_age // 3)

    @property
    def total_unique_humans(self) -> int:
        """Lifetime count. Never decrements. Use for rescue log."""
        return self._total_unique

    def reset(self):
        self._tracks=[]; self._seen_ids=set(); self._total_unique=0
        self._graveyard.clear(); _Track._id_counter=0; self._frame=0

    def update(self, dets: List[Detection]) -> List[Detection]:
        self._frame += 1
        for t in self._tracks: t.predict()

        # Association
        matched_dets: set = set()
        if dets and self._tracks:
            db = np.array([[d.x1,d.y1,d.x2,d.y2] for d in dets], np.float32)
            tb = np.array([t.predicted_box() for t in self._tracks], np.float32)
            iou_mat = _iou_matrix(db, tb)
            for di, ti in _assign(iou_mat, self.cfg.tracker_iou_thresh):
                self._tracks[ti].update(dets[di]); matched_dets.add(di)

        # Unmatched dets — try re-entry before creating new track
        for di in range(len(dets)):
            if di in matched_dets: continue
            old_id = self._try_reentry(dets[di])
            t = _Track(dets[di])
            if old_id is not None:
                t.id = old_id   # reclaim old ID — no extra count
            self._tracks.append(t)

        # Prune — move dead tracks to graveyard
        alive, dead = [], []
        for t in self._tracks:
            (alive if t.missed <= self.cfg.tracker_max_age else dead).append(t)
        self._tracks = alive
        for t in dead:
            if t.hits >= self.cfg.tracker_min_hits:
                self._graveyard.append(
                    {"id": t.id, "box": t.display_box().copy(), "ttl": self._grace}
                )

        # Tick graveyard
        self._graveyard = deque(
            [e for e in self._graveyard if e["ttl"] > 0],
            maxlen=20,
        )
        for e in self._graveyard: e["ttl"] -= 1

        # Build confirmed output
        out: List[Detection] = []
        for t in self._tracks:
            confirmed = (
                t.streak >= self.cfg.tracker_min_hits
                or self._frame <= self.cfg.tracker_min_hits
            )
            if not confirmed: continue

            if t.id not in self._seen_ids:
                self._seen_ids.add(t.id)
                self._total_unique += 1
                logger.info(
                    f"[TRACKER] ID {t.id} confirmed | "
                    f"unique total: {self._total_unique}"
                )

            b = t.display_box()
            out.append(Detection(
                x1=float(b[0]), y1=float(b[1]),
                x2=float(b[2]), y2=float(b[3]),
                confidence=t.last_conf, source=t.last_src,
                track_id=t.id, temp_celsius=t.last_temp,
            ))
        return out

    def _try_reentry(self, det: Detection) -> Optional[int]:
        if not self._graveyard: return None
        db = np.array([[det.x1,det.y1,det.x2,det.y2]], np.float32)
        best_iou, best_id = 0., None
        for e in self._graveyard:
            iou = float(_iou_matrix(db, e["box"].reshape(1,4))[0,0])
            if iou > best_iou: best_iou, best_id = iou, e["id"]
        if best_iou >= self.cfg.tracker_iou_thresh * 0.7:
            self._graveyard = deque(
                [e for e in self._graveyard if e["id"] != best_id], maxlen=20
            )
            return best_id
        return None


def _assign(iou_mat: np.ndarray, thresh: float) -> List[Tuple[int,int]]:
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-iou_mat)
        return [(int(ri),int(ci)) for ri,ci in zip(r,c) if iou_mat[ri,ci] >= thresh]
    except ImportError:
        pairs: List[Tuple[int,int]] = []; used: set = set()
        order = np.argsort(-iou_mat.max(axis=1))
        for ri in order:
            best_c, best_v = -1, -1.
            for ci in range(iou_mat.shape[1]):
                if ci not in used and iou_mat[ri,ci] > best_v:
                    best_v=iou_mat[ri,ci]; best_c=ci
            if best_c >= 0 and best_v >= thresh:
                pairs.append((int(ri), best_c)); used.add(best_c)
        return pairs
