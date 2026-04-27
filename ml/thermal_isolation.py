"""
Thermal Human Isolation — ml/thermal_isolation.py

Separates human heat signatures from cold background at the
raw 32x24 pixel level, then renders them as clean silhouettes
on the upscaled display frame.

Output modes:
  "highlight"  — original thermal colormap + bright human overlay
  "silhouette" — black background, glowing human blobs only
  "contour"    — original + orange outline around each human

No ML model needed. At 32x24 resolution with a 30C+ thermal
contrast between human body and snow/debris, adaptive thresholding
is more reliable than any model trained at this resolution.
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple
from ml.detection import Detection


# ── Kernel bank ───────────────────────────────────────────────────
_K3  = np.ones((3, 3),  np.uint8)
_K5  = np.ones((5, 5),  np.uint8)

# Human temperature range (MLX90640, through clothing/debris)
HUMAN_TEMP_MIN = 26.0
HUMAN_TEMP_MAX = 39.0

# Human shape constraints at native 32x24
ASPECT_MIN = 0.20    # w/h
ASPECT_MAX = 2.80    # w/h
AREA_MIN_NATIVE = 2  # px² at 32x24 (very small = partially visible)
AREA_MAX_NATIVE = 60 # px² at 32x24 (too large = not one person)


class ThermalIsolator:
    """
    Takes raw MLX90640 temperature data and produces:
      1. A clean binary human mask  (32x24 → upscaled)
      2. An overlay frame showing ONLY human blobs highlighted
         against a darkened/neutralised background
      3. Detection bounding boxes in display coordinates
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._bg_mean: Optional[float] = None
        self._bg_std:  Optional[float] = None
        self._alpha = 0.04            # slow background adaptation
        self._vw = cfg.display_width
        self._vh = cfg.display_height
        self._sx = cfg.display_width  / 32.0
        self._sy = cfg.display_height / 24.0
        # Running per-blob temperature history for stability
        self._temp_history: dict = {}   # cell -> deque of temps

    def process(
        self,
        raw:    np.ndarray,             # (24, 32) float32 °C
        visual: np.ndarray,             # (VH, VW, 3) BGR colormapped
        mode: str = "highlight",        # "highlight" | "silhouette" | "contour"
    ) -> Tuple[np.ndarray, List[Detection], np.ndarray]:
        """
        Returns:
          isolated_frame : (VH, VW, 3)  — output image with humans highlighted
          detections     : list of Detection in display coords
          human_mask_up  : (VH, VW)     — binary uint8 mask of human regions
        """
        # Step 1: adaptive background
        threshold = self._update_background(raw)

        # Step 2: human mask at native 32x24
        human_mask_native = self._build_native_mask(raw, threshold)

        # Step 3: upscale mask to display resolution
        human_mask_up = cv2.resize(
            human_mask_native, (self._vw, self._vh),
            interpolation=cv2.INTER_NEAREST,
        )
        # Smooth the upscaled mask to remove blockiness from 32x24 pixels
        human_mask_up = cv2.morphologyEx(human_mask_up, cv2.MORPH_CLOSE, _K5)
        human_mask_up = cv2.GaussianBlur(human_mask_up, (5, 5), 1)
        _, human_mask_up = cv2.threshold(human_mask_up, 127, 255, cv2.THRESH_BINARY)

        # Step 4: extract detections from mask
        detections = self._mask_to_detections(raw, human_mask_up, threshold)

        # Step 5: render output frame
        isolated = self._render(visual, human_mask_up, detections, mode)

        return isolated, detections, human_mask_up

    # ── Background model ──────────────────────────────────────────

    def _update_background(self, raw: np.ndarray) -> float:
        """
        Adaptive background temperature using EMA on cold pixels only.
        Cold pixels = background (snow, debris, rock).
        Hot pixels  = potential humans — excluded from background estimate.
        """
        med   = float(np.median(raw))
        sigma = float(raw.std())
        cold  = raw < (med + 2.0 * sigma)

        if cold.any():
            bg_m = float(raw[cold].mean())
            bg_s = float(raw[cold].std())
        else:
            bg_m, bg_s = med, sigma

        if self._bg_mean is None:
            self._bg_mean, self._bg_std = bg_m, bg_s
        else:
            a = self._alpha
            self._bg_mean = (1 - a) * self._bg_mean + a * bg_m
            self._bg_std  = (1 - a) * self._bg_std  + a * bg_s

        # Threshold: background + max(8°C, 5σ)
        return self._bg_mean + max(8.0, 5.0 * self._bg_std)

    # ── Native mask ───────────────────────────────────────────────

    def _build_native_mask(self, raw: np.ndarray, threshold: float) -> np.ndarray:
        """
        Build binary human mask at native 32x24 resolution.
        Two conditions must both be true per pixel:
          1. Temperature above adaptive threshold (hot relative to background)
          2. Temperature within human range (26-39°C)
        """
        mask = (
            (raw > threshold)          &
            (raw >= HUMAN_TEMP_MIN)    &
            (raw <= HUMAN_TEMP_MAX)
        ).astype(np.uint8) * 255

        # Morphological clean at native resolution (cheap — 32x24 image)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _K3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K3)
        return mask

    # ── Detections from mask ──────────────────────────────────────

    def _mask_to_detections(
        self,
        raw:     np.ndarray,
        mask_up: np.ndarray,
        threshold: float,
    ) -> List[Detection]:
        """
        Find contours in the upscaled mask, apply human shape filters,
        and return Detection objects in display coordinates.
        """
        contours, _ = cv2.findContours(
            mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        max_area = int(self._vw * self._vh * 0.35)
        dets: List[Detection] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80 or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
                continue

            # Map back to native to read actual temperature
            rx1 = max(0,  int(x     / self._sx))
            ry1 = max(0,  int(y     / self._sy))
            rx2 = min(31, int((x+w) / self._sx))
            ry2 = min(23, int((y+h) / self._sy))
            if rx2 <= rx1 or ry2 <= ry1:
                continue

            region   = raw[ry1:ry2+1, rx1:rx2+1]
            blob_max = float(region.max())
            blob_avg = float(region.mean())

            if blob_max < HUMAN_TEMP_MIN:
                continue

            # Confidence based on thermal contrast above background
            excess = (blob_max - threshold) / 10.0
            conf   = float(np.clip(0.45 + excess * 0.45, 0.45, 0.95))

            dets.append(Detection(
                x1=float(x), y1=float(y),
                x2=float(x+w), y2=float(y+h),
                confidence=conf,
                source="thermal",
                temp_celsius=blob_max,
            ))

        return dets

    # ── Render ────────────────────────────────────────────────────

    def _render(
        self,
        visual:    np.ndarray,
        mask_up:   np.ndarray,
        dets:      List[Detection],
        mode:      str,
    ) -> np.ndarray:
        if mode == "silhouette":
            return self._render_silhouette(mask_up)
        elif mode == "contour":
            return self._render_contour(visual, mask_up, dets)
        else:
            return self._render_highlight(visual, mask_up, dets)

    def _render_highlight(
        self,
        visual:  np.ndarray,
        mask_up: np.ndarray,
        dets:    List[Detection],
    ) -> np.ndarray:
        """
        Original thermal colormap with two treatments:
          - Background: darkened (dim non-human regions)
          - Human blobs: brightened + orange-white glow overlay
        """
        out = visual.copy()

        # Dim the background (non-human regions)
        bg_mask = cv2.bitwise_not(mask_up)
        bg_region = cv2.bitwise_and(out, out, mask=bg_mask)
        # Darken background by 50%
        bg_dark = (bg_region * 0.45).astype(np.uint8)
        np.copyto(out, bg_dark, where=bg_mask[:,:,np.newaxis].astype(bool))

        # Human region: boost brightness + add warm glow
        human_region = cv2.bitwise_and(out, out, mask=mask_up)
        # Add orange-white warmth to the human blob
        warm_overlay = np.zeros_like(out)
        warm_overlay[:, :] = (20, 60, 30)   # BGR: slight orange boost
        warm_add = cv2.bitwise_and(warm_overlay, warm_overlay, mask=mask_up)
        human_boosted = cv2.add(human_region, warm_add)
        np.copyto(out, human_boosted, where=mask_up[:,:,np.newaxis].astype(bool))

        return out

    def _render_silhouette(self, mask_up: np.ndarray) -> np.ndarray:
        """
        Pure black background. Human blobs as a glowing orange shape.
        Maximum contrast — humans glow on pure black.
        """
        out = np.zeros((self._vh, self._vw, 3), dtype=np.uint8)

        # Create a gradient glow — brighter at center, fades at edges
        dist = cv2.distanceTransform(mask_up, cv2.DIST_L2, 5)
        if dist.max() > 0:
            dist_norm = (dist / dist.max() * 255).astype(np.uint8)
        else:
            dist_norm = mask_up.copy()

        # Orange glow: R=255, G=varies with dist, B=0
        out[:, :, 2] = dist_norm          # R channel — orange intensity
        out[:, :, 1] = (dist_norm * 0.4).astype(np.uint8)  # G — slight warmth
        out[:, :, 0] = 0                   # B — none

        return out

    def _render_contour(
        self,
        visual:  np.ndarray,
        mask_up: np.ndarray,
        dets:    List[Detection],
    ) -> np.ndarray:
        """
        Original thermal colormap + orange contour outline around humans.
        Subtle — shows context but clearly marks human regions.
        """
        out = visual.copy()
        contours, _ = cv2.findContours(
            mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Draw filled semi-transparent overlay on human regions
        overlay = out.copy()
        cv2.fillPoly(overlay, contours, (0, 80, 200))  # BGR: dark orange fill
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
        # Draw bright contour outline
        cv2.drawContours(out, contours, -1, (0, 140, 255), 2)
        return out
