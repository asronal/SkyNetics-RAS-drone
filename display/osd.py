"""
OSD Overlay — display/osd.py

Renders a minimal, clean flight-style OSD with Montserrat font via PIL.
"""

import os
import cv2
import numpy as np
import time
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont
from ml.detection import Detection

OSD_WHITE   = (240, 240, 240)
OSD_GREEN   = (60,  220, 80)
OSD_RED     = (210, 30,  30)
OSD_CYAN    = (40,  210, 200)
OSD_ORANGE  = (255, 140, 0)
OSD_GRAY    = (140, 140, 140)

# ── Font setup (Montserrat via Pillow) ──────────────────────────────
_FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts", "Montserrat.ttf")
_FONT_PATH = os.path.normpath(_FONT_PATH)

def _load(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_FONT_PATH, size)

# Pre-load sizes used across the OSD (pt sizes tuned for 1280×720)
_F_SM   = _load(14)   # small labels (km/h, etc.)
_F_MED  = _load(17)   # standard telemetry
_F_LG   = _load(20)   # armed/mode status
_F_XL   = _load(22)   # alert banner
_F_TOP  = _load(16)   # top-bar FPS / RSSI

# Start time for fly timer
START_TIME = time.time()


def _bgr2rgb(color):
    """Convert OpenCV BGR tuple to RGB for Pillow."""
    if len(color) == 3:
        return (color[2], color[1], color[0])
    return color


def _o(img: np.ndarray, text: str, pos, font: ImageFont.FreeTypeFont = None,
       color=OSD_WHITE, shadow: bool = True):
    """
    Draw Montserrat text onto a BGR numpy array via PIL.
    pos = (x, y) — top-left of text baseline (matching the old cv2 convention).
    Colors are BGR tuples (same as rest of codebase); converted internally.
    """
    if font is None:
        font = _F_MED

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    rgb = _bgr2rgb(color)
    x, y = pos

    if shadow:
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=rgb)

    result = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    img[:] = result


def _text_w(text: str, font: ImageFont.FreeTypeFont) -> int:
    """Return the pixel width of a string rendered with the given font."""
    dummy = Image.new("RGB", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _line(img, p1, p2, color=OSD_WHITE, t=1):
    cv2.line(img, p1, p2, color, t, cv2.LINE_AA)


class OSDRenderer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.W   = cfg.display_width
        self.H   = cfg.display_height

    def render(
        self,
        frame:    np.ndarray,
        dets:     List[Detection],
        num_humans:      int    = 0,
        total_unique:    int    = 0,
        radar_present:   bool   = False,
        radar_state:     int    = 0,
        radar_dist_cm:   int    = 0,
        radar_strength:  int    = 0,
        fps:             float  = 0.0,
        frame_id:        int    = 0,
        recording:       bool   = False,
        sensor_ok:       Optional[dict] = None,
        yolo_backend:    str    = "none",
        total_ms:        float  = 0.0,
        fc_telemetry:    Optional[dict] = None,
        anomaly_triggered: bool = False,
        inference_fps:   float  = 0.0,
        video_fps:       float  = 0.0,
    ) -> np.ndarray:
        out = frame.copy()
        H, W = out.shape[:2]
        cx, cy = W // 2, H // 2

        fc = fc_telemetry or {}

        # MAVLink Variables
        lat       = fc.get("lat", 0.0)
        lon       = fc.get("lon", 0.0)
        sats      = fc.get("sats", 0)
        alt       = fc.get("alt_m", 0.0)
        speed     = fc.get("speed_kmh", 0.0)
        volts     = fc.get("battery_v", 0.0)
        armed     = fc.get("armed", False)
        pitch     = fc.get("pitch", 0.0)
        roll      = fc.get("roll", 0.0)
        hdg       = fc.get("heading", 0.0)
        mode      = fc.get("mode", "UNKNOWN")
        rssi      = fc.get("rssi", 0)
        throttle  = fc.get("throttle", 0)
        batt_rem  = fc.get("batt_rem", 0)
        dist_home = fc.get("dist_home", 0.0)

        # ── 0. Top Center: FPS Metrics ──────────────────────────
        ifps_col = OSD_GREEN  if inference_fps >= 3.0  else OSD_ORANGE
        vfps_col = OSD_GREEN  if video_fps     >= 15.0 else OSD_ORANGE
        ifps_txt = f"INF {inference_fps:4.1f} FPS"
        vfps_txt = f"VID {video_fps:4.1f} FPS"
        gap      = 20
        ifps_w   = _text_w(ifps_txt, _F_TOP)
        total_w  = ifps_w + gap + _text_w(vfps_txt, _F_TOP)
        fx       = cx - total_w // 2
        _o(out, ifps_txt, (fx, 10),              _F_TOP, ifps_col)
        _o(out, vfps_txt, (fx + ifps_w + gap, 10), _F_TOP, vfps_col)

        # ── 1. Top Left: LAT / LON ──────────────────────────────
        _o(out, f"LAT  {lat: .7f}", (30, 30), _F_MED, OSD_WHITE)
        _o(out, f"LON  {lon: .7f}", (30, 55), _F_MED, OSD_WHITE)

        # ── 2. Top Right: RSSI & Timer ──────────────────────────
        fly_sec = int(time.time() - START_TIME)
        fly_mn  = fly_sec // 60
        fly_s   = fly_sec % 60
        rssi_txt  = f"RSSI {rssi}%"
        timer_txt = f"{fly_mn:02d}:{fly_s:02d}"
        _o(out, rssi_txt,  (W - _text_w(rssi_txt,  _F_TOP) - 20, 10), _F_TOP, OSD_WHITE)
        _o(out, timer_txt, (W - _text_w(timer_txt, _F_TOP) - 20, 32), _F_TOP, OSD_WHITE)
        if recording:
            _o(out, "● REC", (W - 90, 54), _F_SM, OSD_RED)

        # ── 3. Bottom left panel (Power) ───────────────────────
        # BAT/AMP/MAH are rendered on top of the thermal PiP in rescue_display.py
        by = H - 160

        # ── 4. Center-Left: Disarm Status ───────────────────────
        cx_left = W // 2 - 250
        cell_v  = (volts / 4.0) if volts > 5.0 else volts
        _o(out, f"CELL  {cell_v:.2f} v", (cx_left, by),      _F_MED, OSD_WHITE)
        status_txt = "ARMED" if armed else "DISARMED"
        _o(out, status_txt,               (cx_left, by + 30), _F_LG,  OSD_WHITE)
        _o(out, f"MODE  {mode}",           (cx_left, by + 58), _F_MED, OSD_WHITE)

        # ── 5. Center-right Telemetry ──────────────────────────
        cx_right = W // 2 + 150
        ry = H // 2 - 50
        _o(out, f"HOME  {dist_home:.0f} M", (cx_right, ry),       _F_MED, OSD_WHITE)
        _o(out, f"THR   {throttle}%",        (cx_right, ry + 30),  _F_MED, OSD_WHITE)
        _o(out, f"SAT   {sats}",             (cx_right, ry + 60),  _F_MED, OSD_WHITE)
        _o(out, f"ALT   {alt:.1f} M",        (cx_right, ry + 90),  _F_MED, OSD_WHITE)

        # ── 6. Bottom Center: Alert Banner ──────────────────────
        if num_humans > 0:
            alert_text = "HUMAN DETECTED!"
            col = OSD_ORANGE
        elif anomaly_triggered:
            alert_text = "POSSIBLE HUMAN (THERMAL)"
            col = OSD_CYAN
        else:
            alert_text = "RESCUE DRONE OSD"
            col = OSD_WHITE

        atw = _text_w(alert_text, _F_XL)
        _o(out, alert_text, (cx - atw // 2, H - 36), _F_XL, col)

        # ── 7. Center Crosshair & Speed ─────────────────────────
        _line(out, (cx - 15, cy), (cx - 5, cy), OSD_WHITE, 2)
        _line(out, (cx + 5,  cy), (cx + 15, cy), OSD_WHITE, 2)
        _line(out, (cx, cy - 15), (cx, cy - 5),  OSD_WHITE, 2)
        _line(out, (cx, cy + 5),  (cx, cy + 15), OSD_WHITE, 2)
        cv2.circle(out, (cx, cy), 3, OSD_WHITE, 1, cv2.LINE_AA)

        _o(out, f"{speed:.0f}", (cx + 22, cy - 8),  _F_MED, OSD_WHITE)
        _o(out, "km/h",         (cx + 22, cy + 14), _F_SM,  OSD_GRAY)

        # ── 8. Attitude Telemetry ────────────────────────────────
        _o(out, f"HDG  {hdg:03.0f}",           (cx - 45, cy + 70),  _F_MED, OSD_WHITE)
        _o(out, f"P {pitch:+.0f}  R {roll:+.0f}", (cx - 45, cy + 96), _F_SM,  OSD_WHITE)

        # ── 9. Bounding Boxes ────────────────────────────────────
        for det in dets:
            x1 = int(det.x1); y1 = int(det.y1)
            x2 = int(det.x2); y2 = int(det.y2)
            box_col = OSD_ORANGE
            L = 20
            _line(out, (x1, y1),   (x1+L, y1),   box_col, 1)
            _line(out, (x1, y1),   (x1, y1+L),   box_col, 1)
            _line(out, (x2, y1),   (x2-L, y1),   box_col, 1)
            _line(out, (x2, y1),   (x2, y1+L),   box_col, 1)
            _line(out, (x1, y2),   (x1+L, y2),   box_col, 1)
            _line(out, (x1, y2),   (x1, y2-L),   box_col, 1)
            _line(out, (x2, y2),   (x2-L, y2),   box_col, 1)
            _line(out, (x2, y2),   (x2, y2-L),   box_col, 1)

            cx_b  = (x1 + x2) // 2
            label = "HUMAN"
            lw    = _text_w(label, _F_SM)
            _o(out, label, (cx_b - lw // 2, y1 - 20), _F_SM, box_col)

        return out
