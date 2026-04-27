"""
OSD Overlay — display/osd.py

Renders a minimal, clean flight-style OSD mimicking the DJI FPV layout.
"""

import cv2
import numpy as np
import time
from typing import List, Optional
from ml.detection import Detection

OSD_WHITE   = (240, 240, 240)
OSD_GREEN   = (60,  220, 80)
OSD_RED     = (30,  30,  210)
OSD_CYAN    = (200, 210, 40)
OSD_ORANGE  = (0,   140, 255)
OSD_GRAY    = (140, 140, 140)

# Use widely spaced font
FONT        = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL  = cv2.FONT_HERSHEY_PLAIN
FONT_THIN   = cv2.FONT_HERSHEY_SIMPLEX

# Start time for recording / fly time simulation
START_TIME = time.time()

def _o(img, text, pos, scale=0.5, color=OSD_WHITE, thickness=1, shadow=True, spacing=1):
    """Draw text, optionally spacing out characters to mimic DJI OSD."""
    if spacing <= 1:
        if shadow:
            cv2.putText(img, text, (pos[0]+1, pos[1]+1), FONT, scale, (0,0,0), thickness+1, cv2.LINE_AA)
        cv2.putText(img, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)
    else:
        # Manually space characters
        x, y = pos
        for char in text:
            if shadow:
                cv2.putText(img, char, (x+1, y+1), FONT, scale, (0,0,0), thickness+1, cv2.LINE_AA)
            cv2.putText(img, char, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
            x += int(cv2.getTextSize(char, FONT, scale, thickness)[0][0]) + spacing


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
    ) -> np.ndarray:
        out = frame.copy()
        H, W = out.shape[:2]
        cx, cy = W // 2, H // 2

        fc = fc_telemetry or {}

        # MAVLink Variables
        lat = fc.get("lat", 0.0)
        lon = fc.get("lon", 0.0)
        sats = fc.get("sats", 0)
        alt = fc.get("alt_m", 0.0)
        speed = fc.get("speed_kmh", 0.0)
        volts = fc.get("battery_v", 0.0)
        armed = fc.get("armed", False)
        pitch = fc.get("pitch", 0.0)
        roll = fc.get("roll", 0.0)
        hdg = fc.get("heading", 0.0)
        mode = fc.get("mode", "UNKNOWN")
        rssi = fc.get("rssi", 0)
        throttle = fc.get("throttle", 0)
        batt_rem = fc.get("batt_rem", 0)
        current_a = fc.get("current_a", 0.0)
        batt_mah = fc.get("batt_mah", 0)
        dist_home = fc.get("dist_home", 0.0)

        # ── 1. Top Left: LAT / LON ──────────────────────────────
        _o(out, f"LAT {lat: .7f}", (40, 50), 0.6, OSD_WHITE, 2, spacing=4)
        _o(out, f"LON {lon: .7f}", (40, 85), 0.6, OSD_WHITE, 2, spacing=4)

        # ── 2. Top Right: RSSI & Timer ──────────────────────────────
        fly_sec = int(time.time() - START_TIME)
        fly_mn = fly_sec // 60
        fly_s = fly_sec % 60
        _o(out, f"RSSI {rssi}%", (W - 130, 30), 0.5, OSD_WHITE, 2)
        _o(out, f"{fly_mn:02d}:{fly_s:02d}", (W - 130, 60), 0.5, OSD_WHITE, 2)
        if recording:
            _o(out, "REC", (W - 180, 60), 0.5, OSD_RED, 2)

        # ── 3. Bottom left panel (Power) ───────────────────────
        by = H - 120
        # Battery voltage & remaining %
        _o(out, f"BAT  {volts:.1f}V   {batt_rem}%", (40, by), 0.6, OSD_WHITE, 2, spacing=4)
        # Current draw
        _o(out, f"AMP  {current_a:.1f} A", (40, by + 40), 0.6, OSD_WHITE, 2, spacing=4)
        # Capacity consumed
        _o(out, f"MAH  {batt_mah}", (40, by + 80), 0.6, OSD_WHITE, 2, spacing=4)

        # ── 4. Center-Left Disarm Status ───────────────────────
        cx_left = W // 2 - 250
        cell_v = (volts / 4.0) if volts > 5.0 else volts
        _o(out, f"CELL {cell_v:.2f} v", (cx_left, by), 0.6, OSD_WHITE, 2, spacing=4)
        
        status_txt = "ARMED" if armed else "DISARMED"
        _o(out, status_txt, (cx_left, by + 40), 0.7, OSD_WHITE, 2, spacing=6)
        _o(out, f"MODE {mode}", (cx_left, by + 80), 0.6, OSD_WHITE, 2, spacing=4)

        # ── 5. Center-right Telemetry ──────────────────────────
        cx_right = W // 2 + 150
        ry = H // 2 - 50
        _o(out, f"HOME {dist_home:.0f} M", (cx_right, ry), 0.6, OSD_WHITE, 2, spacing=4)
        _o(out, f"THR  {throttle}%", (cx_right, ry + 40), 0.6, OSD_WHITE, 2, spacing=4)
        _o(out, f"SAT  {sats}", (cx_right, ry + 80), 0.6, OSD_WHITE, 2, spacing=4)
        _o(out, f"ALT  {alt:.1f} M", (cx_right - 10, ry + 120), 0.6, OSD_WHITE, 2, spacing=4)

        # ── 6. Bottom Center: Alert / Flip Switch ──────────────
        if num_humans > 0:
            alert_text = "HUMAN DETECTED!"
            col = OSD_ORANGE
        elif anomaly_triggered:
            alert_text = "POSSIBLE HUMAN (THERMAL)"
            col = OSD_CYAN
        else:
            alert_text = "RESCUE DRONE OSD"
            col = OSD_WHITE

        atw = cv2.getTextSize(alert_text, FONT, 0.7, 2)[0][0]
        atx = cx - atw // 2 - 20
        _o(out, alert_text, (atx, H - 40), 0.7, col, 2, spacing=10)

        # ── 7. Center Crosshair & Speed ────────────────────────
        _line(out, (cx - 15, cy), (cx - 5, cy), OSD_WHITE, 2)
        _line(out, (cx + 5, cy), (cx + 15, cy), OSD_WHITE, 2)
        _line(out, (cx, cy - 15), (cx, cy - 5), OSD_WHITE, 2)
        _line(out, (cx, cy + 5), (cx, cy + 15), OSD_WHITE, 2)
        # small circle
        cv2.circle(out, (cx, cy), 3, OSD_WHITE, 1, cv2.LINE_AA)

        _o(out, f"{speed:.0f}", (cx + 30, cy + 5), 0.6, OSD_WHITE, 2)
        _o(out, "km/h", (cx + 30, cy + 25), 0.4, OSD_WHITE, 1)

        # ── 8. Extra Attitude Telemetry ────────────────────────
        _o(out, f"HDG {hdg:03.0f}", (cx - 40, cy + 80), 0.6, OSD_WHITE, 2, spacing=4)
        _o(out, f"P {pitch:+02.0f} R {roll:+02.0f}", (cx - 50, cy + 110), 0.5, OSD_WHITE, 1, spacing=2)

        # ── 9. Bounding Boxes (Minimal) ────────────────────────
        for det in dets:
            x1 = int(det.x1); y1 = int(det.y1)
            x2 = int(det.x2); y2 = int(det.y2)

            box_col = OSD_ORANGE
            # Super thin brackets
            L = 20
            # Top-left
            _line(out, (x1, y1), (x1+L, y1), box_col, 1)
            _line(out, (x1, y1), (x1, y1+L), box_col, 1)
            # Top-right
            _line(out, (x2, y1), (x2-L, y1), box_col, 1)
            _line(out, (x2, y1), (x2, y1+L), box_col, 1)
            # Bottom-left
            _line(out, (x1, y2), (x1+L, y2), box_col, 1)
            _line(out, (x1, y2), (x1, y2-L), box_col, 1)
            # Bottom-right
            _line(out, (x2, y2), (x2-L, y2), box_col, 1)
            _line(out, (x2, y2), (x2, y2-L), box_col, 1)

            cx_b = (x1 + x2) // 2
            # minimal text above
            label = "HUMAN"
            lw = cv2.getTextSize(label, FONT_SMALL, 0.9, 1)[0][0]
            # Thin font
            cv2.putText(out, label, (cx_b - lw//2, y1 - 10), FONT_SMALL, 0.9, box_col, 1, cv2.LINE_AA)

        return out
