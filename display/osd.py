"""
OSD Overlay — display/osd.py

Renders a minimal, clean flight-style OSD with Montserrat font via PIL.

The OSD is built as a DrawList of small draw operations (lines, text sprites,
images). The same list can be drawn straight onto a BGR video frame (OpenCV
display path) or onto a transparent premultiplied-alpha BGRA layer (GPU display
path), where only the regions whose operations changed need redrawing.
"""

import os
import cv2
import numpy as np
import time
from collections import OrderedDict
from functools import lru_cache
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont
from ml.detection import Detection

OSD_WHITE   = (255, 255, 255)   # pure white for maximum contrast
OSD_GREEN   = (60,  220, 80)
OSD_RED     = (210, 30,  30)
OSD_CYAN    = (40,  210, 200)
OSD_ORANGE  = (255, 140, 0)
OSD_GRAY    = (140, 140, 140)

# ── Adaptive text contrast ──────────────────────────────────────────
# This is an avalanche/snow rescue drone — a large fraction of flight time
# is spent over bright white snow or sky, where plain white text with a
# thin black outline can wash out. We sample the local background
# luminance under each text element (from a cheap downsampled grayscale
# copy of the frame, computed once per render) and pick a fill/outline
# pair that stays legible against it, while keeping each element's hue —
# darkened orange still reads as "orange", not just "some dark color".
# Per-element EMA smoothing (keyed by a stable id per on-screen element)
# stops the style flickering every frame as the camera pans across an
# edge between bright snow and dark rock.
_LUMA_GRID_W, _LUMA_GRID_H = 64, 36     # coarse — only need "bright or dark here"
_LUMA_BRIGHT_THRESH = 150.0             # 0-255 grayscale; snow/sky read reliably above this
_LUMA_EMA_ALPHA     = 0.15              # smoothing factor for the per-element luminance sample

# ── Font setup (Montserrat via Pillow) ──────────────────────────────
_FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts", "Montserrat.ttf")
_FONT_PATH = os.path.normpath(_FONT_PATH)

def _load(size: int, weight: str) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(_FONT_PATH, size)
    # Montserrat.ttf is a variable font whose default instance is "Thin"
    # (weight 100): the white fill was a hairline inside the black outline.
    try:
        font.set_variation_by_name(weight)
    except (OSError, ValueError):
        pass   # a static (non-variable) font file — use it as it is
    return font


class OsdStyle:
    """Fonts and drawing rules for one output.

    "hdmi": the 1280x720 layout (scaled up to the monitor by the GPU).
    "composite": 720x576 PAL / 720x480 NTSC analog video for the VTX — larger
    bold text inside the title-safe area (analog goggles overscan the edges),
    2px lines (1px horizontal lines flicker on interlaced video), and fixed
    white text with a black outline: thin coloured text smears in composite's
    low colour bandwidth, so the adaptive colour styling is HDMI-only."""

    def __init__(self, name, weight, sm, med, lg, xl, top, line, adaptive):
        self.name, self.line, self.adaptive = name, line, adaptive
        self.f_sm  = _load(sm,  weight)   # small labels (km/h, etc.)
        self.f_med = _load(med, weight)   # standard telemetry
        self.f_lg  = _load(lg,  weight)   # armed/mode status
        self.f_xl  = _load(xl,  weight)   # alert banner
        self.f_top = _load(top, weight)   # top-bar FPS / RSSI


# Pixel sizes tuned for 1280x720 / 720x576 respectively
HDMI_STYLE      = OsdStyle("hdmi",      "SemiBold", 14, 17, 20, 22, 16, line=1, adaptive=True)
COMPOSITE_STYLE = OsdStyle("composite", "Bold",     20, 24, 26, 28, 20, line=2, adaptive=False)

# Start time for fly timer
START_TIME = time.time()


# Reused for every _text_w() call instead of allocating a new dummy
# Image+Draw each time — textbbox() only queries font glyph metrics, it
# never touches the image's pixels, so one shared 1x1 context is safe to
# reuse across all text/font combinations. With adaptive contrast now
# calling _text_w() for every OSD element (~20/frame) to size the
# luminance-sample region, this removes ~20 PIL object allocations/frame.
_DUMMY_IMG  = Image.new("RGB", (1, 1))
_DUMMY_DRAW = ImageDraw.Draw(_DUMMY_IMG)

_STROKE = 2   # outline width (px) used for every OSD text element


@lru_cache(maxsize=1024)
def _text_w(text: str, font: ImageFont.FreeTypeFont) -> int:
    """Return the pixel width of a string rendered with the given font."""
    bbox = _DUMMY_DRAW.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


# ── Cached text sprites ─────────────────────────────────────────────
# Drawing ~20 stroked strings through PIL plus a full 1280x720
# numpy<->PIL round-trip cost ~45ms per frame on the Pi 4 — more than the
# whole 33ms budget for 30fps video. Nearly all OSD strings are identical
# from one frame to the next, so each (text, font, colours) combination is
# rasterised ONCE into a small premultiplied BGR sprite + alpha, and every
# frame just alpha-blends those small sprites with numpy.
_SPRITES: "OrderedDict[tuple, list]" = OrderedDict()
_SPRITE_CACHE_MAX = 512


def _sprite(text: str, font: ImageFont.FreeTypeFont, fill, stroke):
    """[premultiplied BGR, inverse alpha, origin dx, origin dy, premultiplied BGRA (lazy)]"""
    key = (text, font, fill, stroke)
    s = _SPRITES.get(key)
    if s is not None:
        _SPRITES.move_to_end(key)
        return s
    l, t, r, b = font.getbbox(text, stroke_width=_STROKE)
    w, h = max(1, r - l), max(1, b - t)
    # Separate coverage masks for the whole stroked glyph and the inner fill,
    # so the colour is exact: fill where the glyph is, outline elsewhere.
    full = Image.new("L", (w, h), 0)
    ImageDraw.Draw(full).text((-l, -t), text, font=font, fill=255,
                              stroke_width=_STROKE, stroke_fill=255)
    inner = Image.new("L", (w, h), 0)
    ImageDraw.Draw(inner).text((-l, -t), text, font=font, fill=255)
    a  = np.asarray(full,  dtype=np.uint16)[..., None]
    fm = np.asarray(inner, dtype=np.uint16)[..., None]
    fill_bgr   = np.array(fill[::-1],   dtype=np.uint16)
    stroke_bgr = np.array(stroke[::-1], dtype=np.uint16)
    colour = (stroke_bgr * (255 - fm) + fill_bgr * fm) // 255
    s = [colour * a, 255 - a, l, t, None]
    _SPRITES[key] = s
    if len(_SPRITES) > _SPRITE_CACHE_MAX:
        _SPRITES.popitem(last=False)
    return s


def _blit(img: np.ndarray, sprite, x: int, y: int):
    """Alpha-blend a cached sprite onto `img` with its text origin at (x, y).

    `img` is either a BGR frame or a premultiplied BGRA layer; the same
    "over" formula applies to both because the sprite is premultiplied."""
    pre, inv, l, t = sprite[0], sprite[1], sprite[2], sprite[3]
    if img.shape[2] == 4:
        if sprite[4] is None:
            sprite[4] = np.concatenate([pre, (255 - inv) * 255], axis=2)
        pre = sprite[4]
    sh, sw = inv.shape[:2]
    H, W = img.shape[:2]
    x0, y0 = int(x) + l, int(y) + t
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(W, x0 + sw), min(H, y0 + sh)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sy, sx = slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0)
    roi = img[fy0:fy1, fx0:fx1]
    roi[:] = ((roi * inv[sy, sx] + pre[sy, sx]) // 255).astype(np.uint8)


def _col(img: np.ndarray, color):
    """cv2 colour for `img`: unchanged for BGR, opaque alpha appended for BGRA."""
    return (*color, 255) if img.shape[2] == 4 else color


class DrawList:
    """Ordered OSD draw operations: (signature, bounding rect, draw function).

    `signature` captures everything that affects the op's pixels, so a GPU
    overlay can tell exactly which screen regions changed between updates.
    Each draw function takes (img, ox, oy) and draws with the origin shifted
    by (-ox, -oy), which lets a changed region be redrawn into its own
    buffer. Drawing every op in order at offset (0, 0) gives exactly the same
    pixels as calling the cv2 / sprite functions directly."""

    def __init__(self):
        self.ops = []

    def draw(self, img: np.ndarray):
        for _, _, fn in self.ops:
            fn(img, 0, 0)

    def line(self, p1, p2, color, t=1):
        (x1, y1), (x2, y2) = p1, p2
        m = t + 2
        rect = (min(x1, x2) - m, min(y1, y2) - m, max(x1, x2) + m + 1, max(y1, y2) + m + 1)
        def fn(img, ox, oy):
            cv2.line(img, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy), _col(img, color), t, cv2.LINE_AA)
        self.ops.append((("line", p1, p2, color, t), rect, fn))

    def circle(self, c, r, color, t=1):
        x, y = c
        m = r + t + 2
        def fn(img, ox, oy):
            cv2.circle(img, (x - ox, y - oy), r, _col(img, color), t, cv2.LINE_AA)
        self.ops.append((("circle", c, r, color, t), (x - m, y - m, x + m + 1, y + m + 1), fn))

    def rectangle(self, p1, p2, color, t=1):
        (x1, y1), (x2, y2) = p1, p2
        m = t + 1
        def fn(img, ox, oy):
            cv2.rectangle(img, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy), _col(img, color), t)
        self.ops.append((("rect", p1, p2, color, t), (x1 - m, y1 - m, x2 + m + 1, y2 + m + 1), fn))

    def text(self, text, font, fill, stroke, x, y):
        s = _sprite(text, font, fill, stroke)
        sh, sw = s[1].shape[:2]
        rect = (x + s[2], y + s[3], x + s[2] + sw, y + s[3] + sh)
        def fn(img, ox, oy):
            _blit(img, s, x - ox, y - oy)
        self.ops.append((("text", text, id(font), fill, stroke, x, y), rect, fn))

    def cv_text(self, text, org, scale, color, thickness=1, shadow=True, font=cv2.FONT_HERSHEY_DUPLEX):
        """cv2.putText (Hershey) with an optional 1px drop shadow."""
        x, y = org
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness + 1)
        m = thickness + 4
        rect = (x - m, y - th - m, x + tw + m + 1, y + base + m + 1)
        def fn(img, ox, oy):
            if shadow:
                cv2.putText(img, text, (x + 1 - ox, y + 1 - oy), font, scale, _col(img, (0, 0, 0)),
                            thickness + 1, cv2.LINE_AA)
            cv2.putText(img, text, (x - ox, y - oy), font, scale, _col(img, color), thickness, cv2.LINE_AA)
        self.ops.append((("cvtext", text, org, scale, color, thickness, shadow), rect, fn))

    def image(self, src: np.ndarray, x: int, y: int, key):
        """Opaque BGR image pasted at (x, y). `key` must change whenever its pixels do."""
        h, w = src.shape[:2]
        def fn(img, ox, oy):
            H, W = img.shape[:2]
            dx0, dy0 = x - ox, y - oy
            fx0, fy0 = max(0, dx0), max(0, dy0)
            fx1, fy1 = min(W, dx0 + w), min(H, dy0 + h)
            if fx1 <= fx0 or fy1 <= fy0:
                return
            roi = img[fy0:fy1, fx0:fx1]
            roi[..., :3] = src[fy0 - dy0:fy1 - dy0, fx0 - dx0:fx1 - dx0]
            if img.shape[2] == 4:
                roi[..., 3] = 255
        self.ops.append((("image", key, x, y, w, h), (x, y, x + w, y + h), fn))


def luma_grid(gray: np.ndarray) -> np.ndarray:
    """64x36 grid of mean luminance from a grayscale image (any size)."""
    return cv2.resize(gray, (_LUMA_GRID_W, _LUMA_GRID_H), interpolation=cv2.INTER_AREA)


class OSDRenderer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.W   = cfg.display_width
        self.H   = cfg.display_height
        self.style = COMPOSITE_STYLE if getattr(cfg, "osd_style", "hdmi") == "composite" else HDMI_STYLE
        self._luma_ema: dict = {}   # per-element id -> smoothed background luminance

    @staticmethod
    def luma_grid_bgr(frame: np.ndarray) -> np.ndarray:
        # Cheap downsampled grayscale of the untouched video, used to sample
        # local background brightness for adaptive text contrast. Built from
        # every 4th pixel: only a coarse 64x36 "bright or dark here" grid is
        # needed, and a full-res cvtColor + INTER_AREA shrink cost ~4.5ms per
        # frame on the Pi 4.
        return luma_grid(cv2.cvtColor(np.ascontiguousarray(frame[::4, ::4, :3]), cv2.COLOR_BGR2GRAY))

    def _bg_luma(self, gray_small, key, x, y, tw, th, sw, sh) -> float:
        """Sample the mean background luminance under a text region from the
        pre-downsampled grayscale frame, EMA-smoothed per element id."""
        gh, gw = gray_small.shape[:2]
        gx1 = max(0, min(gw - 1, int(x * gw / sw)))
        gy1 = max(0, min(gh - 1, int(y * gh / sh)))
        gx2 = max(gx1 + 1, min(gw, int((x + max(tw, 1)) * gw / sw)))
        gy2 = max(gy1 + 1, min(gh, int((y + max(th, 1)) * gh / sh)))
        raw = float(gray_small[gy1:gy2, gx1:gx2].mean())
        prev = self._luma_ema.get(key, raw)
        smoothed = (1 - _LUMA_EMA_ALPHA) * prev + _LUMA_EMA_ALPHA * raw
        self._luma_ema[key] = smoothed
        # Per-track keys (human_<id>) accumulate over a long mission as
        # track ids cycle — bound the dict rather than leak indefinitely.
        if len(self._luma_ema) > 128:
            self._luma_ema.clear()
        return smoothed

    @staticmethod
    def _style_for(color, luma: float):
        """Fill/outline pair that keeps `color`'s hue but stays legible
        whether the video under the text is bright (snow/sky) or dark."""
        if luma >= _LUMA_BRIGHT_THRESH:
            fill   = tuple(max(0, int(c * 0.35)) for c in color)
            stroke = (255, 255, 255)
        else:
            fill   = color
            stroke = (0, 0, 0)
        return fill, stroke

    def render(self, frame: np.ndarray, dets: List[Detection], **kwargs) -> np.ndarray:
        """Draw the OSD onto a copy of a BGR frame."""
        out = frame.copy()
        dl = DrawList()
        self.layout(dl, self.luma_grid_bgr(out), out.shape[1], out.shape[0], dets, **kwargs)
        dl.draw(out)
        return out

    def text(self, dl: DrawList, gray_small, W, H, text, pos, font, color=OSD_WHITE, key=None):
        """One OSD text element: adaptive colours on HDMI, fixed white on black for composite."""
        x, y = pos
        if self.style.adaptive:
            luma = self._bg_luma(gray_small, key or text, x, y, _text_w(text, font), font.size, W, H)
            fill, stroke = self._style_for(color, luma)
        else:
            fill, stroke = OSD_WHITE, (0, 0, 0)
        dl.text(text, font, fill, stroke, x, y)

    # ── Layout geometry ─────────────────────────────────────────────────

    @staticmethod
    def _safe(W, H):
        """(margin x, margin y, row pitch) — composite keeps everything inside
        the title-safe area (5% each side), with one text row per `row` px."""
        return round(W * 0.05), round(H * 0.05), max(26, round(H * 30 / 576))

    def pip_rect(self, W, H):
        """(x, y, w, h) of the thermal picture-in-picture."""
        if self.style.name != "composite":
            return 20, H - 240 - 20, 320, 240
        mx, my, row = self._safe(W, H)
        w = round(W * 0.27) // 4 * 4
        h = w * 3 // 4
        return mx, H - my - row - 4 - h, w, h       # just above the alert row

    def battery(self, dl: DrawList, gray_small, W, H, fc: dict):
        """BAT / AMP / MAH readout, stacked above the thermal PiP."""
        px, py, _, _ = self.pip_rect(W, H)
        lines = (f"BAT  {fc.get('battery_v', 0.0):.1f}V  {fc.get('batt_rem', 0)}%",
                 f"AMP  {fc.get('current_a', 0.0):.1f} A",
                 f"MAH  {fc.get('batt_mah', 0)}")
        if self.style.name != "composite":
            for i, t in enumerate(lines):
                dl.cv_text(t, (px + 4, py - 72 + 24 * i), 0.48, (235, 235, 235))
            return
        _, _, row = self._safe(W, H)
        for i, t in enumerate(lines):
            self.text(dl, gray_small, W, H, t, (px, py - 4 - row * (3 - i)), self.style.f_med, key=f"bat{i}")

    def layout(
        self,
        dl:       DrawList,
        gray_small: np.ndarray,
        W:        int,
        H:        int,
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
    ):
        """Append the OSD's draw operations to `dl`. `gray_small` is the
        luma grid of the video under the OSD (see luma_grid)."""
        s = self.style
        comp = s.name == "composite"
        cx, cy = W // 2, H // 2
        mx, my, row = self._safe(W, H)

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
        dist_home = fc.get("dist_home", 0.0)

        # ── Phase 1: crosshair + bbox corner brackets ───────────────────
        a0, a1 = (6, 18) if comp else (5, 15)
        dl.line((cx - a1, cy), (cx - a0, cy), OSD_WHITE, 2)
        dl.line((cx + a0, cy), (cx + a1, cy), OSD_WHITE, 2)
        dl.line((cx, cy - a1), (cx, cy - a0), OSD_WHITE, 2)
        dl.line((cx, cy + a0), (cx, cy + a1), OSD_WHITE, 2)
        dl.circle((cx, cy), 3, OSD_WHITE, s.line)

        box_col = OSD_ORANGE[::-1]   # cv2 takes BGR; the OSD_* colours are RGB (text sprites)
        L = 24 if comp else 20
        for det in dets:
            x1 = int(det.x1); y1 = int(det.y1)
            x2 = int(det.x2); y2 = int(det.y2)
            dl.line((x1, y1),   (x1+L, y1),   box_col, s.line)
            dl.line((x1, y1),   (x1, y1+L),   box_col, s.line)
            dl.line((x2, y1),   (x2-L, y1),   box_col, s.line)
            dl.line((x2, y1),   (x2, y1+L),   box_col, s.line)
            dl.line((x1, y2),   (x1+L, y2),   box_col, s.line)
            dl.line((x1, y2),   (x1, y2-L),   box_col, s.line)
            dl.line((x2, y2),   (x2-L, y2),   box_col, s.line)
            dl.line((x2, y2),   (x2, y2-L),   box_col, s.line)

        # ── Phase 2: text elements as cached sprites ───────────────────
        def txt(text, pos, font, color=OSD_WHITE, key=None):
            self.text(dl, gray_small, W, H, text, pos, font, color, key)

        def right(text, font):          # x that right-aligns `text` to the safe edge
            return W - mx - _text_w(text, font)

        # 0. Top Center: FPS Metrics (composite: third row, clear of LAT / RSSI)
        ifps_col = OSD_GREEN  if inference_fps >= 3.0  else OSD_ORANGE
        vfps_col = OSD_GREEN  if video_fps     >= 15.0 else OSD_ORANGE
        if comp:
            ifps_txt, vfps_txt, gap, fy = f"INF {inference_fps:.1f}", f"VID {video_fps:.1f}", 24, my + 2 * row
        else:
            ifps_txt, vfps_txt, gap, fy = f"INF {inference_fps:4.1f} FPS", f"VID {video_fps:4.1f} FPS", 20, 10
        ifps_w   = _text_w(ifps_txt, s.f_top)
        total_w  = ifps_w + gap + _text_w(vfps_txt, s.f_top)
        fx       = cx - total_w // 2
        txt(ifps_txt, (fx, fy),              s.f_top, ifps_col, key="ifps")
        txt(vfps_txt, (fx + ifps_w + gap, fy), s.f_top, vfps_col, key="vfps")

        # 1. Top Left: LAT / LON
        lat_txt, lon_txt = f"LAT  {lat: .7f}", f"LON  {lon: .7f}"
        if comp:
            txt(lat_txt, (mx, my),       s.f_med, key="lat")
            txt(lon_txt, (mx, my + row), s.f_med, key="lon")
        else:
            txt(lat_txt, (30, 30), s.f_med, key="lat")
            txt(lon_txt, (30, 55), s.f_med, key="lon")

        # 2. Top Right: RSSI & Timer
        fly_sec = int(time.time() - START_TIME)
        rssi_txt  = f"RSSI {rssi}%"
        timer_txt = f"{fly_sec // 60:02d}:{fly_sec % 60:02d}"
        if comp:
            txt(rssi_txt,  (right(rssi_txt,  s.f_top), my),       s.f_top, key="rssi")
            txt(timer_txt, (right(timer_txt, s.f_top), my + row), s.f_top, key="timer")
            if recording:
                txt("• REC", (right("• REC", s.f_top), my + 2 * row), s.f_top, OSD_RED, key="rec")
        else:
            txt(rssi_txt,  (W - _text_w(rssi_txt,  s.f_top) - 20, 10), s.f_top, key="rssi")
            txt(timer_txt, (W - _text_w(timer_txt, s.f_top) - 20, 32), s.f_top, key="timer")
            if recording:
                # "•" — Montserrat has no "●" (it rendered as an empty box)
                txt("• REC", (W - 90, 54), s.f_sm, OSD_RED, key="rec")

        # 3./4. Power + arming status (BAT/AMP/MAH: battery(), over the PiP).
        # HDMI: centre-left, bottom. Composite: left column, under LAT / LON.
        if comp:
            sx, sy, sdy = mx, my + 4 * row, (0, row, 2 * row)
        else:
            sx, sy, sdy = W // 2 - 250, H - 160, (0, 30, 58)
        cell_v  = (volts / 4.0) if volts > 5.0 else volts
        txt(f"CELL  {cell_v:.2f} v", (sx, sy + sdy[0]), s.f_med, key="cell")
        txt("ARMED" if armed else "DISARMED", (sx, sy + sdy[1]), s.f_lg, key="status")
        txt(f"MODE  {mode}", (sx, sy + sdy[2]), s.f_med, key="mode")

        # 5. Center-right Telemetry
        if comp:
            tx, ty, tdy = W - mx - _text_w("HOME  9999 M", s.f_med), cy - 2 * row, row
        else:
            tx, ty, tdy = W // 2 + 150, H // 2 - 50, 30
        txt(f"HOME  {dist_home:.0f} M", (tx, ty),           s.f_med, key="home")
        txt(f"THR   {throttle}%",        (tx, ty + tdy),     s.f_med, key="thr")
        txt(f"SAT   {sats}",             (tx, ty + 2 * tdy), s.f_med, key="sat")
        txt(f"ALT   {alt:.1f} M",        (tx, ty + 3 * tdy), s.f_med, key="alt")

        # 6. Bottom Center: Alert Banner
        if num_humans > 0:
            alert_text = "HUMAN DETECTED!"
            col = OSD_ORANGE
        elif anomaly_triggered:
            alert_text = "POSSIBLE HUMAN (THERMAL)"
            col = OSD_CYAN
        else:
            alert_text = "RESCUE DRONE OSD"
            col = OSD_WHITE

        atw = _text_w(alert_text, s.f_xl)
        txt(alert_text, (cx - atw // 2, H - my - row if comp else H - 36), s.f_xl, col, key="alert")

        # 7. Speed readout (crosshair itself drawn in phase 1)
        if comp:
            txt(f"{speed:.0f}", (cx + 30, cy - 14), s.f_med, key="speed")
            txt("km/h",         (cx + 30, cy + 12), s.f_sm, OSD_GRAY, key="kmh")
        else:
            txt(f"{speed:.0f}", (cx + 22, cy - 8),  s.f_med, key="speed")
            txt("km/h",         (cx + 22, cy + 14), s.f_sm, OSD_GRAY, key="kmh")

        # 8. Attitude Telemetry
        hx, hy, hdy = (cx - 50, cy + 2 * row, row) if comp else (cx - 45, cy + 70, 26)
        txt(f"HDG  {hdg:03.0f}",              (hx, hy),       s.f_med, key="hdg")
        txt(f"P {pitch:+.0f}  R {roll:+.0f}", (hx, hy + hdy), s.f_sm,  key="pr")

        # 9. Bounding box labels (corner brackets drawn in phase 1). Keyed by
        # track_id (stable per tracked person) rather than the label text
        # itself, so each person's smoothed luminance follows them around.
        for det in dets:
            x1 = int(det.x1); y1 = int(det.y1); x2 = int(det.x2)
            cx_b  = (x1 + x2) // 2
            label = "HUMAN"
            lw    = _text_w(label, s.f_sm)
            txt(label, (cx_b - lw // 2, y1 - (row - 2 if comp else 20)), s.f_sm, OSD_ORANGE,
                key=f"human_{det.track_id}")
