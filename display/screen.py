"""
Output helpers: which display profile to use (HDMI or the analog composite
output), and keeping an X display awake while the OSD is running.

An idle X session blanks the screen and powers the monitor down (DPMS) after
10 minutes without keyboard/mouse input. For an FPV OSD that is black goggles
mid-flight — and while the display is powered down, X only delivers about one
vblank per second, which throttles GPU-presented video to ~1fps.
"""

import glob
import os
import re
import shutil
import subprocess
import logging
from typing import Optional, Tuple

logger = logging.getLogger("screen")


def _xset(*args) -> str:
    return subprocess.run(["xset", *args], capture_output=True, text=True, timeout=3).stdout


def composite_size() -> Optional[Tuple[int, int]]:
    """(width, height) of the composite output when it is the active display,
    else None. Enabled with dtoverlay=vc4-kms-v3d,composite (which turns HDMI
    off on a Pi 4)."""
    for conn in sorted(glob.glob("/sys/class/drm/card*-Composite-*")):
        try:
            with open(os.path.join(conn, "enabled")) as f:
                if f.read().strip() != "enabled":
                    continue
        except OSError:
            continue
        # The X server knows the mode actually in use; without X, the
        # connector's first (preferred) mode — PAL or NTSC per vc4.tv_norm.
        if os.environ.get("DISPLAY") and shutil.which("xrandr"):
            try:
                out = subprocess.run(["xrandr", "--current"], capture_output=True, text=True, timeout=3).stdout
                m = re.search(r"^Composite\S*\s+connected\s+(?:primary\s+)?(\d+)x(\d+)", out, re.M)
                if m:
                    return int(m.group(1)), int(m.group(2))
            except Exception:
                pass
        try:
            with open(os.path.join(conn, "modes")) as f:
                m = re.match(r"(\d+)x(\d+)", f.readline())
            if m:
                return int(m.group(1)), int(m.group(2))
        except OSError:
            pass
        return 720, 576
    return None


def apply_display_profile(cfg, requested: Optional[str] = None) -> str:
    """Resolve cfg.display_profile ("auto" / "hdmi" / "composite") and set the
    resolution, camera streams, frame rate and OSD style for it."""
    profile = requested or cfg.display_profile
    size = composite_size() if profile in ("auto", "composite") else None
    if profile == "auto":
        profile = "composite" if size else "hdmi"
    if profile == "composite":
        w, h = size or (720, 576)
        cfg.display_width, cfg.display_height = w, h
        # 4:3 like the output: every camera stream shares the ISP's crop, so a
        # 16:9 ML stream next to a 4:3 display stream would be squashed.
        cfg.rgb_width, cfg.rgb_height = 640, 480
        cfg.rgb_fps = 25 if h >= 576 else 30       # PAL 25 / NTSC 29.97 frames/s
        cfg.osd_style = "composite"
    return profile


class ScreenBlankingInhibitor:
    def __init__(self):
        self._restore = []

    def start(self):
        if not os.environ.get("DISPLAY") or not shutil.which("xset"):
            return
        try:
            q = _xset("q")
            saver   = re.search(r"timeout:\s+(\d+)\s+cycle:\s+(\d+)", q)
            blank   = re.search(r"prefer blanking:\s+(yes|no)", q)
            dpms_t  = re.search(r"Standby:\s+(\d+)\s+Suspend:\s+(\d+)\s+Off:\s+(\d+)", q)
            dpms_on = "DPMS is Enabled" in q
            restore = []
            if saver:
                restore.append(("s", saver.group(1), saver.group(2)))
            if blank:
                restore.append(("s", "blank" if blank.group(1) == "yes" else "noblank"))
            if dpms_t:
                restore.append(("dpms", *dpms_t.groups()))
            restore.append(("+dpms",) if dpms_on else ("-dpms",))
            # "force on" re-enables DPMS, so it must come before "-dpms".
            for args in (("dpms", "force", "on"), ("s", "off"), ("s", "noblank"), ("-dpms",)):
                _xset(*args)
            self._restore = restore
            logger.info("[SCREEN] X screen blanking / DPMS power-off disabled while running")
        except Exception as e:
            logger.warning(f"[SCREEN] Could not disable screen blanking: {e}")

    def restore(self):
        try:
            for args in self._restore:
                _xset(*args)
        except Exception as e:
            logger.warning(f"[SCREEN] Could not restore screen blanking settings: {e}")
        self._restore = []
