"""
Correctness checks for the OSD layer (no camera or screen needed):

  1. The GPU path's transparent OSD layer, blended over a frame, must match the
     OpenCV path's render of the same frame (only premultiplied-alpha rounding).
  2. After many incremental updates (moving boxes, changing text) the
     OverlayCompositor canvas must be identical to a fresh full redraw.

Both checks run for the HDMI and the composite (720x576) output profiles.

usage:  ~/venv/bin/python tools/check_overlay.py [--save DIR]   (exit code 1 on failure)
        --save DIR  also writes each profile's rendered OSD frame to DIR as PNG
"""
import os, sys, time, math
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np, cv2, logging
logging.disable(logging.WARNING)
from config import Config
from pipeline.detection_pipeline import FrameData
from ml.detection import Detection
from sensors.thermal_camera import ThermalCamera
import display.osd as osd
from display.rescue_display import RescueDisplay
from display.gpu_display import OverlayCompositor, _composite
from display.screen import apply_display_profile

# Freeze the OSD's flight timer: two renders on either side of a second
# boundary would differ in its "00:00" text and fail check 1 by chance.
osd.time = type("FrozenClock", (), {"time": staticmethod(lambda: osd.START_TIME + 10.0)})
save_dir = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
tele = dict(lat=28.6132456, lon=77.2096754, sats=18, alt_m=95.6, speed_kmh=2.4, battery_v=15.43, mode="ANGLE",
            rssi=72, throttle=64, batt_rem=80, current_a=12.3, batt_mah=640, dist_home=128)


def check(profile: str) -> bool:
    cfg = Config(demo_mode=True, snapshot_enabled=False, fullscreen=False)
    apply_display_profile(cfg, profile)
    W, H = cfg.display_width, cfg.display_height
    rng = np.random.default_rng(5)
    np.random.seed(5)
    raw, vis = ThermalCamera(cfg)._synthetic()
    # A frame with bright and dark areas so both adaptive text styles are exercised.
    cam = np.clip(cv2.GaussianBlur(rng.integers(0, 256, (cfg.rgb_height, cfg.rgb_width, 3)).astype(np.uint8),
                                   (0, 0), 25) * 3.0 - 200, 0, 255).astype(np.uint8)
    base = cv2.resize(cam, (W, H))
    sx, sy = W / 1280, H / 720

    def fd_at(i):
        a = i * 0.15
        dets = [Detection(sx * (500 + 200 * math.cos(a)), sy * (200 + 80 * math.sin(a)),
                          sx * (700 + 200 * math.cos(a)), sy * (520 + 80 * math.sin(a)), 0.7, "rgb", 1),
                Detection(sx * 900, sy * (300 + i * 3), sx * 1000, sy * (450 + i * 3), 0.5, "anomaly", 2)]
        return FrameData(time.time(), 100 + i // 10, raw, vis, cam, None, dets, True, True, {"total_ms": 700}, 1.4,
                         {"thermal": True, "rgb": True, "radar": False, "fc": True}, "onnx",
                         dict(tele, alt_m=95.6 + i * 0.1), 2)

    ok = True
    sw, gp = RescueDisplay(cfg, window=False), RescueDisplay(cfg, window=False)
    pw, ph = sw._osd.pip_rect(W, H)[2:]
    for d in (sw, gp):
        d._thermal_pane(fd_at(0), pw, ph)            # start the thermal isolator worker
    time.sleep(0.6)
    sw._video_fps = 29.9
    soft = sw._build(fd_at(0))
    if save_dir:
        cv2.imwrite(os.path.join(save_dir, f"osd_{profile}.png"), soft)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    comp = OverlayCompositor(W, H)
    comp.update(gp.build_overlay(fd_at(0), np.ascontiguousarray(gray[::4, ::4]), 29.9).ops)
    d = np.abs(soft.astype(int) - _composite(cv2.cvtColor(base, cv2.COLOR_BGR2RGBA), comp.canvas).astype(int)).max(2)
    print(f"[1] {profile} {W}x{H}: GPU OSD layer over frame vs OpenCV render: max diff {d.max()}, "
          f"pixels >8 levels: {(d > 8).sum()}")
    ok &= (d > 8).sum() == 0

    for view, name in enumerate(("RGB", "thermal", "radar")):
        gp._main_view_idx = view
        comp2 = OverlayCompositor(W, H)
        t0, px = time.perf_counter(), 0
        for i in range(60):
            dl = gp.build_overlay(fd_at(i), gray, 29.0 + (i % 7) * 0.1)
            px += sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in comp2.update(dl.ops))
        fresh = OverlayCompositor(W, H)
        fresh.update(dl.ops)
        dd = int(np.abs(comp2.canvas.astype(int) - fresh.canvas.astype(int)).max())
        print(f"[2] {profile} {name:7s} view: incremental OSD canvas vs fresh redraw after 60 updates: max diff {dd} "
              f"({(time.perf_counter() - t0) / 60 * 1000:.1f} ms/update, ~{px / 60 / 1000:.0f}k px uploaded/update)")
        ok &= dd == 0
    sw.close(); gp.close()
    return ok


ok = all([check("hdmi"), check("composite")])
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
