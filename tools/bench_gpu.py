"""
Benchmark the GPU display path exactly as main.py runs it (display process +
ML/OSD worker process) on the real camera, MLX90640, YOLO and screen.

Two moving fake tracked humans are injected so the OSD has boxes and labels
to animate. Every presented camera frame is timestamped together with its
SensorTimestamp, so skipped frames and presentation gaps are both measured.

usage (from a desktop session, or with DISPLAY=:0 XAUTHORITY=~/.Xauthority):
    [PROFILE=composite] ~/venv/bin/python tools/bench_gpu.py [seconds]

Notes: a remote-desktop viewer (AnyDesk) encoding the screen, or a monitor in
DPMS power-save, both distort the numbers — check before trusting a run.
"""
import os, sys, time, threading, dataclasses, math
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np
from ml.detection import Detection
import display.rescue_display as rd

# ── runs in BOTH processes (the spawned worker re-imports this module) ──
def _fake(frame_id):
    a = frame_id * 0.5
    return [Detection(500 + 250 * math.cos(a), 150 + 100 * math.sin(a), 700 + 250 * math.cos(a),
                      500 + 100 * math.sin(a), 0.7, "rgb", 901),
            Detection(950 + 80 * math.sin(a), 250, 1100 + 80 * math.sin(a), 520, 0.6, "anomaly", 902)]
_orig = rd.RescueDisplay.build_overlay
def _patched(self, fd, gray, video_fps):
    return _orig(self, dataclasses.replace(fd, tracked_humans=_fake(fd.frame_id)), gray, video_fps)
rd.RescueDisplay.build_overlay = _patched

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    from config import Config
    from display.gpu_display import run_gpu
    from display.screen import apply_display_profile

    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 30
    cfg = Config(fullscreen=True, snapshot_enabled=False)
    print("profile:", apply_display_profile(cfg, os.environ.get("PROFILE")),
          f"{cfg.display_width}x{cfg.display_height}, camera {cfg.rgb_fps}fps requested")
    present, rep = [], []

    def hook(gpu):
        orig_render, orig_repaint = gpu.view.render_request, gpu.view.repaint
        def timed_render(req):
            orig_render(req)
            present.append((time.perf_counter(), req.get_metadata().get("SensorTimestamp", 0)))
        def timed_repaint(req, update_viewport=False):
            t0 = time.perf_counter(); orig_repaint(req, update_viewport)
            rep.append((time.perf_counter() - t0) * 1000)
        gpu.view.render_request, gpu.view.repaint = timed_render, timed_repaint
        threading.Timer(dur + 6, gpu._sig.set).start()          # same path as SIGINT

    t_start = time.perf_counter()
    rc = run_gpu(cfg, present_hook=hook)

    t = np.array([p[0] for p in present]); s = np.array([p[1] for p in present], np.int64)
    keep = t > t_start + 6.0                                     # skip camera/worker warm-up
    t, s = t[keep], s[keep]
    gaps = np.diff(t) * 1000
    frame_ns = np.median(np.diff(s))
    skipped = int(np.sum(np.round(np.diff(s) / frame_ns) - 1))
    span = t[-1] - t[0]
    print(f"\n=== GPU display ({span:.0f}s, camera period {frame_ns / 1e6:.2f} ms) ===")
    print(f"camera frames presented : {len(t) / span:5.1f} fps   skipped {skipped} of {skipped + len(t)}")
    print(f"present gap (ms)        : median {np.median(gaps):5.1f}  p95 {np.percentile(gaps, 95):5.1f}  "
          f"p99 {np.percentile(gaps, 99):5.1f}  max {gaps.max():6.1f}  | >50ms: {(gaps > 50).sum()}  >100ms: {(gaps > 100).sum()}")
    print(f"repaint (incl. OSD uploads + swap): median {np.median(rep):.2f} ms  p95 {np.percentile(rep, 95):.2f} ms")
    # SensorTimestamp (start of sensor readout) and perf_counter are both
    # CLOCK_MONOTONIC. Excludes the display's own scan-out after the swap.
    lat = (t * 1e9 - s) / 1e6
    print(f"latency sensor->swap   : median {np.median(lat):5.1f} ms  p95 {np.percentile(lat, 95):5.1f} ms")
    print(f"worker exit code        : {rc}")
