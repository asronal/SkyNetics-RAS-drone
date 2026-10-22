"""
Benchmark the OpenCV display path (main.py --display opencv) on real hardware:
the same threaded loop as main.py, measuring how many NEW camera frames reach
the window and the gaps between them.

usage:
    FULLSCREEN=1 QT_QPA_PLATFORM=xcb ~/venv/bin/python tools/bench_opencv.py [seconds]
    (QT_QPA_PLATFORM defaults to offscreen, which hides the real cost of
     showing frames — use xcb on the real display for meaningful numbers)
"""
import os, sys, time, threading, dataclasses
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np, logging
logging.disable(logging.WARNING)
from config import Config
from pipeline.detection_pipeline import DetectionPipeline
from display.rescue_display import RescueDisplay

dur = float(sys.argv[1]) if len(sys.argv) > 1 else 30
cfg = Config(fullscreen=os.environ.get("FULLSCREEN") == "1", snapshot_enabled=False)
pipe = DetectionPipeline(cfg)
pipe.initialize()
display = RescueDisplay(cfg)

state = {"fd": None}; lock = threading.Lock()
def inference():
    for fd in pipe.run():
        with lock: state["fd"] = fd
inf = threading.Thread(target=inference, daemon=True)
inf.start()

time.sleep(3.0)
ticks = []                                   # (tick start, capture timestamp of the frame shown)
t_end = time.perf_counter() + dur
while time.perf_counter() < t_end:
    t0 = time.perf_counter()
    with lock: fd = state["fd"]
    if fd is not None:
        fresh = pipe.rgb.read()
        if fresh is not None:
            fd = dataclasses.replace(fd, rgb_frame=fresh)
        display.render(fd)
        ticks.append((t0, pipe.rgb.capture_ts))
    dt = 1.0 / 30 - (time.perf_counter() - t0)
    if dt > 0:
        time.sleep(dt)
inf_fps = state["fd"].fps if state["fd"] else 0
pipe.stop(); inf.join(timeout=5); pipe.shutdown(); display.close()

ts = np.array([t[0] for t in ticks]); caps = np.array([t[1] for t in ticks])
new = np.r_[True, np.diff(caps) > 0]
gaps = np.diff(ts[new]) * 1000
print(f"\n=== OpenCV display ({dur:.0f}s, fullscreen={cfg.fullscreen}, Qt={os.environ['QT_QPA_PLATFORM']}) ===")
print(f"display ticks         : {len(ticks) / dur:5.1f} fps   new camera frames shown: {new.sum() / dur:5.1f} fps")
print(f"new-frame gap (ms)    : median {np.median(gaps):5.1f}  p95 {np.percentile(gaps, 95):5.1f}  p99 {np.percentile(gaps, 99):5.1f}  "
      f"max {gaps.max():6.1f}  | >50ms: {(gaps > 50).sum()}  >100ms: {(gaps > 100).sum()}")
print(f"inference             : {inf_fps:5.2f} fps")
