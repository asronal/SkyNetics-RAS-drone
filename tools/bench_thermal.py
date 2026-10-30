#!/usr/bin/env python3
"""MLX90640 read cost: frames/s and CPU per frame, split into status polling,
data read and maths.  Usage: ~/venv/bin/python tools/bench_thermal.py SECONDS [2|4|8]"""
import sys, time, collections
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from config import Config
from sensors.thermal_camera import ThermalCamera
import adafruit_mlx90640 as M
DUR = float(sys.argv[1]); HZ = sys.argv[2] if len(sys.argv) > 2 else "4"
cfg = Config(); cam = ThermalCamera(cfg); assert cam.initialize()
s = cam._sensor
s.refresh_rate = {"2": M.RefreshRate.REFRESH_2_HZ, "4": M.RefreshRate.REFRESH_4_HZ, "8": M.RefreshRate.REFRESH_8_HZ}[HZ]
cpu = collections.Counter(); calls = collections.Counter()
def timed(name, fn):
    def w(*a, **k):
        c = time.thread_time()
        try: return fn(*a, **k)
        finally: cpu[name] += time.thread_time() - c; calls[name] += 1
    return w
orig_read = s._I2CReadWords
def rd(addr, buf, end=None):
    name = "poll" if addr == 0x8000 else "read"
    c = time.thread_time()
    try: return orig_read(addr, buf, end=end) if end is not None else orig_read(addr, buf)
    finally: cpu[name] += time.thread_time() - c; calls[name] += 1
s._I2CReadWords = rd
s._CalculateTo = timed("maths", s._CalculateTo)
time.sleep(0.5)
n = err = 0; t0 = time.monotonic(); c0 = time.thread_time()
while time.monotonic() - t0 < DUR:
    raw, vis = cam.read()
    if vis is None: err += 1; time.sleep(0.02); continue
    n += 1
wall = time.monotonic() - t0; tot = time.thread_time() - c0
print(f"{HZ} Hz: {n/wall:.2f} frames/s, {wall/max(n,1)*1000:.0f} ms/frame, skipped/err {err}")
print(f"  thread CPU {tot/wall*100:.0f}% of a core = poll {cpu['poll']/wall*100:.0f}% ({calls['poll']/max(n,1):.0f} polls/frame) "
      f"+ data read {cpu['read']/wall*100:.0f}% + maths {cpu['maths']/wall*100:.0f}% + rest {(tot-cpu['poll']-cpu['read']-cpu['maths'])/wall*100:.0f}%")
