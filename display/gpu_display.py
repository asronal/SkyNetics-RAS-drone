"""
GPU display path — hardware-composited video with the OSD as a separate layer.

  display process (this one)                worker process
  ─────────────────────────                 ──────────────────────────────────
  camera ISP ─dmabuf─> GL texture ─┐        thermal + YOLO + fusion + tracking
                                   ├─> screen      │
  OSD texture <── changed regions ─┘        OSD DrawList ─> changed regions
        ^             (pipe)                       │
        └──────────────────────────────────────────┘
  lores frames ──(shared memory)──> YOLO / OSD contrast sampling

Each camera frame is drawn by the GPU straight from the camera's DMA buffer —
no per-frame image copies, resizes or colour conversions in Python — so the
video keeps the camera's frame rate. The OSD is a premultiplied-alpha layer:
only regions whose draw operations changed are redrawn and re-uploaded, right
before the next video frame is drawn, so OSD and video always match.

Why two processes: the GL render thread makes ~30 OpenGL calls per frame via
ctypes, each of which releases and re-takes the GIL, and any CPU-bound Python
thread in the same process (OSD building, the MLX90640 driver) can make each
re-take wait. Measured over 30s with the full pipeline: frame gaps >50ms
dropped from 6-15 (worker as a thread) to 1-3 (worker process), and from 31
to 8 with two extra CPU-bound processes running, since the worker can also run
at a lower priority than the video.

Needs a desktop (X11) session — Picamera2's GL preview widget renders into an
X window.
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"   # before any OpenGL import — Picamera2's GL widget uses EGL

import math
import multiprocessing as mp
import queue
import signal
import threading
import time
import logging
from collections import deque
from multiprocessing import shared_memory
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger("gpu_display")

# Camera buffers for the display stream. Measured sensor->swap latency is the
# same (~41-43ms) with 6, 4 or 3 buffers — it is readout + ISP + vsync, not
# queueing — while 3 buffers start skipping camera frames.
_BUFFER_COUNT = 6


class GpuDisplayUnavailable(RuntimeError):
    """The GPU display could not start (the OpenCV display can be used instead)."""


def available():
    """(ok, reason) — whether the GPU display can run in this session."""
    if not os.environ.get("DISPLAY"):
        return False, "no X desktop session (DISPLAY not set)"
    try:
        import PyQt5.QtWidgets  # noqa: F401
        from picamera2.previews.q_gl_picamera2 import _get_qglpicamera2  # noqa: F401
    except Exception as e:
        return False, f"Picamera2 GL preview unavailable: {e}"
    return True, ""


# ── Shared state between the display and worker processes ─────────────

def _lores_shape(cfg):
    stride = (cfg.rgb_width + 63) // 64 * 64          # Pi ISP output rows are 64-byte aligned
    return (cfg.rgb_height * 3 // 2, stride)


class _Link:
    """Shared memory + sync primitives. Created in the display process,
    re-attached in the worker from the picklable args()."""

    def __init__(self, cfg, ctx=None, args=None):
        self.lores_shape = _lores_shape(cfg)
        self.main_shape = (cfg.display_height, cfg.display_width, 4)
        if args is None:                                  # owner (display process)
            self._owner = True
            self.shm_lores = shared_memory.SharedMemory(create=True, size=int(np.prod(self.lores_shape)))
            self.shm_main = shared_memory.SharedMemory(create=True, size=int(np.prod(self.main_shape)))
            self.lores_lock, self.lores_seq = ctx.Lock(), ctx.Value("Q", 0, lock=False)
            self.lores_ts, self.video_fps = ctx.Value("d", 0.0, lock=False), ctx.Value("d", 0.0, lock=False)
            self.want_main, self.main_ready = ctx.Value("b", 0, lock=False), ctx.Event()
            self.stop, self.quit = ctx.Event(), ctx.Event()
            self.keys = ctx.Queue()
            self.overlay_rx, self.overlay_tx = ctx.Pipe(duplex=False)
        else:                                             # worker process
            self._owner = False
            (lores_name, main_name, self.lores_lock, self.lores_seq, self.lores_ts, self.video_fps,
             self.want_main, self.main_ready, self.stop, self.quit, self.keys, self.overlay_tx) = args
            self.shm_lores = shared_memory.SharedMemory(name=lores_name)
            self.shm_main = shared_memory.SharedMemory(name=main_name)
        self.lores = np.ndarray(self.lores_shape, np.uint8, buffer=self.shm_lores.buf)
        self.main = np.ndarray(self.main_shape, np.uint8, buffer=self.shm_main.buf)

    def args(self):
        return (self.shm_lores.name, self.shm_main.name, self.lores_lock, self.lores_seq, self.lores_ts,
                self.video_fps, self.want_main, self.main_ready, self.stop, self.quit, self.keys,
                self.overlay_tx)

    def close(self):
        del self.lores, self.main
        for shm in (self.shm_lores, self.shm_main):
            shm.close()
            if self._owner:
                shm.unlink()


def _i420(yuv: np.ndarray, w: int, h: int) -> np.ndarray:
    """Tightly packed I420 from a (possibly row-padded) Picamera2 YUV420 array."""
    if yuv.shape == (h * 3 // 2, w):
        return yuv
    s = yuv.shape[1]
    y = yuv[:h, :w]
    u = yuv[h:h + h // 4].reshape(h // 2, s // 2)[:, :w // 2]
    v = yuv[h + h // 4:h + h // 2].reshape(h // 2, s // 2)[:, :w // 2]
    return np.concatenate([y.ravel(), u.ravel(), v.ravel()]).reshape(h * 3 // 2, w)


class SharedLoresFeed:
    """BGR frames for the ML pipeline from the camera's ISP-scaled lores stream
    (written to shared memory by the display process) — the same interface as
    sensors.rgb_camera.RGBCamera."""

    def __init__(self, link: _Link, w: int, h: int):
        self._link, self._w, self._h = link, w, h

    is_online = property(lambda self: self._link.lores_seq.value > 0)
    capture_ts = property(lambda self: self._link.lores_ts.value)
    fps = property(lambda self: self._link.video_fps.value)

    def initialize(self) -> bool:
        return True

    def release(self):
        pass

    def _copy(self) -> Optional[np.ndarray]:
        if self._link.lores_seq.value == 0:
            return None
        with self._link.lores_lock:
            return self._link.lores.copy()

    def read(self) -> Optional[np.ndarray]:
        yuv = self._copy()
        return None if yuv is None else cv2.cvtColor(_i420(yuv, self._w, self._h), cv2.COLOR_YUV2BGR_I420)

    def gray(self) -> Optional[np.ndarray]:
        """Luma of the newest frame, every 2nd pixel — for adaptive OSD text contrast."""
        if self._link.lores_seq.value == 0:
            return None
        with self._link.lores_lock:
            return self._link.lores[:self._h:2, :self._w:2].copy()


# ── Incremental OSD layer (worker process) ────────────────────────────

def _merge_rects(rects, gap: int = 8, limit: int = 24):
    """Merge overlapping / nearly touching rects; one bounding box if too many remain."""
    rects = [list(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out = []
        while rects:
            r = rects.pop()
            i = 0
            while i < len(rects):
                o = rects[i]
                if r[0] - gap < o[2] and o[0] - gap < r[2] and r[1] - gap < o[3] and o[1] - gap < r[3]:
                    r = [min(r[0], o[0]), min(r[1], o[1]), max(r[2], o[2]), max(r[3], o[3])]
                    rects.pop(i)
                    changed = True
                else:
                    i += 1
            out.append(r)
        rects = out
    if len(rects) > limit:
        rects = [[min(r[0] for r in rects), min(r[1] for r in rects),
                  max(r[2] for r in rects), max(r[3] for r in rects)]]
    return [tuple(r) for r in rects]


class OverlayCompositor:
    """The OSD as a premultiplied BGRA layer. update() redraws only the regions
    whose draw operations appeared, disappeared or changed since last time,
    and returns those regions so only they need uploading to the GPU."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.canvas = np.zeros((h, w, 4), np.uint8)
        self._prev_ops: list = []

    def update(self, ops: list) -> list:
        prev_sigs = {op[0] for op in self._prev_ops}
        cur_sigs = {op[0] for op in ops}
        dirty = [op[1] for op in self._prev_ops if op[0] not in cur_sigs] + \
                [op[1] for op in ops if op[0] not in prev_sigs]
        self._prev_ops = ops
        clipped = []
        for x0, y0, x1, y1 in dirty:
            x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(self.w, int(x1)), min(self.h, int(y1))
            if x1 > x0 and y1 > y0:
                clipped.append((x0, y0, x1, y1))
        regions = _merge_rects(clipped)
        for x0, y0, x1, y1 in regions:
            # Redraw every op touching the region, in order, into a clean tile,
            # then copy back just the region. The tile is grown to contain each
            # of those ops whole: anti-aliased cv2 lines rasterise slightly
            # differently when clipped at a tile edge. Images clip exactly, so
            # they don't grow it — a full-screen image (thermal / radar view)
            # made every tile full-screen: 50-90ms per update instead of ~3ms.
            hits = [(sig, r, fn) for sig, r, fn in ops if r[0] < x1 and r[2] > x0 and r[1] < y1 and r[3] > y0]
            grow = [r for sig, r, _ in hits if sig[0] != "image"]
            tx0 = max(0, min([x0] + [int(r[0]) for r in grow]))
            ty0 = max(0, min([y0] + [int(r[1]) for r in grow]))
            tx1 = min(self.w, max([x1] + [int(r[2]) for r in grow]))
            ty1 = min(self.h, max([y1] + [int(r[3]) for r in grow]))
            tile = np.zeros((ty1 - ty0, tx1 - tx0, 4), np.uint8)
            for _, _, fn in hits:
                fn(tile, tx0, ty0)
            self.canvas[y0:y1, x0:x1] = tile[y0 - ty0:y1 - ty0, x0 - tx0:x1 - tx0]
        return regions


def _composite(frame_rgbx: np.ndarray, osd_bgra: np.ndarray) -> np.ndarray:
    """Camera frame (Picamera2 XBGR8888 = RGBX bytes) with the OSD layer over it, as BGR."""
    bgr = cv2.cvtColor(frame_rgbx, cv2.COLOR_RGBA2BGR)
    a = osd_bgra[..., 3:4].astype(np.uint16)
    out = (bgr.astype(np.uint16) * (255 - a) + osd_bgra[..., :3].astype(np.uint16) * 255 + 127) // 255
    return out.astype(np.uint8)


def _worker_main(cfg, link_args):
    """Worker process: sensors + ML pipeline, OSD layer, snapshots/recording."""
    from pipeline.detection_pipeline import DetectionPipeline
    from display.rescue_display import RescueDisplay

    # Video must never wait for ML: when cores are contended, let the display
    # process win. (Lowering priority needs no privileges.)
    try:
        os.nice(10)
    except OSError:
        pass

    # Ctrl-C reaches every process in the terminal's process group, and
    # `systemctl stop` SIGTERMs every process of the service. Treat both as a
    # stop request, so the inference thread is still joined before exit
    # (exiting mid-YOLO aborts the process) instead of dying on KeyboardInterrupt.
    stop_req = threading.Event()
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop_req.set())
    parent = os.getppid()     # changes if the display process dies — don't keep running orphaned

    link = _Link(cfg, args=link_args)
    W, H = cfg.display_width, cfg.display_height
    pipeline = DetectionPipeline(cfg, rgb=SharedLoresFeed(link, cfg.rgb_width, cfg.rgb_height))
    disp, inf_thread = None, None
    try:
        pipeline.initialize()
        state = {"fd": None, "failed": False}
        lock = threading.Lock()
        def inference():
            try:
                for fd in pipeline.run():
                    with lock:
                        state["fd"] = fd
            except Exception:
                logger.exception("[GPU] inference thread failed")
                state["failed"] = True
        inf_thread = threading.Thread(target=inference, daemon=True, name="inference")
        inf_thread.start()

        disp = RescueDisplay(cfg, window=False)
        comp = OverlayCompositor(W, H)
        feed = pipeline.rgb
        period = 1.0 / max(1.0, float(cfg.rgb_fps))
        next_t, last_log = time.perf_counter(), time.time()
        update_ms: deque = deque(maxlen=120)
        snap_request = False

        def grab_main() -> Optional[np.ndarray]:
            link.main_ready.clear()
            link.want_main.value = 1
            if not link.main_ready.wait(0.25):
                link.want_main.value = 0
                logger.warning("[GPU] No camera frame for snapshot/recording within 0.25 s — skipped")
                return None
            return link.main.copy()

        while not (link.stop.is_set() or stop_req.is_set()) and inf_thread.is_alive() \
                and os.getppid() == parent:
            while True:                                    # keys forwarded from the display window
                try:
                    if disp.handle_key(link.keys.get_nowait()) == "snapshot":
                        snap_request = True
                except queue.Empty:
                    break
            if disp.should_quit():
                link.quit.set()

            with lock:
                fd = state["fd"]
            if fd is not None:
                t0 = time.perf_counter()
                dl = disp.build_overlay(fd, feed.gray(), link.video_fps.value)
                regions = comp.update(dl.ops)
                if regions:
                    link.overlay_tx.send([(x0, y0, cv2.cvtColor(comp.canvas[y0:y1, x0:x1], cv2.COLOR_BGRA2RGBA))
                                          for x0, y0, x1, y1 in regions])

                # Snapshots and recordings need the picture as seen in the goggles:
                # grab the camera frame and blend the OSD over it on the I/O thread.
                paths = disp.auto_snapshot_paths(fd)
                if snap_request:
                    snap_request = False
                    paths.append(disp.manual_snapshot_path(fd))
                record = disp.record_due()
                if paths or record:
                    frame = grab_main()
                    if frame is not None:
                        osd = comp.canvas.copy()
                        for p in paths:
                            disp._enqueue_io("snapshot", lambda f=frame, o=osd, p=p: (p, _composite(f, o)))
                        if record:
                            disp._enqueue_io("record", lambda f=frame, o=osd: _composite(f, o))

                update_ms.append((time.perf_counter() - t0) * 1000)
                if time.time() - last_log >= 3.0:
                    last_log = time.time()
                    logger.info("[PERF] video=%.1ffps (GPU, every camera frame) | osd update median=%.1fms | "
                                "inference=%.1ffps", link.video_fps.value, float(np.median(update_ms)), fd.fps)

            next_t += period
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                next_t = time.perf_counter()
        if state["failed"]:
            raise SystemExit(1)      # non-zero exit code, so main.py (and systemd) see the failure
    except Exception:
        logger.exception("[GPU] worker failed")
        raise
    finally:
        # Stop + join the inference thread before teardown — exiting while
        # YOLO is mid-inference aborts the process.
        pipeline.stop()
        if inf_thread is not None:
            inf_thread.join(timeout=5.0)
        pipeline.shutdown()
        if disp is not None:
            disp.close()
        link.close()


# ── GL widget + camera (display process) ──────────────────────────────

def _view_class():
    from PyQt5.QtCore import Qt
    from OpenGL.GLES2 import (
        glBindTexture, glBlendFunc, glTexImage2D, glTexParameteri, glTexSubImage2D,
        GL_CLAMP_TO_EDGE, GL_LINEAR, GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_RGBA, GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
        GL_UNSIGNED_BYTE,
    )
    from picamera2.previews.q_gl_picamera2 import _get_qglpicamera2
    from picamera2.previews.qt_compatibility import _QT_BINDING

    class OsdGlView(_get_qglpicamera2(_QT_BINDING.PyQt5)):
        """Picamera2's zero-copy GL preview, plus a persistent OSD texture that
        is updated in place (instead of set_overlay(), which re-uploads the
        whole image and forces an extra buffer swap on every call)."""

        overlay_size = (1280, 720)
        on_key: Optional[Callable[[str], None]] = None

        def init_gl(self):
            super().init_gl()                       # leaves the GL context current
            self._pending: list = []
            self._pending_lock = threading.Lock()
            self._presents: deque = deque(maxlen=90)
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)   # OSD layer is premultiplied alpha
            w, h = self.overlay_size
            glBindTexture(GL_TEXTURE_2D, self.overlay_texture)
            for p, v in ((GL_TEXTURE_MAG_FILTER, GL_LINEAR), (GL_TEXTURE_MIN_FILTER, GL_LINEAR),
                         (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)):
                glTexParameteri(GL_TEXTURE_2D, p, v)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         np.zeros((h, w, 4), np.uint8))
            self.overlay_present = True

        def push_overlay(self, blocks):
            """Queue (x, y, RGBA block) uploads; applied before the next frame is drawn."""
            with self._pending_lock:
                self._pending.extend(blocks)

        def repaint(self, completed_request, update_viewport=False):
            # Called with the GL context current, right before the frame is drawn.
            with self._pending_lock:
                pending, self._pending = self._pending, []
            if pending:
                glBindTexture(GL_TEXTURE_2D, self.overlay_texture)
                for x, y, block in pending:
                    h, w = block.shape[:2]
                    glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, w, h, GL_RGBA, GL_UNSIGNED_BYTE, block)
            super().repaint(completed_request, update_viewport)

        def render_request(self, completed_request):
            super().render_request(completed_request)
            self._presents.append(time.perf_counter())

        def video_fps(self) -> float:
            p = list(self._presents)
            return (len(p) - 1) / (p[-1] - p[0]) if len(p) > 1 and p[-1] > p[0] else 0.0

        def keyPressEvent(self, event):
            ch = "\x1b" if event.key() == Qt.Key_Escape else event.text().lower()
            if ch and self.on_key:
                self.on_key(ch)

    return OsdGlView


def _paced_fps(requested: float, refresh: float) -> float:
    """Camera rate that divides the display refresh exactly — e.g. 29.995 on a
    59.99Hz HDMI mode, 25 on 50Hz PAL composite, 29.97 on NTSC — so every frame
    stays on screen for the same number of refreshes. (30fps into a 25fps PAL
    picture drops every 6th frame: a visible 5Hz judder.)"""
    if not refresh or refresh < 20:
        return requested
    return refresh / max(1, math.ceil(refresh / requested - 0.01))   # never faster than requested


class GpuDisplay:
    def __init__(self, cfg, link: _Link):
        ok, why = available()
        if not ok:
            raise GpuDisplayUnavailable(why)
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
        from PyQt5.QtCore import Qt, QTimer
        from PyQt5.QtWidgets import QApplication
        from picamera2 import Picamera2, MappedArray
        from sensors.rgb_camera import _camera_tuning

        self.cfg, self.link = cfg, link
        self.W, self.H = cfg.display_width, cfg.display_height
        self._Qt, self._QTimer, self._MappedArray = Qt, QTimer, MappedArray
        self._sig = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None

        self._app = QApplication.instance() or QApplication([])
        refresh = self._app.primaryScreen().refreshRate()
        self.fps = _paced_fps(float(cfg.rgb_fps), refresh)

        tuning = _camera_tuning(cfg)
        self.picam2 = Picamera2(tuning=Picamera2.load_tuning_file(tuning)) if tuning else Picamera2()
        try:
            self._setup(cfg, link)
        except Exception:
            # Release the camera so the OpenCV display fallback can open it.
            self.picam2.close()
            raise
        logger.info(f"[GPU] Zero-copy GL display ready | video {self.W}x{self.H} @ {self.fps:.3f}fps "
                    f"(paced to the {refresh:.2f}Hz display) | ML frames {cfg.rgb_width}x{cfg.rgb_height}")

    def _setup(self, cfg, link: _Link):
        from sensors.rgb_camera import _SENSOR_W, _SENSOR_H, _focus_controls
        self.picam2.configure(self.picam2.create_video_configuration(
            # Display stream: drawn by the GPU straight from the camera's buffers.
            main={"size": (self.W, self.H), "format": "XBGR8888"},
            # ISP-scaled frames for YOLO + OSD contrast sampling (the ISP must output YUV420 here).
            lores={"size": (cfg.rgb_width, cfg.rgb_height), "format": "YUV420"},
            # Full-frame sensor readout keeps the IMX708's PDAF data valid — see sensors/rgb_camera.py.
            raw={"size": (_SENSOR_W, _SENSOR_H)},
            controls={"FrameRate": self.fps, **_focus_controls(cfg), "NoiseReductionMode": 1},
            buffer_count=_BUFFER_COUNT, display="main",
        ))
        self.picam2.post_callback = self._on_frame

        View = _view_class()
        View.overlay_size = (self.W, self.H)
        self.view = View(self.picam2, width=self.W, height=self.H, keep_ar=True)
        self.view.setWindowTitle("Rescue Drone  |  Human Detection")
        self.view.on_key = lambda ch: link.keys.put_nowait(ch)

    # Runs on the Qt thread for every camera frame — keep it tiny.
    def _on_frame(self, request):
        link = self.link
        with self._MappedArray(request, "lores") as m:
            with link.lores_lock:
                link.lores[:] = m.array
        link.lores_ts.value = time.time()
        link.lores_seq.value += 1
        if link.want_main.value:
            with self._MappedArray(request, "main") as m:
                link.main[:] = m.array
            link.want_main.value = 0
            link.main_ready.set()

    def _receive_overlay(self):
        rx = self.link.overlay_rx
        while not self.link.stop.is_set():
            try:
                if rx.poll(0.2):
                    self.view.push_overlay(rx.recv())
            except (EOFError, OSError):
                return

    def run(self, alive: Callable[[], bool]):
        """Show the video until Q/Esc, SIGINT/SIGTERM, or the worker exits."""
        if self.cfg.fullscreen:
            self.view.showFullScreen()
            self.view.setCursor(self._Qt.BlankCursor)
        else:
            self.view.show()
        self.picam2.start()
        self._rx_thread = threading.Thread(target=self._receive_overlay, daemon=True, name="osd-rx")
        self._rx_thread.start()

        # Qt's C++ event loop never hands control back for KeyboardInterrupt, so
        # signals just set a flag that a 100ms timer turns into a clean quit.
        prev = {s: signal.signal(s, lambda *_: self._sig.set()) for s in (signal.SIGINT, signal.SIGTERM)}
        def tick():
            self.link.video_fps.value = self.view.video_fps()
            if self._sig.is_set() or self.link.quit.is_set() or not alive():
                self._app.quit()
        timer = self._QTimer()
        timer.timeout.connect(tick)
        timer.start(100)
        try:
            self._app.exec_()
        finally:
            timer.stop()
            for s, h in prev.items():
                signal.signal(s, h)

    def close(self):
        for fn in (self.picam2.stop, self.view.close, self.picam2.close):
            try:
                fn()
            except Exception as e:
                logger.debug(f"[GPU] cleanup: {e}")
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)


def run_gpu(cfg, present_hook: Optional[Callable] = None) -> int:
    """Run the whole system with the GPU display. Raises GpuDisplayUnavailable
    if the display can't start (caller can fall back to the OpenCV display).
    Returns the worker's exit code."""
    from display.screen import ScreenBlankingInhibitor

    ctx = mp.get_context("spawn")
    link = _Link(cfg, ctx=ctx)
    # Start the worker before Picamera2/Qt/GL exist in this process.
    worker = ctx.Process(target=_worker_main, args=(cfg, link.args()), name="ml-osd")
    worker.start()

    def stop_worker():
        link.stop.set()
        worker.join(timeout=8.0)
        if worker.is_alive():
            logger.warning("[GPU] worker did not stop in time — terminating")
            worker.terminate()
            worker.join(timeout=2.0)

    gpu = None
    try:
        gpu = GpuDisplay(cfg, link)
        if tuple(gpu.picam2.camera_config["lores"]["size"]) != (cfg.rgb_width, cfg.rgb_height) or \
           gpu.picam2.stream_map["lores"].configuration.stride != link.lores_shape[1]:
            raise GpuDisplayUnavailable("unexpected lores stream layout")
    except Exception as e:
        if gpu is not None:
            gpu.close()          # free the camera for the OpenCV display fallback
        stop_worker()
        link.close()
        raise e if isinstance(e, GpuDisplayUnavailable) else GpuDisplayUnavailable(str(e)) from e

    if present_hook is not None:
        present_hook(gpu)
    screen = ScreenBlankingInhibitor()
    screen.start()       # an idle desktop blanks after 10 min — black goggles mid-flight
    try:
        gpu.run(worker.is_alive)
    finally:
        stop_worker()
        gpu.close()
        screen.restore()
        link.close()
    return worker.exitcode or 0
