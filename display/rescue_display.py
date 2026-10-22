"""
Rescue Display — integrates ThermalIsolator and OSD overlay.

Full-screen : Selectable via `V` key (RGB, Thermal, Radar)
PiP (toggle) : thermal view with human-only isolation in bottom-left

Key controls:
  V — cycle main view (RGB / Thermal / Radar)
  T — toggle Thermal PiP
  M — cycle thermal display mode (highlight / silhouette / contour)
  S — save snapshot
  Q — quit

Two ways to put this on screen share the same layout code (a DrawList):
  * render()        — OpenCV window: the OSD is drawn into each video frame.
  * build_overlay() — GPU display (display/gpu_display.py): the OSD becomes a
                      separate transparent layer over hardware-composited video.
"""

import cv2
import numpy as np
import time
import logging
import threading
import queue
from typing import Optional, List

from pipeline.detection_pipeline import FrameData
from ml.detection import Detection
from ml.thermal_isolation import ThermalIsolator
from display.osd import OSDRenderer, DrawList, luma_grid

logger = logging.getLogger("display")

BG_DARK   = (15,  12,  10)
BG_PANEL  = (23,  17,  13)
ORANGE    = (0,   140, 255)
BLUE      = (220, 140, 50)
GREEN     = (80,  220, 60)
RED       = (30,  30,  210)
CYAN      = (200, 220, 50)
WHITE     = (235, 235, 235)
GRAY      = (120, 120, 120)
GRAY_DIM  = (65,  65,  65)
BORDER    = (40,  35,  30)
CHIP_BG   = (35,  28,  22)
FONT      = cv2.FONT_HERSHEY_DUPLEX

THERMAL_MODES = ["highlight", "silhouette", "contour"]

# The recording VideoWriter is opened at this rate, so frames are handed to it
# at this rate too — writing every display frame (up to 30/s) into an 8fps
# file made recordings play back several times too fast.
RECORD_FPS = 8

def _txt(img, text, pos, scale=0.48, color=WHITE, thickness=1, shadow=True):
    if shadow:
        cv2.putText(img, text, (pos[0]+1, pos[1]+1),
                    FONT, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)

def _chip(img, text, x, y, color, w=None):
    tw, th = cv2.getTextSize(text, FONT, 0.38, 1)[0]
    pw = w if w else tw + 14; ph = th + 8
    cv2.rectangle(img, (x, y), (x+pw, y+ph), CHIP_BG, -1)
    cv2.rectangle(img, (x, y), (x+pw, y+ph), BORDER, 1)
    cv2.putText(img, text, (x+(pw-tw)//2, y+ph-4), FONT, 0.38, color, 1, cv2.LINE_AA)
    return pw + 6

def _corner_box(img, x1, y1, x2, y2, col, t=2):
    cv2.rectangle(img, (x1,y1), (x2,y2), col, 1)
    L = max(10, min(20, (x2-x1)//5))
    for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img,(px,py),(px+dx*L,py),col,t,cv2.LINE_AA)
        cv2.line(img,(px,py),(px,py+dy*L),col,t,cv2.LINE_AA)


class RescueDisplay:
    def __init__(self, cfg, window: bool = True):
        self.cfg          = cfg
        self.W            = cfg.display_width
        self.H            = cfg.display_height
        self._WIN         = "Rescue Drone  |  Human Detection"
        self._window      = window
        self._writer: Optional[cv2.VideoWriter] = None
        self._quit        = False
        self._mode_idx    = 0
        self._show_thermal = True
        self._main_view_idx = 0  # 0: RGB, 1: Thermal, 2: Radar

        self._isolator = ThermalIsolator(cfg)

        self._last_snapshot_time = {} # track_id -> timestamp
        self._last_record = 0.0
        self._osd      = OSDRenderer(cfg)

        # ── Box motion extrapolation ──────────────────────────────────
        # Inference (tracker/YOLO) only produces a new set of boxes at the
        # thermal-gated inference rate (~1-2fps), but this display renders up
        # to 30fps. Without this, a box sits frozen for many video frames
        # then jumps to its next position — jerky against the smooth video.
        # We keep the last two inference cycles' boxes per track_id and
        # linearly project each box forward using its own recent velocity,
        # so it glides continuously instead of holding-then-snapping.
        self._prev_dets: dict = {}   # track_id -> Detection (previous inference cycle)
        self._prev_ts:   float = 0.0
        self._curr_dets: dict = {}   # track_id -> Detection (latest inference cycle)
        self._curr_ts:   float = 0.0
        self._last_frame_id: int = -1

        # ThermalIsolator runs on its own worker thread so a new thermal frame
        # never stalls the 30fps display loop — see _thermal_pane(). The
        # display keeps showing the previous isolated frame until the next
        # one is ready (thermal only updates ~1-2x/s anyway).
        self._iso_lock = threading.Lock()
        self._iso_event = threading.Event()
        self._iso_job = None                      # newest (key, raw, visual, mode) awaiting the worker
        self._iso_pending_key = None
        self._iso_key = None                      # key of the newest finished isolated frame
        self._iso_val: Optional[np.ndarray] = None
        self._pane_cache: dict = {}               # finished (pane, version), reused until the source changes
        self._pane_version = 0
        self._radar_bg_cache = None               # ((w, h), static radar scope image)
        threading.Thread(target=self._iso_worker, daemon=True, name="thermal-iso").start()

        # Rolling video-FPS tracker (display render rate)
        self._vfps_buf: list = []
        self._vfps_last: float = time.time()
        self._video_fps: float = 0.0

        # ── Background disk I/O ─────────────────────────────────────────
        # cv2.imwrite() (snapshots) and VideoWriter.write() (recording) can
        # each cost tens of ms (JPEG/mp4v encode + SD card write). Doing
        # that synchronously inside render() stalls the live display right
        # when a human is detected — exactly when smoothness matters most.
        # A bounded queue + worker thread keeps the display loop non-
        # blocking; if the writer falls behind, we drop the oldest pending
        # job rather than ever blocking render().
        self._io_queue: "queue.Queue" = queue.Queue(maxsize=8)
        self._io_thread = threading.Thread(
            target=self._io_worker, daemon=True, name="display-io"
        )
        self._io_thread.start()

        if window:
            cv2.namedWindow(self._WIN, cv2.WINDOW_NORMAL)
            if getattr(cfg, 'fullscreen', True):
                cv2.setWindowProperty(self._WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.resizeWindow(self._WIN, self.W, self.H)
        if cfg.record:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(cfg.output_video, fourcc, RECORD_FPS, (self.W, self.H))

    def _io_worker(self):
        """Runs on its own thread: drains (kind, payload) jobs and does the
        actual disk I/O, so render() never blocks on it."""
        while True:
            job = self._io_queue.get()
            if job is None:
                break
            kind, payload = job
            try:
                if callable(payload):         # deferred image (GPU path composes off the display thread)
                    payload = payload()
                if kind == "snapshot":
                    path, img = payload
                    cv2.imwrite(path, img)
                    logger.info(f"Snapshot saved: {path}")
                elif kind == "record" and self._writer:
                    self._writer.write(payload)
            except Exception as e:
                logger.warning(f"[display-io] Write failed: {e}")

    def _enqueue_io(self, kind: str, payload):
        try:
            self._io_queue.put_nowait((kind, payload))
        except queue.Full:
            logger.warning(f"[display-io] Queue full — dropping {kind} job")

    # ── Recording / snapshots / keys (shared by both display paths) ─────

    def record_due(self) -> bool:
        """True when the next recording frame is due (at RECORD_FPS)."""
        if not self._writer:
            return False
        now = time.time()
        if now - self._last_record < 1.0 / RECORD_FPS:
            return False
        self._last_record = now
        return True

    def auto_snapshot_paths(self, fd: FrameData) -> List[str]:
        """Paths of auto-snapshots due now (once per snapshot_cooldown per track)."""
        paths = []
        if self.cfg.snapshot_enabled and getattr(fd, "tracked_humans", None):
            now = time.time()
            for t in fd.tracked_humans:
                last = self._last_snapshot_time.get(t.track_id, 0)
                if now - last >= self.cfg.snapshot_cooldown:
                    paths.append(f"{self.cfg.snapshot_dir}/auto_snap_TRK{t.track_id}_{fd.frame_id:05d}.jpg")
                    self._last_snapshot_time[t.track_id] = now
            # Keyed by track_id, which keeps climbing over a long mission —
            # bound it rather than leak indefinitely (same pattern as
            # OSDRenderer._luma_ema and AnomalyDetector._conf_ema).
            if len(self._last_snapshot_time) > 256:
                self._last_snapshot_time.clear()
        return paths

    def manual_snapshot_path(self, fd: FrameData) -> str:
        return f"{self.cfg.snapshot_dir}/snap_{fd.frame_id:05d}.jpg"

    def handle_key(self, ch: str) -> Optional[str]:
        """Apply a key press. Returns "snapshot" when the caller should save one."""
        if ch in ("q", "\x1b"):
            self._quit = True
        elif ch == "s":
            return "snapshot"
        elif ch == "m":
            self._mode_idx = (self._mode_idx + 1) % len(THERMAL_MODES)
            logger.info(f"Thermal mode: {THERMAL_MODES[self._mode_idx]}")
        elif ch == "t":
            self._show_thermal = not self._show_thermal
        elif ch == "v":
            self._main_view_idx = (self._main_view_idx + 1) % 3
        return None

    # ── OpenCV display path ─────────────────────────────────────────────

    def render(self, fd: FrameData):
        # -- Video FPS (rolling 30-frame average) --
        now_t = time.time()
        elapsed = now_t - self._vfps_last
        self._vfps_last = now_t
        if elapsed > 0:
            self._vfps_buf.append(elapsed)
            if len(self._vfps_buf) > 30:
                self._vfps_buf.pop(0)
            self._video_fps = 1.0 / (sum(self._vfps_buf) / len(self._vfps_buf))

        canvas = self._build(fd)
        if self.record_due(): self._enqueue_io("record", canvas)
        cv2.imshow(self._WIN, canvas)

        for p in self.auto_snapshot_paths(fd):
            self._enqueue_io("snapshot", (p, canvas))

        key = cv2.waitKey(1) & 0xFF
        if key != 255 and self.handle_key(chr(key).lower()) == "snapshot":
            self._enqueue_io("snapshot", (self.manual_snapshot_path(fd), canvas))

    def should_quit(self): return self._quit

    @property
    def video_fps(self) -> float:
        """Rolling measured display render rate (what's actually on screen)."""
        return self._video_fps

    def close(self):
        # Let any pending snapshot/recording jobs drain before releasing the
        # writer — otherwise the io thread could call write() on a writer
        # that's already been released.
        self._io_queue.put(None)
        self._io_thread.join(timeout=3.0)
        if self._writer: self._writer.release()
        if self._window:
            cv2.destroyAllWindows()

    def _osd_kwargs(self, fd: FrameData) -> dict:
        rd = fd.radar
        return dict(
            num_humans     = fd.num_humans,
            total_unique   = getattr(fd, "total_unique_humans", 0),
            radar_present  = rd is not None and rd.human_present,
            radar_state    = rd.target_state    if rd else 0,
            radar_dist_cm  = rd.detect_dist_cm  if rd else 0,
            radar_strength = max(rd.moving_energy, rd.static_energy) if rd else 0,
            fps            = fd.fps,
            frame_id       = fd.frame_id,
            recording      = self.cfg.record,
            sensor_ok      = fd.sensor_status,
            yolo_backend   = fd.yolo_backend,
            total_ms       = fd.timing_ms.get("total_ms", 0),
            fc_telemetry   = fd.fc_telemetry,
            anomaly_triggered = getattr(fd, "anomaly_triggered", False),
            inference_fps  = fd.fps,
            video_fps      = self._video_fps,
        )

    def _layout(self, fd: FrameData, dl: DrawList, gray_small: np.ndarray, dets: list):
        """OSD + thermal PiP + battery readout, appended to `dl` in draw order."""
        # Persistent OSD over ANY view
        self._osd.layout(dl, gray_small, self.W, self.H, dets, **self._osd_kwargs(fd))

        # Overlay Thermal PiP if NOT Main View and PiP is enabled
        if self._show_thermal and self._main_view_idx != 1:
            pip_x, pip_y, pip_w, pip_h = self._osd.pip_rect(self.W, self.H)
            # Composite: no colour scale / tiny labels — unreadable at that
            # size on analog video.
            clean = self._osd.style.name == "composite"
            thermal_pip, ver = self._thermal_pane_v(fd, pip_w, pip_h, clean=clean)
            dl.rectangle((pip_x-2, pip_y-2), (pip_x+pip_w+1, pip_y+pip_h+1), BORDER, 2)
            dl.image(thermal_pip, pip_x, pip_y, ("pane", ver))

            # ── Battery info displayed ABOVE the PiP window ───────────
            self._osd.battery(dl, gray_small, self.W, self.H, fd.fc_telemetry or {})

    def _build(self, fd: FrameData) -> np.ndarray:
        if self._main_view_idx == 0:
            base_layers = self._get_rgb_base(fd, self.W, self.H)
        elif self._main_view_idx == 1:
            base_layers = self._thermal_pane(fd, self.W, self.H, clean=True)
        else:
            base_layers = self._radar_bg(self.W, self.H)

        self._roll_detection_window(fd)
        scaled = self._extrapolated_dets()

        canvas = base_layers.copy()
        dl = DrawList()
        if self._main_view_idx == 2:
            self._radar_ops(dl, fd, self.W, self.H)
        self._layout(fd, dl, self._osd.luma_grid_bgr(canvas), scaled)
        dl.draw(canvas)
        return canvas

    # ── GPU display path ────────────────────────────────────────────────

    def build_overlay(self, fd: FrameData, video_gray: Optional[np.ndarray], video_fps: float) -> DrawList:
        """The OSD as draw operations for a transparent layer over hardware-
        composited video. `video_gray` is the camera's luma (any size) for
        adaptive text contrast; the thermal / radar views are opaque layers."""
        self._video_fps = video_fps
        self._roll_detection_window(fd)
        scaled = self._extrapolated_dets()

        dl = DrawList()
        if self._main_view_idx == 0:
            gray_small = luma_grid(video_gray) if video_gray is not None \
                else np.zeros((36, 64), np.uint8)
        elif self._main_view_idx == 1:
            pane, ver = self._thermal_pane_v(fd, self.W, self.H, clean=True)
            dl.image(pane, 0, 0, ("pane", ver))
            gray_small = self._osd.luma_grid_bgr(pane)
        else:
            bg = self._radar_bg(self.W, self.H)
            dl.image(bg, 0, 0, ("radar_bg", self.W, self.H))
            self._radar_ops(dl, fd, self.W, self.H)
            gray_small = self._osd.luma_grid_bgr(bg)

        self._layout(fd, dl, gray_small, scaled)
        return dl

    # ── Box motion extrapolation ────────────────────────────────────

    _MAX_EXTRAPOLATION_ALPHA = 1.5   # cap projection to 1.5x one inference cycle

    def _roll_detection_window(self, fd: FrameData):
        """Whenever a NEW inference result arrives (frame_id changes), shift
        the current cycle's boxes into 'previous' and record the new ones as
        'current', keyed by track_id. Between inference cycles this is a
        no-op — the display just keeps extrapolating from the same window."""
        if fd.frame_id == self._last_frame_id:
            return
        self._prev_dets, self._prev_ts = self._curr_dets, self._curr_ts
        self._curr_dets = {d.track_id: d for d in fd.tracked_humans if d.track_id is not None}
        self._curr_ts = time.time()
        self._last_frame_id = fd.frame_id

    def _extrapolated_dets(self) -> list:
        """Project each track's box forward from its last two known
        positions using its own recent velocity, evaluated at the current
        wall-clock time. Gives smooth continuous motion at full display fps
        instead of holding a box still for a whole inference cycle."""
        now = time.time()
        dt_cycle = self._curr_ts - self._prev_ts
        out = []
        for tid, cur in self._curr_dets.items():
            prev = self._prev_dets.get(tid)
            if prev is None or dt_cycle <= 1e-3:
                out.append(cur)
                continue
            alpha = (now - self._curr_ts) / dt_cycle
            alpha = max(0.0, min(alpha, self._MAX_EXTRAPOLATION_ALPHA))
            out.append(Detection(
                x1=cur.x1 + (cur.x1 - prev.x1) * alpha,
                y1=cur.y1 + (cur.y1 - prev.y1) * alpha,
                x2=cur.x2 + (cur.x2 - prev.x2) * alpha,
                y2=cur.y2 + (cur.y2 - prev.y2) * alpha,
                confidence=cur.confidence, source=cur.source,
                track_id=cur.track_id, temp_celsius=cur.temp_celsius,
            ))
        return out

    # ── Thermal pane ──────────────────────────────────────────────

    def _iso_worker(self):
        """Runs ThermalIsolator off the display thread. Only the newest job
        is kept — if thermal frames arrive faster than this finishes, the
        stale ones are skipped rather than queued."""
        while True:
            self._iso_event.wait()
            with self._iso_lock:
                job, self._iso_job = self._iso_job, None
                self._iso_event.clear()
            if job is None:
                continue
            key, raw, visual, mode = job
            try:
                val = self._isolator.process(raw, visual, mode)[0]
            except Exception as e:
                logger.warning(f"[display] Thermal isolation failed: {e}")
                val = visual
            with self._iso_lock:
                self._iso_key, self._iso_val = key, val

    def _thermal_pane(self, fd: FrameData, pw: int, ph: int, clean: bool = False) -> np.ndarray:
        """Returns a CACHED pane — callers must copy before drawing on it."""
        return self._thermal_pane_v(fd, pw, ph, clean)[0]

    def _thermal_pane_v(self, fd: FrameData, pw: int, ph: int, clean: bool = False):
        """(pane, version) — the version changes whenever the pane's pixels do."""
        mode = THERMAL_MODES[self._mode_idx]

        if fd.thermal_visual is not None and fd.thermal_raw is not None:
            # ThermalIsolator.process() only needs to run once per new thermal
            # frame (and per mode), never per display tick — re-applying it to
            # the SAME frame would also skew its slow background EMA. Hand new
            # frames to the worker and keep showing the newest finished result.
            key = (fd.frame_id, mode)
            with self._iso_lock:
                if key != self._iso_key and key != self._iso_pending_key:
                    self._iso_job = (key, fd.thermal_raw, fd.thermal_visual, mode)
                    self._iso_pending_key = key
                    self._iso_event.set()
                src_key, src = self._iso_key, self._iso_val
            if src is None:                       # first frame: worker not done yet
                src_key, src = ("plain", fd.frame_id), fd.thermal_visual
        elif fd.thermal_visual is not None:
            src_key, src = ("plain", fd.frame_id), fd.thermal_visual
        else:
            src_key, src = ("offline",), None

        # Resizing + the per-row colour scale bar cost ~5ms; the source only
        # changes ~1-2x/s, so build each pane once per source and reuse it.
        pkey = (src_key, pw, ph, clean)
        hit = self._pane_cache.get(pkey)
        if hit is not None:
            return hit

        if src is None:
            pane = np.full((ph, pw, 3), BG_DARK, dtype=np.uint8)
            _txt(pane, "THERMAL OFFLINE", (pw//2-60, ph//2), 0.5, GRAY_DIM)
        else:
            pane = cv2.resize(src, (pw, ph), interpolation=cv2.INTER_LINEAR)
            if not clean:
                self._temp_scale(pane, pw, ph)

                # Show the dynamic span on PiP
                _txt(pane, f"THERMAL: {mode.upper()}", (5, ph - 10), 0.35, GRAY_DIM, shadow=True)

        self._pane_version += 1
        self._pane_cache = {k: v for k, v in self._pane_cache.items() if k[0] == src_key}
        self._pane_cache[pkey] = (pane, self._pane_version)
        return pane, self._pane_version

    def _thermal_box(self, pane, det, tw, th, sw, sh):
        sx, sy = tw/max(sw,1), th/max(sh,1)
        x1,y1,x2,y2 = int(det.x1*sx),int(det.y1*sy),int(det.x2*sx),int(det.y2*sy)
        _corner_box(pane, x1, y1, x2, y2, ORANGE, t=1)

    def _temp_scale(self, pane, pw, ph):
        bx,by,bws,bhs = pw-15, 10, 5, ph-20
        for i in range(bhs):
            t = 1.0 - i/bhs
            if t<0.30: s=t/0.30;b=int(80+s*100);g=int(10+s*50);r=int(5+s*25)
            elif t<0.55: s=(t-0.30)/0.25;b=int(180-s*20);g=int(60+s*120);r=20
            elif t<0.75: s=(t-0.55)/0.20;b=int(200-s*180);g=180;r=int(20+s*180)
            elif t<0.90: s=(t-0.75)/0.15;b=20;g=int(180-s*80);r=int(200+s*55)
            else: s=(t-0.90)/0.10;b=int(s*220);g=int(100+s*155);r=255
            cv2.line(pane,(bx,by+i),(bx+bws,by+i),(b,g,r),1)
        cv2.rectangle(pane,(bx,by),(bx+bws,by+bhs),BORDER,1)

        # Dynamic Range T_min / T_max drawing
        t_min = f"{self.cfg.thermal_min_temp:.1f}C"
        t_max = f"{self.cfg.thermal_max_temp:.1f}C"
        _txt(pane, t_max, (bx - 35, by + 10), 0.3, WHITE)
        _txt(pane, t_min, (bx - 35, by + bhs), 0.3, WHITE)


    # ── RGB pane ──────────────────────────────────────────────────

    def _get_rgb_base(self, fd: FrameData, pw: int, ph: int) -> np.ndarray:
        if fd.rgb_frame is not None:
            self._last_rgb_frame = cv2.resize(fd.rgb_frame, (pw, ph))

        if hasattr(self, "_last_rgb_frame") and self._last_rgb_frame is not None:
            base = self._last_rgb_frame
        else:
            base = np.full((ph, pw, 3), BG_DARK, dtype=np.uint8)
            _txt(base, "RGB CAM WARMING UP...", (pw // 2 - 110, ph // 2),
                 0.55, GRAY_DIM, shadow=False)

        return base

    # ── Radar pane ────────────────────────────────────────────────

    def _radar_bg(self, pw: int, ph: int) -> np.ndarray:
        """Static part of the radar scope (background, rings, axes) — cached.
        The moving parts are separate draw ops (_radar_ops), so the GPU path
        only redraws/uploads where the sweep and targets are, not the screen."""
        key = (pw, ph)
        if self._radar_bg_cache is not None and self._radar_bg_cache[0] == key:
            return self._radar_bg_cache[1]
        base = np.full((ph, pw, 3), (10, 20, 15), dtype=np.uint8) # Dark greenish
        cx, cy = pw // 2, ph // 2

        # Concentric Rings
        for r in range(100, min(pw, ph) // 2 + 100, 100):
            cv2.circle(base, (cx, cy), r, (0, 80, 0), 1, cv2.LINE_AA)

        cv2.line(base, (cx, 0), (cx, ph), (0, 80, 0), 1)
        cv2.line(base, (0, cy), (pw, cy), (0, 80, 0), 1)
        self._radar_bg_cache = (key, base)
        return base

    def _radar_ops(self, dl: DrawList, fd: FrameData, pw: int, ph: int):
        """Moving parts of the radar scope: the sweep and the targets."""
        cx, cy = pw // 2, ph // 2

        # Sweeping Line
        now = time.time()
        angle = (now % 2.0) / 2.0 * 2 * np.pi
        length = min(pw, ph) // 2
        ex = int(cx + length * np.cos(angle))
        ey = int(cy + length * np.sin(angle))
        dl.line((cx, cy), (ex, ey), (0, 255, 0), 2)

        # Every target at its real LD2450 position (radar at the centre,
        # straight ahead = up) — not at an angle tied to the decorative sweep.
        rd = fd.radar
        if rd:
            px_per_mm = length / max(1, self.cfg.ld2410_max_range_cm * 10)
            for tgt in rd.targets:
                tx = int(cx + tgt.x_mm * px_per_mm)
                ty = int(cy - tgt.y_mm * px_per_mm)
                dist = int(tgt.distance_cm)
                if abs(tgt.speed_cm_s) >= 10:   # Moving (same threshold as PresenceData)
                    dl.circle((tx, ty), 10, (0, 0, 255), -1)
                    dl.cv_text(f"MOVING {dist}cm", (tx+15, ty), 0.5, WHITE)
                else:                          # Stationary
                    dl.rectangle((tx-8, ty-8), (tx+8, ty+8), (0, 255, 255), -1)
                    dl.cv_text(f"STAT {dist}cm", (tx+15, ty), 0.5, WHITE)
