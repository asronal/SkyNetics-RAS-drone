"""
Detection Pipeline v3 — Cascade for RPi4 + MLX90640 + LD2410C-P

One detection cycle per MLX90640 frame. Measured on the Pi 4 (100 kHz I2C):
  MLX90640 frame      ~520ms   own thread, back-to-back — 85% of it is waiting
                               on I2C, so it overlaps everything below
  YOLOv8n ONNX        ~300ms
  Anomaly + fusion + tracking ~5ms
  → ~1.9 cycles/s, set by the sensor (2 frames/s at its 4 Hz subpage rate).
    Reading the sensor inline after YOLO gave 1.0/s: the read also waited out
    a random part of the sensor's cycle (~725ms per read).
"""

import logging
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Iterator

from config import Config
from sensors.thermal_camera import ThermalCamera
from sensors.rgb_camera import RGBCamera
from sensors.ld2450_radar import LD2450Sensor, PresenceData
from sensors.flight_controller import FlightController
from ml.models import AnomalyDetector, YOLODetector, SensorFusion, HumanTracker
from ml.detection import Detection

logger = logging.getLogger("pipeline")

# Cycle period when the thermal sensor is offline (its frames normally gate
# each cycle; ~0.52 s per read at 4 Hz).
_NO_THERMAL_PERIOD_S = 0.5


@dataclass
class FrameData:
    timestamp: float
    frame_id: int
    thermal_raw: Optional[np.ndarray]
    thermal_visual: Optional[np.ndarray]
    rgb_frame: Optional[np.ndarray]
    radar: Optional[PresenceData]
    tracked_humans: List[Detection]
    anomaly_triggered: bool
    yolo_ran: bool
    timing_ms: dict
    fps: float
    sensor_status: dict
    yolo_backend: str
    fc_telemetry: dict = field(default_factory=dict)
    total_unique_humans: int = 0

    @property
    def num_humans(self) -> int:
        """People the camera AI confirmed (heat-only blobs excluded)."""
        return sum(not d.heat_only for d in self.tracked_humans)

    @property
    def num_heat(self) -> int:
        return sum(d.heat_only for d in self.tracked_humans)

    @property
    def radar_confirms(self) -> bool:
        r = self.radar
        return r is not None and r.human_present

    @property
    def stationary_detected(self) -> bool:
        """True if a stationary target (buried survivor) detected by radar."""
        r = self.radar
        return r is not None and r.stationary_human


class DetectionPipeline:
    def __init__(self, cfg: Config, rgb=None):
        self.cfg = cfg
        self.thermal  = ThermalCamera(cfg)
        # `rgb` lets another component own the camera and feed frames in —
        # the GPU display does, since it needs the camera for zero-copy video.
        self.rgb      = rgb if rgb is not None else RGBCamera(cfg)
        self.radar    = LD2450Sensor(cfg)
        self.fc       = FlightController(cfg)
        self.anomaly  = AnomalyDetector(cfg)
        self.yolo     = YOLODetector(cfg)
        self.fusion   = SensorFusion(cfg)
        self.tracker  = HumanTracker(cfg)
        self._fid     = 0
        self._fps_buf: list = []
        self._last_alert = 0.0
        self._last_yolo_dets: List[Detection] = []
        self._total_unique: int = 0
        # FIX: cache the last good RGB frame so the display never goes
        # black during the camera warmup period or during
        # a watchdog restart. Without this, every frame where rgb.read()
        # returns None renders a solid black pane.
        self._last_rgb: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._last_thermal_ok = time.monotonic()
        self._last_thermal_warn = 0.0
        self._last_yield: Optional[float] = None
        # Newest (raw, visual, read_ms) from the thermal thread; None once consumed.
        self._thermal_cv = threading.Condition()
        self._thermal_frame: Optional[tuple] = None
        self._thermal_thread: Optional[threading.Thread] = None

    def initialize(self):
        ok_t = self.thermal.initialize()
        if self.thermal.is_online:
            self._thermal_thread = threading.Thread(
                target=self._thermal_loop, daemon=True, name="thermal"
            )
            self._thermal_thread.start()
        self.rgb.initialize()
        self.radar.initialize()
        self.fc.initialize()
        self.anomaly.load()
        self.yolo.load()

        if not ok_t and not self.cfg.demo_mode:
            # Keep searching with the RGB camera alone: the display shows
            # THERMAL OFFLINE, and the cycles are paced by run() instead.
            logger.error("[PIPELINE] Thermal camera OFFLINE — continuing with RGB detection only")

        logger.info(
            f"[PIPELINE] Ready\n"
            f"  Thermal  : {'online' if self.thermal.is_online else 'OFFLINE'}\n"
            f"  RGB      : {'online' if self.rgb.is_online else 'OFFLINE'}\n"
            f"  LD2410   : {'online' if self.radar.is_online else 'OFFLINE'}\n"
            f"  FC       : {'online' if self.fc.is_online else ('OFFLINE' if self.cfg.fc_enabled else 'disabled')}\n"
            f"  YOLO     : {self.yolo.backend}"
        )

    def shutdown(self):
        # The thermal thread owns the sensor — let it finish its read first.
        self._stop.set()
        if self._thermal_thread is not None:
            self._thermal_thread.join(timeout=2.0)
        self.thermal.release()
        self.rgb.release()
        self.radar.shutdown()
        self.fc.shutdown()
        logger.info("[PIPELINE] All sensors shut down")

    def stop(self):
        """Make run() return after its current cycle. Callers should then join
        the thread iterating run() before shutdown(): letting the interpreter
        exit while YOLO is mid-inference aborts the process (onnxruntime
        "terminate called without an active exception")."""
        self._stop.set()

    def _thermal_loop(self):
        """Reads the MLX90640 back-to-back on its own thread. This thread is the
        only one touching the sensor (Blinka's I2C is not thread-safe), and
        reading continuously keeps getFrame() in step with the sensor's
        subpages, while YOLO runs on the pipeline thread in parallel."""
        # Synthetic frames are instant — pace them like the real sensor.
        pace = 1.0 / max(1, self.cfg.thermal_fps) if self.cfg.demo_mode else 0.0
        while not self._stop.is_set():
            t = time.perf_counter()
            raw, visual = self.thermal.read()
            if visual is None:
                # Rate-limited or an I2C error (ThermalCamera backs off itself).
                time.sleep(0.02)
                continue
            with self._thermal_cv:
                self._thermal_frame = (raw, visual, (time.perf_counter() - t) * 1000)
                self._thermal_cv.notify()
            if pace:
                time.sleep(max(0.0, pace - (time.perf_counter() - t)))

    def _next_thermal(self) -> Optional[tuple]:
        """(raw, visual, read_ms) from the thermal thread, or None after a
        0.25 s wait. Without a thermal sensor: (None, None, 0.0), paced like
        the real sensor so YOLO's duty cycle (and the display) stays the same."""
        if self._thermal_thread is None:
            self._stop.wait(_NO_THERMAL_PERIOD_S)
            return None, None, 0.0
        with self._thermal_cv:
            if self._thermal_frame is None:
                self._thermal_cv.wait(timeout=0.25)
            frame, self._thermal_frame = self._thermal_frame, None
        return frame

    def run(self) -> Iterator[FrameData]:
        while not self._stop.is_set():
            # 1. Wait for the next thermal frame — it's the gate for one cycle.
            # RGB/radar are only read once we know this cycle will be used.
            frame = self._next_thermal()
            if frame is None:
                # Only warn if we've had no frame for 3+ seconds.
                now = time.monotonic()
                # Warn every 5s while it lasts — not on every poll.
                if now - self._last_thermal_ok > 3.0 and now - self._last_thermal_warn > 5.0:
                    self._last_thermal_warn = now
                    logger.warning(
                        f"[PIPELINE] No thermal frame for {now - self._last_thermal_ok:.0f}s — check I2C / sensor"
                    )
                continue
            self._last_thermal_ok = time.monotonic()

            t0 = time.perf_counter()
            raw, visual, thermal_ms = frame
            # Time the read took on the thermal thread (overlaps this cycle).
            timing = {"thermal_ms": thermal_ms}

            t = time.perf_counter()
            rgb_new = self.rgb.read()
            # FIX: cache last good RGB frame — prevents black pane during
            # camera warmup and watchdog restart gaps.
            if rgb_new is not None:
                self._last_rgb = rgb_new
            rgb = self._last_rgb
            radar_data = self.radar.get_presence()
            timing["sensors_ms"] = (time.perf_counter() - t) * 1000

            # 2. Anomaly detector — every frame
            t = time.perf_counter()
            if raw is None:   # thermal offline
                anomaly_dets, triggered = [], False
            else:
                anomaly_dets, triggered = self.anomaly.detect(raw, visual)
            timing["anomaly_ms"] = (time.perf_counter() - t) * 1000

            # 3. YOLOv8 — Decoupled from thermal trigger for maximum RGB face detection
            t = time.perf_counter()
            yolo_ran = False
            if self.yolo.should_run():
                self._last_yolo_dets = self.yolo.detect(
                    rgb if rgb is not None else visual,
                    self.cfg.display_width,
                    self.cfg.display_height
                )
                yolo_ran = True
            timing["yolo_ms"] = (time.perf_counter() - t) * 1000

            # 4. Fusion (thermal + anomaly + LD2410 presence)
            t = time.perf_counter()
            radar_present = radar_data is not None and radar_data.human_present
            radar_conf    = radar_data.confidence if radar_data else 0.0
            fused = self.fusion.fuse(
                anomaly_dets, self._last_yolo_dets,
                radar_present, radar_conf
            )
            timing["fusion_ms"] = (time.perf_counter() - t) * 1000

            # 5. Tracking
            t = time.perf_counter()
            tracked = list(self.tracker.update(fused))
            timing["tracker_ms"] = (time.perf_counter() - t) * 1000

            # 6. Alert FC if humans detected
            people = sum(not d.heat_only for d in tracked)
            if people:
                self._alert(people, radar_data)

            elapsed = time.perf_counter() - t0
            timing["total_ms"] = elapsed * 1000
            self._fid += 1

            self._total_unique = self.tracker.total_unique_humans
            yield FrameData(
                timestamp=time.time(),
                frame_id=self._fid,
                thermal_raw=raw,
                thermal_visual=visual,
                rgb_frame=rgb,
                radar=radar_data,
                tracked_humans=tracked,
                anomaly_triggered=triggered,
                yolo_ran=yolo_ran,
                timing_ms=timing,
                fps=self._fps(elapsed),
                sensor_status={
                    "thermal": self.thermal.is_online,
                    "rgb":     self.rgb.is_online,
                    "radar":   self.radar.is_online,
                    "fc":      self.fc.is_online,
                },
                yolo_backend=self.yolo.backend,
                fc_telemetry=self.fc.get_telemetry(),
                total_unique_humans=self._total_unique,
            )

    def _alert(self, n: int, radar: Optional[PresenceData]):
        now = time.time()
        if now - self._last_alert < self.cfg.alert_cooldown_sec:
            return
        self._last_alert = now
        stat = " | STATIONARY (buried?)" if radar and radar.stationary_human else ""
        logger.info(f"[PIPELINE] ALERT: {n} HUMAN(S) DETECTED{stat}")
        # FC is read-only — no data is transmitted back to the flight controller

    def _fps(self, elapsed: float) -> float:
        # Rate of delivered results: wall-clock time between them, including
        # the polls spent waiting for the thermal sensor (which `elapsed`,
        # this cycle's processing time, leaves out).
        now = time.perf_counter()
        if self._last_yield is not None:
            elapsed = now - self._last_yield
        self._last_yield = now
        self._fps_buf.append(elapsed)
        if len(self._fps_buf) > 20: self._fps_buf.pop(0)
        avg = sum(self._fps_buf) / len(self._fps_buf)
        return 1.0 / avg if avg > 0 else 0.0
