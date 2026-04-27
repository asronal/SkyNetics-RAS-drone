"""
Detection Pipeline v3 — Cascade for RPi4 + MLX90640 + LD2410C-P

Frame budget @ 4fps = 250ms:
  Sensor read          ~15ms
  Anomaly detector     ~15ms   (every frame)
  YOLOv8n ONNX        ~200ms  (every 3rd frame → ~67ms amortized)
  Fusion + track       ~10ms
  Total               ~107ms  ✓ fits budget
"""

import logging
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
        return len(self.tracked_humans)

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
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.thermal  = ThermalCamera(cfg)
        self.rgb      = RGBCamera(cfg)
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
        # black during the camera warmup period (_WARMUP_FRAMES) or during
        # a watchdog restart. Without this, every frame where rgb.read()
        # returns None renders a solid black pane.
        self._last_rgb: Optional[np.ndarray] = None

    def initialize(self):
        ok_t = self.thermal.initialize()
        self.rgb.initialize()
        self.radar.initialize()
        self.fc.initialize()
        self.anomaly.load()
        self.yolo.load()

        if not ok_t and not self.cfg.demo_mode:
            raise RuntimeError("MLX90640 thermal camera failed — cannot operate without it")

        logger.info(
            f"[PIPELINE] Ready\n"
            f"  Thermal  : {'online' if self.thermal.is_online else 'OFFLINE'}\n"
            f"  RGB      : {'online' if self.rgb.is_online else 'OFFLINE'}\n"
            f"  LD2410   : {'online' if self.radar.is_online else 'OFFLINE'}\n"
            f"  FC       : {'online' if self.fc.is_online else 'disabled'}\n"
            f"  YOLO     : {self.yolo.backend}"
        )

    def shutdown(self):
        self.thermal.release()
        self.rgb.release()
        self.radar.shutdown()
        self.fc.shutdown()
        logger.info("[PIPELINE] All sensors shut down")

    def run(self) -> Iterator[FrameData]:
        while True:
            t0 = time.perf_counter()
            timing = {}

            # 1. Read sensors
            t = time.perf_counter()
            raw, visual = self.thermal.read()
            rgb_new = self.rgb.read()
            # FIX: cache last good RGB frame — prevents black pane during
            # camera warmup (_WARMUP_FRAMES) and watchdog restart gaps.
            if rgb_new is not None:
                self._last_rgb = rgb_new
            rgb = self._last_rgb
            radar_data = self.radar.get_presence()
            timing["sensors_ms"] = (time.perf_counter() - t) * 1000

            # thermal.read() returns (None, None) both when rate-limited (normal,
            # between 250ms refresh cycles) and on actual I2C errors.
            # Only warn + skip if we've had no frame for 3+ seconds.
            if visual is None and not self.cfg.demo_mode:
                if not hasattr(self, "_last_thermal_ok"):
                    self._last_thermal_ok = time.monotonic()
                if time.monotonic() - self._last_thermal_ok > 3.0:
                    logger.warning("[PIPELINE] No thermal frame for 3s — check I2C / sensor")
                time.sleep(0.04)   # yield CPU; next getFrame() won't be ready anyway
                continue
            else:
                self._last_thermal_ok = time.monotonic()

            # 2. Anomaly detector — every frame
            t = time.perf_counter()
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
            if tracked:
                self._alert(len(tracked), radar_data)

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
        self._fps_buf.append(elapsed)
        if len(self._fps_buf) > 20: self._fps_buf.pop(0)
        avg = sum(self._fps_buf) / len(self._fps_buf)
        return 1.0 / avg if avg > 0 else 0.0
