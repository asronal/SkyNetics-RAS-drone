"""
HLK-LD2450 24GHz FMCW Human Trajectory Tracking Radar
Replaces the 1D LD2410 with native 2D positioning (X, Y) for up to 3 targets!

Why this module is vastly superior for the Rescue Drone:
  1. It outputs actual 2D coordinates (X, Y) allowing a TRUE top-down radar scope!
  2. Tracks up to 3 simultaneous humans in real-time.
  3. Outputs velocity (Speed cm/s) and Distance directly.
  4. Max tracking distance is 8 Metres with a 60° azimuth.

Wiring (RPi4 — UART0; the flight controller uses UART2, see README / config.py):
  LD2450 TX → RPi GPIO15 / Pin 10 (UART0 RX)
  LD2450 RX → RPi GPIO14 / Pin 8  (UART0 TX)
  5V        → 5V (Pin 2 or 4)
  GND       → GND (Pin 6, 14, or 30)

  UART0 needs `dtoverlay=disable-bt` in /boot/firmware/config.txt, and
  `console=serial0,115200` removed from /boot/firmware/cmdline.txt.
  Port: cfg.ld2410_port (/dev/ttyAMA0).

Data Protocol:
  Baud rate : 256000 (Set in your config file)
  Header    : AA FF 03 00
  Payload   : 24 bytes (3 targets * 8 bytes each)
              Each target: X(int16), Y(int16), Speed(int16), Distance_Resolution(int16)
  Footer    : 55 CC
"""

import logging
import math
import struct
import serial
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger("ld2450")

FRAME_HEADER = bytes([0xAA, 0xFF, 0x03, 0x00])
FRAME_END    = bytes([0x55, 0xCC])
FRAME_LEN    = 30     # header 4 + 3 targets x 8 + footer 2

# The radar reports ~10x/s. Older data means the radar stopped talking
# (unplugged, UART error) — report no presence instead of the last targets forever.
_STALE_S = 1.0


@dataclass
class RadarTarget:
    x_mm: int       # Lateral distance left/right (mm) -> neg is left, pos is right
    y_mm: int       # Longitudinal distance (mm) -> straight ahead
    speed_cm_s: int # Radial velocity (cm/s) -> pos is approaching, neg is leaving
    resolution: int
    
    @property
    def distance_cm(self) -> float:
        # Range to the target, not just its forward (Y) component.
        return math.hypot(self.x_mm, self.y_mm) / 10.0


@dataclass
class PresenceData:
    """Struct maintaining backwards compatibility with our LD2410 pipeline, while adding 2D lists."""
    targets: List[RadarTarget] = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def human_present(self) -> bool:
        return len(self.targets) > 0

    @property
    def stationary_human(self) -> bool:
        # Consider speed < 10 cm/s to be "stationary/buried"
        return any(abs(t.speed_cm_s) < 10 for t in self.targets)

    @property
    def detect_dist_cm(self) -> int:
        if not self.targets: return 0
        return int(min([t.distance_cm for t in self.targets]))

    @property
    def moving_energy(self) -> int:
        return 80 if self.human_present else 0
        
    @property
    def static_energy(self) -> int:
        return 80 if self.stationary_human else 0

    @property
    def target_state(self) -> int:
        if not self.targets: return 0
        has_moving = any(abs(t.speed_cm_s) >= 10 for t in self.targets)
        has_static = any(abs(t.speed_cm_s) < 10 for t in self.targets)
        if has_moving and has_static: return 3
        if has_static: return 2
        return 1

    @property
    def confidence(self) -> float:
        return 1.0 if self.human_present else 0.0


class LD2450Sensor:
    def __init__(self, cfg):
        self.cfg = cfg
        self._data: Optional[PresenceData] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._online = False

    def initialize(self) -> bool:
        if self.cfg.demo_mode:
            self._online = True
            self._running = True
            self._thread = threading.Thread(target=self._demo_loop, daemon=True)
            self._thread.start()
            logger.info("[LD2450] Demo mode — synthetic 2D presence data")
            return True

        try:
            # LD2450 runs at 256000 baud by default — honour cfg.ld2410_baud
            # instead of hardcoding, so the config value actually takes effect.
            port = serial.Serial(
                port=self.cfg.ld2410_port,
                baudrate=self.cfg.ld2410_baud,
                timeout=1.0,
            )

            self._running = True
            self._thread = threading.Thread(
                target=self._read_loop, args=(port,), daemon=True
            )
            self._thread.start()
            self._online = True
            logger.info(
                f"[LD2450] 2D Tracking Radar online on {self.cfg.ld2410_port} @ {self.cfg.ld2410_baud}"
            )
            return True

        except serial.SerialException as e:
            logger.error(f"[LD2450] Serial error: {e}")
            return False

    def get_presence(self) -> Optional[PresenceData]:
        with self._lock:
            data = self._data
        if data is None or time.time() - data.timestamp > _STALE_S:
            return None
        return data

    def shutdown(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._online = False

    def _read_loop(self, port: serial.Serial):
        buf = bytearray()
        while self._running:
            try:
                # Whatever has arrived (at least 1 byte, or the 1s timeout):
                # a fixed read(128) waited ~0.4s to fill at the radar's ~10 frames/s.
                chunk = port.read(port.in_waiting or 1)
                if chunk:
                    buf.extend(chunk)
                    buf, data = self._parse_buffer(buf)
                    if data:
                        with self._lock:
                            self._data = data
            except serial.SerialException as e:
                logger.error(f"[LD2450] Read error: {e}")
                self._online = False
                break
        try:
            port.close()
        except Exception:
            pass

    def _parse_buffer(self, buf: bytearray):
        """Consume every complete frame in `buf`. Returns (unparsed tail, newest
        PresenceData or None). Parsing only one frame per read let the backlog
        grow without bound, so the data reported fell further and further behind."""
        latest = None
        while True:
            # Look for AA FF 03 00
            idx = buf.find(FRAME_HEADER)
            if idx < 0:
                # Keep a possible partial header at the end
                return buf[-(len(FRAME_HEADER) - 1):], latest
            buf = buf[idx:]

            if len(buf) < FRAME_LEN:
                return buf, latest

            if buf[FRAME_LEN-2:FRAME_LEN] != FRAME_END:
                # Bad footer, skip header and search again
                buf = buf[1:]
                continue

            latest = self._parse_payload(buf[4:FRAME_LEN-2])
            buf = buf[FRAME_LEN:]

    def _parse_payload(self, payload: bytes) -> Optional[PresenceData]:
        targets = []
        for i in range(3):
            # Each target is 8 bytes: X, Y, Speed, Res (all uint16 little-endian)
            offset = i * 8
            x_raw, y_raw, v_raw, res = struct.unpack_from("<HHHH", payload, offset)

            # The LD2450 does NOT use two's complement for X/Y/Speed — bit 15
            # is a sign flag (1 = positive, 0 = negative) and bits 0-14 hold the
            # absolute magnitude. Unpacking as plain signed shorts flipped the
            # sign of every "positive" (bit15 set) reading, which is the normal
            # case for a target in front of the radar — so real targets were
            # decoding as negative Y and failing the `y > 0` check below.
            x = self._decode_signed(x_raw)
            y = self._decode_signed(y_raw)
            v = self._decode_signed(v_raw)

            # Unmapped target slots are all-zero bytes -> y decodes to 0.
            if y > 0:
                t = RadarTarget(
                    x_mm=x,
                    y_mm=y,
                    speed_cm_s=v,
                    resolution=res
                )
                targets.append(t)

        return PresenceData(targets=targets, timestamp=time.time())

    @staticmethod
    def _decode_signed(raw: int) -> int:
        """LD2450 sign-magnitude int16: bit15 = sign (1=positive), bits0-14 = magnitude."""
        magnitude = raw & 0x7FFF
        return magnitude if (raw & 0x8000) else -magnitude

    def _demo_loop(self):
        import random
        while self._running:
            # Simulate 1 target walking
            t = RadarTarget(
                x_mm=random.randint(-2000, 2000),
                y_mm=random.randint(1500, 5000),
                speed_cm_s=random.randint(5, 45),
                resolution=100
            )
            data = PresenceData(targets=[t], timestamp=time.time())
            with self._lock:
                self._data = data
            time.sleep(0.1)

    @property
    def is_online(self) -> bool:
        return self._online
