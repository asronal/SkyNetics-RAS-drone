"""
HLK-LD2450 24GHz FMCW Human Trajectory Tracking Radar
Replaces the 1D LD2410 with native 2D positioning (X, Y) for up to 3 targets!

Why this module is vastly superior for the Rescue Drone:
  1. It outputs actual 2D coordinates (X, Y) allowing a TRUE top-down radar scope!
  2. Tracks up to 3 simultaneous humans in real-time.
  3. Outputs velocity (Speed cm/s) and Distance directly.
  4. Max tracking distance is 8 Metres with a 60° azimuth.

Wiring (RPi4 - Dual UART Setup):
  Since UART0 (Pins 8/10) is used by the Flight Controller, we enable UART3 for the Radar.
  1. Add `dtoverlay=uart3` to `/boot/firmware/config.txt` (or `/boot/config.txt`) and reboot.
  2. Update your Python config to use `/dev/ttyAMA1` (the typical OS assigned port for UART3).
  
  LD2450 TX → RPi GPIO5 / Pin 29 (UART3 RX)
  LD2450 RX → RPi GPIO4 / Pin 7  (UART3 TX)
  5V        → 5V (Pin 2 or 4)
  GND       → GND (Pin 6, 14, or 30)

Data Protocol:
  Baud rate : 256000 (Set in your config file)
  Header    : AA FF 03 00
  Payload   : 24 bytes (3 targets * 8 bytes each)
              Each target: X(int16), Y(int16), Speed(int16), Distance_Resolution(int16)
  Footer    : 55 CC
"""

import logging
import struct
import serial
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger("ld2450")

FRAME_HEADER = bytes([0xAA, 0xFF, 0x03, 0x00])
FRAME_END    = bytes([0x55, 0xCC])


@dataclass
class RadarTarget:
    x_mm: int       # Lateral distance left/right (mm) -> neg is left, pos is right
    y_mm: int       # Longitudinal distance (mm) -> straight ahead
    speed_cm_s: int # Radial velocity (cm/s) -> pos is approaching, neg is leaving
    resolution: int
    
    @property
    def distance_cm(self) -> float:
        return self.y_mm / 10.0


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
            # Note: LD2450 runs at 256000 baud
            baud = 256000 
            port = serial.Serial(
                port=self.cfg.ld2410_port,
                baudrate=baud,
                timeout=1.0,
            )

            self._running = True
            self._thread = threading.Thread(
                target=self._read_loop, args=(port,), daemon=True
            )
            self._thread.start()
            self._online = True
            logger.info(
                f"[LD2450] 2D Tracking Radar online on {self.cfg.ld2410_port} @ {baud}"
            )
            return True

        except serial.SerialException as e:
            logger.error(f"[LD2450] Serial error: {e}")
            return False

    def get_presence(self) -> Optional[PresenceData]:
        with self._lock:
            return self._data

    def shutdown(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._online = False

    def _read_loop(self, port: serial.Serial):
        buf = bytearray()
        while self._running:
            try:
                chunk = port.read(128) # Larger chunk for 256kbaud
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
        # Look for AA FF 03 00
        idx = buf.find(FRAME_HEADER)
        if idx < 0:
            return buf[-4:], None

        buf = buf[idx:]

        # Frame structure: [Header: 4 bytes] [Payload: 24 bytes] [Footer: 2 bytes] = 30 bytes
        frame_len = 30
        if len(buf) < frame_len:
            return buf, None

        if buf[frame_len-2:frame_len] != FRAME_END:
            # Bad footer, skip header and search again
            return buf[1:], None

        payload = buf[4:28]
        data = self._parse_payload(payload)
        return buf[frame_len:], data

    def _parse_payload(self, payload: bytes) -> Optional[PresenceData]:
        targets = []
        for i in range(3):
            # Each target is 8 bytes: X, Y, Speed, Res (all Int16 / short little-endian)
            offset = i * 8
            # The LD2450 formats signed values specially in some builds, but standard is little endian short
            x, y, v, res = struct.unpack_from("<hhhh", payload, offset)
            
            # Unmapped targets are usually all zeroes, or Y is 0. 
            if y > 0: 
                # Note: The LD2450 often masks the sign bit. For standard FW:
                # X has bit 15 as sign.
                t = RadarTarget(
                    x_mm=x,
                    y_mm=y,
                    speed_cm_s=v,
                    resolution=res
                )
                targets.append(t)
                
        return PresenceData(targets=targets, timestamp=time.time())

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
