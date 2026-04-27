"""
BotWing F722 Flight Controller — Read-Only MAVLink Telemetry Listener
Receives battery, GPS, attitude, and arming status from the FC for the OSD.
Does NOT send any data, commands, or overrides back to the FC.

Protocol: MAVLink (via pymavlink) — LISTEN ONLY

F722 Setup (INAV Configurator):
  Ports tab -> UART where RPi is connected:
    Telemetry Output -> MAVLink
    Baud -> 115200

Wiring (RPi4 to FC):
  FC TX (Telemetry) → RPi GPIO15 / Pin 10 (UART0 RX)
  FC RX (Optional)  → RPi GPIO14 / Pin 8  (UART0 TX)
  GND               → GND (Pin 6, 9, 14, etc.)
  *Note: Only connect 5V if powering the Pi from the FC (not recommended for Pi4).*

Raspberry Pi 4 Setup for UART:
  1. Run `sudo raspi-config` in the terminal.
  2. Navigate to [3] Interface Options -> [I6] Serial Port.
  3. "Would you like a login shell to be accessible over serial?" -> Select `No`
  4. "Would you like the serial port hardware to be enabled?" -> Select `Yes`
  5. Reboot the Raspberry Pi.

  Manual Configuration via files:
  1. Edit `/boot/firmware/config.txt` (or `/boot/config.txt` on older Pi OS):
     - Ensure `enable_uart=1` is present.
     - (Optional, if using GPIO 14/15) Add `dtoverlay=disable-bt` to use the primary UART on those pins.
  2. Edit `/boot/firmware/cmdline.txt` (or `/boot/cmdline.txt`):
     - Make sure to remove `console=serial0,115200` if it's there so the OS doesn't use the UART.
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("flight_ctrl")


class FlightController:
    """
    Read-only MAVLink telemetry receiver.
    Connects to F722 and passively reads:
      - SYS_STATUS   (battery voltage)
      - VFR_HUD      (speed, altitude)
      - GLOBAL_POSITION_INT (lat/lon)
      - GPS_RAW_INT  (satellite count)
      - HEARTBEAT    (armed status, flight mode)
      - ATTITUDE     (pitch, roll, yaw/heading)

    This module NEVER writes or sends any data to the flight controller.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._master = None
        self._online = False
        self._lock = threading.Lock()

        self._reconnect_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Telemetry store — all fields are updated by the watchdog loop
        self._telemetry = {
            "battery_v": 0.0,
            "speed_kmh": 0.0,
            "lat":       0.0,
            "lon":       0.0,
            "alt_m":     0.0,
            "sats":      0,
            "armed":     False,
            "mode":      "UNKNOWN",
            "pitch":     0.0,
            "roll":      0.0,
            "heading":   0.0,
        }

    # ── Public API ────────────────────────────────────────────────

    def initialize(self) -> bool:
        if not self.cfg.fc_enabled:
            logger.info("[FC] Flight controller disabled in config — OSD telemetry unavailable")
            return True

        if self.cfg.demo_mode:
            self._online = True
            logger.info("[FC] Demo mode — synthetic telemetry")
            return True

        # Always start watchdog even if first connect fails; it will retry
        self._reconnect_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True
        )
        self._reconnect_thread.start()

        # Give the watchdog a moment to attempt the first connection
        time.sleep(1.5)
        return self._online

    def get_telemetry(self) -> dict:
        """Return a snapshot of the latest received telemetry."""
        with self._lock:
            return self._telemetry.copy()

    @property
    def is_online(self) -> bool:
        return self._online

    @property
    def is_armed(self) -> bool:
        """True when the FC reports MAV_MODE_FLAG_SAFETY_ARMED."""
        with self._lock:
            return self._telemetry.get("armed", False)

    def shutdown(self):
        self._stop_event.set()
        if self._reconnect_thread:
            self._reconnect_thread.join(timeout=3.0)
        with self._lock:
            if self._master:
                try:
                    self._master.close()
                except Exception:
                    pass
        self._online = False
        logger.info("[FC] Shutdown complete")

    # ── Internal: Connection ──────────────────────────────────────

    def _connect(self) -> bool:
        try:
            from pymavlink import mavutil

            master = mavutil.mavlink_connection(self.cfg.fc_port, baud=self.cfg.fc_baud)

            # Wait for a heartbeat (confirms FC is alive)
            master.wait_heartbeat(timeout=3.0)

            # Ask FC to stream all telemetry messages at 10 Hz.
            # This is a REQUEST — the FC decides whether to honour it.
            # We never override RC or send commands.
            master.mav.request_data_stream_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,   # 10 Hz
                1,    # start streaming
            )

            with self._lock:
                self._master = master
                self._online = True

            logger.info(
                f"[FC] MAVLink connected (read-only) | "
                f"{self.cfg.fc_port} @ {self.cfg.fc_baud} baud"
            )
            return True

        except ImportError:
            logger.error("[FC] pymavlink not installed — run: pip install pymavlink")
            return False
        except Exception as e:
            logger.warning(f"[FC] Could not connect on {self.cfg.fc_port}: {e}")
            return False

    # ── Internal: Receive Loop ────────────────────────────────────

    def _watchdog_loop(self):
        """Background thread: connect, read messages, reconnect on drop."""
        # Attempt initial connection
        if not self._online:
            self._connect()

        while not self._stop_event.is_set():
            if self._online and self._master is not None:
                try:
                    # Drain all pending messages without blocking
                    while True:
                        msg = self._master.recv_match(blocking=False)
                        if msg is None:
                            break
                        self._handle(msg)
                except Exception as e:
                    logger.warning(f"[FC] Read error: {e}")
                    with self._lock:
                        self._online = False

            elif not self._stop_event.is_set():
                # FC dropped — wait then retry
                logger.warning(
                    f"[FC] Disconnected — retrying in "
                    f"{self.cfg.fc_heartbeat_interval_sec:.0f}s…"
                )
                time.sleep(self.cfg.fc_heartbeat_interval_sec)
                self._connect()

            time.sleep(0.05)  # ~20 Hz poll rate

    def _handle(self, msg):
        """Parse a single incoming MAVLink message into the telemetry dict."""
        msg_type = msg.get_type()
        with self._lock:
            if msg_type == "SYS_STATUS":
                self._telemetry["battery_v"] = msg.voltage_battery / 1000.0

            elif msg_type == "VFR_HUD":
                self._telemetry["speed_kmh"] = msg.groundspeed * 3.6
                self._telemetry["alt_m"]     = msg.alt

            elif msg_type == "GLOBAL_POSITION_INT":
                self._telemetry["lat"] = msg.lat / 1e7
                self._telemetry["lon"] = msg.lon / 1e7

            elif msg_type == "GPS_RAW_INT":
                self._telemetry["sats"] = msg.satellites_visible

            elif msg_type == "HEARTBEAT":
                from pymavlink import mavutil
                armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self._telemetry["armed"] = armed
                self._telemetry["mode"]  = mavutil.mode_string_v10(msg)

            elif msg_type == "ATTITUDE":
                import math
                self._telemetry["pitch"]   = math.degrees(msg.pitch)
                self._telemetry["roll"]    = math.degrees(msg.roll)
                hdg = math.degrees(msg.yaw)
                self._telemetry["heading"] = hdg if hdg >= 0 else 360 + hdg
