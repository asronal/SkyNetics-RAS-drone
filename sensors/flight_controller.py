"""
BotWing F722 Flight Controller — Read-Only MAVLink Telemetry Listener
Receives battery, GPS, attitude, and arming status from the FC for the OSD.
Does NOT send any data, commands, or overrides back to the FC.

Protocol: MAVLink (via pymavlink) — LISTEN ONLY

F722 Setup (INAV Configurator):
  Ports tab -> UART where RPi is connected:
    Telemetry Output -> MAVLink
    Baud -> 115200

Wiring (RPi4 to FC — UART2; UART0 is the radar's, see README / config.py):
  FC TX (Telemetry) → RPi GPIO1 / Pin 28 (UART2 RX)
  FC RX (Optional)  → RPi GPIO0 / Pin 27 (UART2 TX)
  GND               → GND (Pin 6, 9, 14, etc.)
  UART2 needs `dtoverlay=uart2`; port: cfg.fc_port.
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

# The FC sends a HEARTBEAT every second. A silent link (cable unplugged, FC
# rebooted) raises no error on a serial port, so it is detected by timeout.
_HEARTBEAT_TIMEOUT_S = 5.0


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
        self._last_heartbeat = 0.0
        self._home_lat = None
        self._home_lon = None

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
            "rssi":      0,
            "throttle":  0,
            "batt_rem":  0,
            "current_a": 0.0,
            "batt_mah":  0,
            "dist_home": 0.0,
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
        master = None
        try:
            from pymavlink import mavutil

            master = mavutil.mavlink_connection(self.cfg.fc_port, baud=self.cfg.fc_baud)

            # Wait for a heartbeat (confirms FC is alive). wait_heartbeat()
            # returns None on timeout rather than raising.
            if master.wait_heartbeat(timeout=3.0) is None:
                raise TimeoutError("no heartbeat within 3s")

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
                self._last_heartbeat = time.monotonic()

            logger.info(
                f"[FC] MAVLink connected (read-only) | "
                f"{self.cfg.fc_port} @ {self.cfg.fc_baud} baud"
            )
            return True

        except ImportError:
            logger.error("[FC] pymavlink not installed — run: pip install pymavlink")
        except Exception as e:
            logger.warning(f"[FC] Could not connect on {self.cfg.fc_port}: {e}")
        if master is not None:
            try:
                master.close()
            except Exception:
                pass
        return False

    def _drop_link(self):
        """Mark the FC offline and close the port so a reconnect starts clean."""
        with self._lock:
            master, self._master = self._master, None
            self._online = False
        if master is not None:
            try:
                master.close()
            except Exception:
                pass

    # ── Internal: Receive Loop ────────────────────────────────────

    def _watchdog_loop(self):
        """Background thread: connect, read messages, reconnect on drop."""
        first = True
        while not self._stop_event.is_set():
            if not self._online:
                if not first:
                    # FC dropped — wait then retry
                    logger.warning(
                        f"[FC] Disconnected — retrying in "
                        f"{self.cfg.fc_heartbeat_interval_sec:.0f}s…"
                    )
                    if self._stop_event.wait(self.cfg.fc_heartbeat_interval_sec):
                        break
                first = False
                if not self._connect():
                    continue

            try:
                # Drain all pending messages without blocking
                while True:
                    msg = self._master.recv_match(blocking=False)
                    if msg is None:
                        break
                    self._handle(msg)
            except Exception as e:
                logger.warning(f"[FC] Read error: {e}")
                self._drop_link()
                continue

            if time.monotonic() - self._last_heartbeat > _HEARTBEAT_TIMEOUT_S:
                logger.warning(f"[FC] No heartbeat for {_HEARTBEAT_TIMEOUT_S:.0f}s — link lost")
                self._drop_link()
                continue

            self._stop_event.wait(0.05)  # ~20 Hz poll rate

    def _handle(self, msg):
        """Parse a single incoming MAVLink message into the telemetry dict."""
        msg_type = msg.get_type()
        with self._lock:
            if msg_type == "SYS_STATUS":
                if msg.voltage_battery != 0xFFFF:     # UINT16_MAX = voltage not sent
                    self._telemetry["battery_v"] = msg.voltage_battery / 1000.0
                if msg.current_battery >= 0:
                    self._telemetry["current_a"] = msg.current_battery / 100.0
                if msg.battery_remaining >= 0:
                    self._telemetry["batt_rem"] = msg.battery_remaining

            elif msg_type == "VFR_HUD":
                self._telemetry["speed_kmh"] = msg.groundspeed * 3.6
                self._telemetry["alt_m"]     = msg.alt
                self._telemetry["throttle"]  = getattr(msg, "throttle", 0)

            elif msg_type == "GLOBAL_POSITION_INT":
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                self._telemetry["lat"] = lat
                self._telemetry["lon"] = lon

                # Distance to home
                if self._home_lat is not None and self._home_lon is not None:
                    import math
                    R = 6371000
                    phi1 = math.radians(self._home_lat)
                    phi2 = math.radians(lat)
                    delta_phi = math.radians(lat - self._home_lat)
                    delta_lambda = math.radians(lon - self._home_lon)
                    a = math.sin(delta_phi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2.0)**2
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                    self._telemetry["dist_home"] = R * c

            elif msg_type == "HOME_POSITION":
                self._home_lat = msg.latitude / 1e7
                self._home_lon = msg.longitude / 1e7

            elif msg_type == "GPS_RAW_INT":
                self._telemetry["sats"] = msg.satellites_visible

            elif msg_type == "HEARTBEAT":
                from pymavlink import mavutil
                self._last_heartbeat = time.monotonic()
                armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self._telemetry["armed"] = armed
                self._telemetry["mode"]  = mavutil.mode_string_v10(msg)

            elif msg_type == "ATTITUDE":
                import math
                self._telemetry["pitch"]   = math.degrees(msg.pitch)
                self._telemetry["roll"]    = math.degrees(msg.roll)
                hdg = math.degrees(msg.yaw)
                self._telemetry["heading"] = hdg if hdg >= 0 else 360 + hdg

            elif msg_type == "RC_CHANNELS_RAW":
                if msg.rssi != 255 and msg.rssi > 0:
                    # Scale 0-254 to roughly 0-100%
                    self._telemetry["rssi"] = int((msg.rssi / 254.0) * 100)

            elif msg_type == "BATTERY_STATUS":
                if msg.current_consumed >= 0:
                    self._telemetry["batt_mah"] = msg.current_consumed
