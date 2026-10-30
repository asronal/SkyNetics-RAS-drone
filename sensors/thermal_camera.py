"""
MLX90640 Thermal Camera — I2C driver (32x24, 55deg FOV, 4Hz)
Blue-cold / Orange-hot colormap for rescue thermal imaging.

Cold background (snow/debris): deep blue
Warm regions (human body ~37C): bright orange -> white

Wiring (RPi4):
  VCC -> 3.3V  (Pin 1)
  GND -> GND   (Pin 6)
  SDA -> GPIO2 (Pin 3)
  SCL -> GPIO3 (Pin 5)

Setup:
  sudo raspi-config -> Interfaces -> I2C -> Enable
  Add to /boot/config.txt: dtparam=i2c_arm=on,i2c_arm_baudrate=400000
  pip install adafruit-circuitpython-mlx90640 adafruit-blinka
  Verify: i2cdetect -y 1  (expect 0x33)

Architecture:
  adafruit-blinka's I2C (busio.I2C) is NOT thread-safe on Linux —
  calling getFrame() from a background thread while the main thread
  also holds the I2C object causes bus hangs and stuck reads.

  Instead we run blocking reads in the MAIN pipeline loop with explicit
  timing so we never call getFrame() faster than the sensor refresh rate.
  At 4Hz, each read takes ~250ms naturally — fitting inside our frame budget.
"""

import logging
import numpy as np
import cv2
import time
from typing import Optional, Tuple

logger = logging.getLogger("thermal")

W, H = 32, 24
NPIX = W * H

# Minimum time between the starts of getFrame() calls, and the back-off after
# a failed read, as a fraction of one full frame (2 subpages = 2/refresh s,
# ~520ms at 4Hz), so back-to-back reads are never held up by this.
_MIN_FRAME_INTERVAL_FR: float = 0.45

# Pause between data-ready polls. The driver spins on the status register
# (277 I2C polls per frame at 100 kHz, ~4x more at 400 kHz); 5 ms adds at most
# 5 ms of delay and frees the CPU (and the GIL) for the other threads.
_POLL_SLEEP_S = 0.005
_READY_TIMEOUT_S = 2.0   # longer than a subpage at the slowest rate we use

# Calibration (EEPROM) read attempts at start-up.
_EEPROM_TRIES = 5


_CHESS = (np.indices((H, W)).sum(0) % 2).astype(bool)   # one MLX90640 subpage (chess mode)
_CHECKER_KILL = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 8
_LEVEL_ALPHA = 0.35    # frame-level smoothing per frame (settles in ~4 frames: ~1 s at 8 Hz)


def _install_polite_wait(sensor):
    """Replace the driver's _GetFrameData with the same logic, but sleeping
    between status polls and giving up after _READY_TIMEOUT_S (the driver's
    loop spins forever on a dead sensor, so the thread can't be stopped)."""
    def get_frame_data(frame_data):
        status, control = [0], [0]
        deadline = time.monotonic() + _READY_TIMEOUT_S
        sensor._I2CReadWords(0x8000, status)
        while not status[0] & 0x0008:
            if time.monotonic() > deadline:
                raise RuntimeError("No new frame data")
            time.sleep(_POLL_SLEEP_S)
            sensor._I2CReadWords(0x8000, status)
        cnt = 0
        while status[0] & 0x0008 and cnt < 5:   # re-read if a new subpage arrived meanwhile
            sensor._I2CWriteWord(0x8000, 0x0030)
            sensor._I2CReadWords(0x0400, frame_data, end=832)
            sensor._I2CReadWords(0x8000, status)
            cnt += 1
        if cnt > 4:
            raise RuntimeError("Too many retries")
        sensor._I2CReadWords(0x800D, control)
        frame_data[832] = control[0]
        frame_data[833] = status[0] & 0x0001
        return frame_data[833]
    sensor._GetFrameData = get_frame_data


def _open_mlx90640(mlx, i2c):
    """adafruit_mlx90640.MLX90640(i2c), but only from a verified EEPROM read.

    The driver reads the 832-word calibration EEPROM once and trusts it. On a
    flaky bus (loose wire, weak pull-ups) the read fails or turns into 0xFFFF
    part-way, and 0xFFFF calibration sends _ExtractAlphaParameters into an
    endless loop — start-up hangs forever. Two reads must match and must not
    be mostly 0xFFFF before the parameters are extracted."""
    sensor = mlx.MLX90640.__new__(mlx.MLX90640)
    sensor.i2c_device = mlx.I2CDevice(i2c, 0x33)
    last = None
    problem = "no read"
    for _ in range(_EEPROM_TRIES):
        try:
            sensor._I2CReadWords(0x2400, mlx.eeData)
        except OSError as e:
            problem = f"I2C error ({e})"
            last = None
            time.sleep(0.05)
            continue
        ee = list(mlx.eeData)
        if ee.count(0xFFFF) > 64:
            problem = f"{ee.count(0xFFFF)}/832 calibration words read as 0xFFFF"
            last = None
        elif ee == last:
            sensor._ExtractParameters()
            return sensor
        else:
            problem = "calibration reads don't match"
            last = ee
    raise RuntimeError(f"MLX90640 calibration unreadable after {_EEPROM_TRIES} tries: {problem} "
                       f"— check the sensor's SDA/SCL/3.3V/GND wiring")


def build_rescue_colormap() -> np.ndarray:
    """
    Custom blue-cold / orange-hot colormap for rescue thermal imaging.

    0-30%   : deep blue -> mid blue      (cold: snow, ice, debris)
    30-55%  : blue -> cyan -> teal       (cool: cold ground, rock)
    55-75%  : teal -> yellow-green       (ambient: air, surface)
    75-90%  : yellow-orange              (warm: body surface through debris)
    90-100% : orange -> bright orange -> white  (hot: direct human contact)

    This gives maximum contrast at the human body temperature range,
    making survivors pop vividly against cold backgrounds.
    """
    cmap = np.zeros((256, 1, 3), dtype=np.uint8)

    def lerp(v, lo, hi): return int(lo + (hi - lo) * v)

    for i in range(256):
        t = i / 255.0
        if t < 0.30:
            s = t / 0.30
            r = lerp(s, 5,  30)
            g = lerp(s, 10, 60)
            b = lerp(s, 80, 180)
        elif t < 0.55:
            s = (t - 0.30) / 0.25
            r = lerp(s, 30,  20)
            g = lerp(s, 60,  180)
            b = lerp(s, 180, 200)
        elif t < 0.75:
            s = (t - 0.55) / 0.20
            r = lerp(s, 20,  200)
            g = lerp(s, 180, 180)
            b = lerp(s, 200, 20)
        elif t < 0.90:
            s = (t - 0.75) / 0.15
            r = lerp(s, 200, 255)
            g = lerp(s, 180, 100)
            b = lerp(s, 20,  0)
        else:
            s = (t - 0.90) / 0.10
            r = 255
            g = lerp(s, 100, 255)
            b = lerp(s, 0,   220)

        cmap[i, 0] = [b, g, r]  # OpenCV uses BGR

    return cmap


# Build once at import time
RESCUE_COLORMAP = build_rescue_colormap()


class ThermalCamera:
    """
    MLX90640 driver with rescue-optimized blue-cold/orange-hot colormap.

    IMPORTANT: getFrame() is called synchronously in the main pipeline loop.
    adafruit-blinka's I2C is NOT thread-safe on Linux — background-thread
    ownership causes bus hangs. The rate limiter (_last_read_time) ensures
    we never call getFrame() faster than the sensor refresh (cfg.thermal_fps).
    Returns quickly (cached None/None) if called before the sensor is ready.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._sensor = None
        self._online = False
        self._buf = [0.0] * NPIX          # reused buffer — avoids GC pressure
        self._last_read_time: float = 0.0  # rate-limit guard
        self._min_interval = _MIN_FRAME_INTERVAL_FR * 2.0 / max(1, cfg.thermal_fps)
        self._t_min_ema: Optional[float] = None
        self._t_max_ema: Optional[float] = None
        self._level: Optional[float] = None   # smoothed frame median (_denoise)

    def initialize(self) -> bool:
        if self.cfg.demo_mode:
            self._online = True
            logger.info("[THERMAL] Demo mode | rescue colormap active")
            return True
        try:
            import board, busio
            import adafruit_mlx90640
            i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)
            self._sensor = _open_mlx90640(adafruit_mlx90640, i2c)
            _install_polite_wait(self._sensor)
            hz = self.cfg.thermal_fps
            self._sensor.refresh_rate = getattr(adafruit_mlx90640.RefreshRate, f"REFRESH_{hz}_HZ")
            self._min_interval = _MIN_FRAME_INTERVAL_FR * 2.0 / hz
            self._online = True
            logger.info(f"[THERMAL] MLX90640 online @ {hz}Hz | blue-cold/orange-hot palette")
            return True
        except ImportError:
            logger.error("[THERMAL] pip install adafruit-circuitpython-mlx90640 adafruit-blinka")
            return False
        except Exception as e:
            logger.error(f"[THERMAL] Failed: {e} | Check i2cdetect -y 1 for 0x33")
            return False

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.cfg.demo_mode:
            return self._synthetic()
        if not self._online or self._sensor is None:
            return None, None

        # Rate-limit: mainly a back-off after an I2C error, so a failing
        # sensor isn't hammered (getFrame() waits for new data by itself).
        now = time.monotonic()
        if now - self._last_read_time < self._min_interval:
            return None, None   # pipeline will use last fused result — not an error

        try:
            # Measured from the START of a read: getFrame() itself waits for the
            # sensor's data-ready flag, so back-to-back reads are fine. Timing
            # from the end added ~230ms of idle per frame and knocked the reads
            # out of step with the sensor's subpages.
            self._last_read_time = now
            self._sensor.getFrame(self._buf)
            raw = np.array(self._buf, dtype=np.float32).reshape(H, W)
            raw = np.ascontiguousarray(np.fliplr(raw))
            if self.cfg.thermal_denoise:
                self._denoise(raw)
            return raw, self._to_visual(raw)
        except Exception as e:
            # We log exactly what failed on debug to trace I2C issues,
            # but quietly back-off so we don't spam the console.
            logger.debug(f"[THERMAL] Sensor read failed: {type(e).__name__}: {e}")
            self._last_read_time = time.monotonic()
            return None, None

    def _denoise(self, raw: np.ndarray):
        """In place. This sensor's two chess-pattern subpages come out at different
        levels (2-8 C apart, flipping from frame to frame — measured 2026-10-29, the
        raw pixels themselves jump, not the maths), which shows as a checkerboard,
        and the whole frame's level jumps 2-4 C per frame. Shift both subpages to a
        common median, then smooth the frame level over time. Spatial detail (a
        person is split evenly across both subpages) is untouched."""
        d = 0.5 * (np.median(raw[_CHESS]) - np.median(raw[~_CHESS]))
        raw[_CHESS] -= d
        raw[~_CHESS] += d
        m = float(np.median(raw))
        self._level = m if self._level is None else self._level + _LEVEL_ALPHA * (m - self._level)
        raw += self._level - m
        # What's left of the checkerboard grows with the signal (a hot face still
        # shows dots). Centre 1/2 + 4 neighbours 1/8 cancels a pure checkerboard
        # exactly while keeping blob shapes: 0.5 - 4 * 0.125 = 0.
        raw[:] = cv2.filter2D(raw, -1, _CHECKER_KILL, borderType=cv2.BORDER_REPLICATE)

    def release(self):
        self._sensor = None
        self._online = False

    def _to_visual(self, raw: np.ndarray) -> np.ndarray:
        """
        Normalize to 8-bit using dynamic AGC (Auto Gain Control), apply rescue colormap.
        INTER_CUBIC upscale from 32x24 gives smooth blobs — important so
        human-shaped hot regions look natural rather than blocky.
        """
        # Dynamic AGC
        curr_min = float(np.percentile(raw, 2))
        curr_max = float(np.percentile(raw, 99))
        
        # Ensure at least 5C span to avoid noisy stretch on flat walls
        if curr_max - curr_min < 5.0:
            curr_max = curr_min + 5.0

        if self._t_min_ema is None:
            self._t_min_ema = curr_min
            self._t_max_ema = curr_max
        else:
            alpha = 0.15 # Smooth transitions
            self._t_min_ema = self._t_min_ema * (1 - alpha) + curr_min * alpha
            self._t_max_ema = self._t_max_ema * (1 - alpha) + curr_max * alpha

        t_min = self._t_min_ema
        t_max = self._t_max_ema
        
        # Bubble AGC limits to config so other UI components can reflect the dynamic range
        self.cfg.thermal_min_temp = t_min
        self.cfg.thermal_max_temp = t_max

        norm = np.clip((raw - t_min) / (t_max - t_min) * 255, 0, 255).astype(np.uint8)
        up = cv2.resize(
            norm,
            (self.cfg.display_width, self.cfg.display_height),
            interpolation=cv2.INTER_CUBIC,
        )
        return cv2.applyColorMap(up, RESCUE_COLORMAP)

    def temp_at(self, raw, cx_n, cy_n) -> float:
        r = int(np.clip(cy_n * H, 0, H - 1))
        c = int(np.clip(cx_n * W, 0, W - 1))
        return float(raw[r, c]) if raw is not None else 0.0

    def _synthetic(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realistic MLX90640 simulation:
        Snow background -5 to 0C + sensor noise.
        1-2 human heat blobs 33-37C.
        """
        raw = np.random.uniform(-5.0, 0.0, (H, W)).astype(np.float32)
        raw += np.random.normal(0, 0.3, raw.shape).astype(np.float32)
        for _ in range(np.random.randint(1, 3)):
            cx = np.random.randint(4, W - 4)
            cy = np.random.randint(4, H - 4)
            bw, bh = np.random.randint(2, 5), np.random.randint(4, 8)
            temp = np.random.uniform(33.5, 37.0)
            raw[max(0, cy-bh//2):min(H, cy+bh//2),
                max(0, cx-bw//2):min(W, cx+bw//2)] = temp
        raw = cv2.GaussianBlur(raw, (3, 3), 0.8)
        return raw, self._to_visual(raw)

    @property
    def is_online(self) -> bool:
        return self._online
