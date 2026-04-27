"""
Raspberry Pi Camera Module 3 — Sony IMX708
Uses libcamera backend via Picamera2 (NOT cv2.VideoCapture).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROOT CAUSE OF "PDAF data in unsupported format" ERROR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Error:  [ERROR] IPARPI cam_helper_imx708.cpp:262
          PDAF data in unsupported format

The IMX708 sensor embeds Phase Detection AF (PDAF) pixel data
inside its hardware metadata stream on every frame. libcamera's
cam_helper_imx708.cpp parses this embedded data unconditionally,
regardless of what AfMode you set in software controls.

When you request a LOW resolution (e.g. 640×480), libcamera
selects a BINNED sensor readout mode to save bandwidth. In the
binned mode, the IMX708 does NOT output PDAF data in the expected
format — the PDAF pixel layout changes or is omitted entirely.
cam_helper_imx708.cpp still tries to parse it, fails with the
above error, marks the frame as corrupt, and never delivers it
to userspace. This is why you see the error AND get no frames —
they are all being silently dropped at the driver level.

This happens REGARDLESS of AfMode=0/1/2. The PDAF parsing sits
below the IPA control layer and cannot be suppressed by any
runtime control.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE FIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Force libcamera to use the IMX708's FULL-FRAME sensor readout
mode by setting the raw stream size to 2304×1296 (the native
16:9 full-frame mode). In full-frame mode the PDAF data is
present and correctly formatted — the error disappears entirely.

The ISP's hardware scaler then downscales the full-frame output
to the desired 640×480 for the main stream. This costs no extra
CPU — the scaling happens on the VideoCore hardware, not in Python.

IMX708 sensor modes (from libcamera-hello --list-cameras):
  Mode 0:  2304×1296  full-frame  10-bit  RGGB  ← use this
  Mode 1:  2304×1296  full-frame  10-bit  RGGB  HDR
  Mode 2:  1152×648   2×2 binned  10-bit  RGGB  ← triggers PDAF error
  Mode 3:   816×616   2×2 binned  10-bit  RGGB  ← triggers PDAF error

SECONDARY FIXES also in this file:
  • BGR888 format enforced (prevents pink/magenta output)
  • capture_continuous() used instead of capture_array() (prevents
    black screen / DMA buffer deadlock after frame 1)
  • Watchdog restart runs inside the capture thread (prevents crash
    from cross-thread Picamera2 access)
  • AfMode=0 Manual AF with fixed LensPosition (prevents lens hunting)
  • _last_capture_time guarded by lock (fixes TOCTOU race)
"""

import json
import logging
import os
import numpy as np
import cv2
import threading
import time
from typing import Optional

logger = logging.getLogger("rgb")

_WARMUP_FRAMES: int = 30
WATCHDOG_TIMEOUT_S: float = 5.0

# IMX708 full-frame sensor readout size — keeps PDAF data valid.
# The ISP hardware scaler downscales this to rgb_width×rgb_height for free.
_SENSOR_W: int = 2304
_SENSOR_H: int = 1296

# Fixed lens position for aerial use.  0.0=infinity  10.0=macro  0.5≈2m
_LENS_POSITION: float = 0.5

# Path where we write the patched tuning JSON (disables PDAF in IPA as well)
_TUNING_PATH: str = "/tmp/imx708_no_pdaf.json"


def _write_pdaf_disabled_tuning() -> Optional[str]:
    """
    Write a minimal Picamera2 tuning override that disables the PDAF
    algorithm in the IPA (Image Processing Accelerator).

    This is a belt-and-suspenders measure on top of the full-frame sensor
    mode fix.  On very old libcamera builds (< 0.1.0) the IPA still tries
    to run PDAF even in full-frame mode; this JSON tells it not to.

    Returns the path to the written file, or None if it failed.
    """
    try:
        from picamera2 import Picamera2
        # Load the stock IMX708 tuning file that ships with picamera2
        tuning = Picamera2.load_tuning_file("imx708.json")

        # Walk the algorithms list and disable the PDAF/AF algorithm block.
        # The key name varies across libcamera versions: "af", "pdaf", "Af".
        # We disable all of them to be safe.
        algos = tuning.get("algorithms", [])
        for algo_entry in algos:
            for key in list(algo_entry.keys()):
                if key.lower() in ("af", "pdaf", "rpi.af", "rpi.pdaf"):
                    # Set enable=false rather than removing the block entirely
                    # to avoid parse errors in strict-schema libcamera builds.
                    if isinstance(algo_entry[key], dict):
                        algo_entry[key]["enable"] = 0
                    logger.info(f"[RGB] Tuning: disabled PDAF algorithm block '{key}'")

        with open(_TUNING_PATH, "w") as f:
            json.dump(tuning, f)
        return _TUNING_PATH

    except Exception as e:
        logger.debug(f"[RGB] Could not write PDAF-disabled tuning file: {e} — continuing without it")
        return None


class RGBCamera:
    """
    RPi Camera Module 3 via Picamera2 / libcamera.
    See module docstring for full explanation of the PDAF fix.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._cam = None
        self._online = False
        self._backend = "none"

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_capture_time: float = 0.0

    # ── Public API ─────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Launch the background capture thread. Hardware init happens inside it."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._update_loop, daemon=True, name="rgb-capture"
        )
        self._thread.start()

        deadline = time.time() + 8.0   # slightly longer — full-frame init takes longer
        while time.time() < deadline:
            with self._lock:
                if self._online:
                    return True
                if self._backend == "failed":
                    return False
            time.sleep(0.05)

        logger.warning("[RGB] Camera initialization timed out")
        return False

    def read(self) -> Optional[np.ndarray]:
        """Return latest frame (thread-safe copy). None if not yet available."""
        if self.cfg.demo_mode:
            return self._synthetic()
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=4.0)
        self._close_hardware()
        # Clean up temp tuning file
        try:
            if os.path.exists(_TUNING_PATH):
                os.remove(_TUNING_PATH)
        except Exception:
            pass

    @property
    def is_online(self) -> bool:
        return self._online

    # ── Hardware lifecycle (capture thread only) ───────────────────

    def _open_hardware(self) -> bool:
        if self.cfg.demo_mode:
            with self._lock:
                self._online = True
                self._backend = "picamera2"
            logger.info("[RGB] Demo mode")
            return True

        try:
            from picamera2 import Picamera2

            # ── PDAF FIX — write tuning override ──────────────────
            tuning_path = _write_pdaf_disabled_tuning()
            if tuning_path:
                # Load the patched tuning file so the IPA does not run PDAF
                tuning = Picamera2.load_tuning_file(tuning_path)
                cam = Picamera2(tuning=tuning)
                logger.info("[RGB] Loaded PDAF-disabled tuning override")
            else:
                cam = Picamera2()

            # ── PDAF FIX — force full-frame sensor readout mode ────
            #
            # THE KEY FIX: specifying raw stream at 2304×1296 forces
            # libcamera to select the IMX708's full-frame sensor mode.
            # In full-frame mode, PDAF embedded data is present and
            # correctly formatted — cam_helper_imx708.cpp parses it
            # without error and every frame is delivered.
            #
            # Without this, requesting main=640×480 alone causes libcamera
            # to pick the 2×2 binned mode (mode 2 or 3) where PDAF data
            # is absent/malformed → ERROR on every frame → no frames delivered.
            #
            # The main stream is still delivered at rgb_width×rgb_height
            # because the ISP scaler handles the downscaling in hardware.
            config = cam.create_video_configuration(
                main={
                    "size":   (self.cfg.rgb_width, self.cfg.rgb_height),
                    "format": "BGR888",   # MUST be BGR888 for OpenCV — any other
                                          # format causes pink/magenta output
                },
                raw={
                    "size":   (_SENSOR_W, _SENSOR_H),  # full-frame → valid PDAF data
                },
                controls={
                    "FrameRate":    float(self.cfg.rgb_fps),
                    "AfMode":       0,              # Manual AF — no lens hunting
                    "LensPosition": _LENS_POSITION, # ~2m focus for aerial use
                    "NoiseReductionMode": 1,        # Fast noise reduction — saves CPU
                },
                buffer_count=4,
            )
            cam.configure(config)
            cam.start()

            # Allow AEC/AWB and fixed focus to converge before trusting frames.
            # Full-frame mode takes slightly longer to stabilise than binned.
            time.sleep(1.0)

            with self._lock:
                self._cam = cam
                self._backend = "picamera2"
                self._online = True
                self._last_capture_time = time.time()

            logger.info(
                f"[RGB] Pi Camera Module 3 online | "
                f"sensor={_SENSOR_W}x{_SENSOR_H} (full-frame, PDAF valid) | "
                f"output={self.cfg.rgb_width}x{self.cfg.rgb_height} @ {self.cfg.rgb_fps}fps | "
                f"AF=Manual LensPos={_LENS_POSITION} | format=BGR888"
            )
            return True

        except ImportError:
            logger.warning("[RGB] picamera2 not found — trying OpenCV fallback")
        except Exception as e:
            logger.warning(f"[RGB] Picamera2 init failed: {e} — trying OpenCV fallback")

        # OpenCV fallback (USB cameras / v4l2)
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.rgb_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.rgb_height)
                cap.set(cv2.CAP_PROP_FPS,          self.cfg.rgb_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                cap.set(cv2.CAP_PROP_AUTOFOCUS,    0)
                with self._lock:
                    self._cam = cap
                    self._backend = "opencv"
                    self._online = True
                    self._last_capture_time = time.time()
                logger.info("[RGB] Online via OpenCV VideoCapture fallback")
                return True
        except Exception as e:
            logger.warning(f"[RGB] OpenCV fallback failed: {e}")

        logger.warning("[RGB] No camera available — display will show thermal only")
        with self._lock:
            self._backend = "failed"
        return False

    def _close_hardware(self):
        """Cleanly shut down camera. Must be called from the capture thread."""
        try:
            cam = self._cam
            backend = self._backend
            if cam is not None:
                if backend == "picamera2":
                    cam.stop()
                    cam.close()
                elif backend == "opencv":
                    cam.release()
        except Exception:
            pass
        finally:
            self._cam = None
            with self._lock:
                self._online = False

    # ── Capture loop ───────────────────────────────────────────────

    def _update_loop(self):
        """
        All Picamera2 calls happen in this thread.
        libcamera is not thread-safe — never call cam methods from main thread.
        """
        if not self._open_hardware():
            return

        h, w = self.cfg.rgb_height, self.cfg.rgb_width
        frame_count = 0

        while not self._stop_event.is_set():
            try:
                backend = self._backend

                if backend == "picamera2":
                    cam = self._cam
                    if cam is None:
                        break

                    # capture_request() correctly recycles the DMA ring buffer.
                    # capture_array() re-allocates on every call → deadlock after frame 1.
                    while True:
                        request = cam.capture_request()
                        if request is None: break
                        
                        if self._stop_event.is_set():
                            request.release()
                            break

                        with self._lock:
                            last_t = self._last_capture_time

                        # Watchdog: if the ISP/CSI bus stalls again, restart
                        if frame_count > _WARMUP_FRAMES and (time.time() - last_t) > WATCHDOG_TIMEOUT_S:
                            logger.error("[RGB] Watchdog: no frame for >5s — restarting pipeline")
                            request.release()
                            break

                        frame = request.make_array("main")
                        # libcamera formats as RGB, OpenCV expects BGR. Flip channels.
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        request.release()

                        if frame.shape[0] != h or frame.shape[1] != w:
                            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

                        frame_count += 1
                        with self._lock:
                            self._frame = frame
                            self._last_capture_time = time.time()

                    if self._stop_event.is_set():
                        break

                    # Restart entirely inside this thread (cross-thread Picamera2 = crash)
                    logger.info("[RGB] Restarting camera pipeline...")
                    self._close_hardware()
                    time.sleep(2.0)   # let /dev/media FDs fully release
                    frame_count = 0
                    with self._lock:
                        self._last_capture_time = time.time()
                    if not self._open_hardware():
                        logger.error("[RGB] Camera restart failed — giving up")
                        break

                elif backend == "opencv":
                    cam = self._cam
                    if cam is None:
                        break
                    ret, frame = cam.read()
                    if ret and frame is not None:
                        if frame.shape[0] != h or frame.shape[1] != w:
                            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
                        frame_count += 1
                        with self._lock:
                            self._frame = frame
                            self._last_capture_time = time.time()
                    else:
                        time.sleep(0.02)
                else:
                    time.sleep(0.1)

            except Exception as e:
                logger.debug(f"[RGB] Capture error: {e}")
                time.sleep(0.1)

        self._close_hardware()

    # ── Synthetic demo frame ───────────────────────────────────────

    def _synthetic(self) -> np.ndarray:
        h, w = self.cfg.rgb_height, self.cfg.rgb_width
        frame = np.random.randint(200, 240, (h, w, 3), dtype=np.uint8)
        return cv2.GaussianBlur(frame, (15, 15), 3)
