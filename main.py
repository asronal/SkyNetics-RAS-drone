"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              RESCUE DRONE v3 — Human Detection System                        ║
║              BotWing F722 | MLX90640 | RPi Cam 3 | HLK-LD2450                ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HOW TO RUN MANUALLY (SSH or terminal on the Pi)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Full deployment (cameras + radar + FC telemetry):
    python3 main.py --fc-enabled

  Without a flight controller connected (bench testing):
    python3 main.py

  Laptop / desktop demo (no hardware at all):
    python3 main.py --demo

  Save output as a video file:
    python3 main.py --record

  Headless mode (SSH, no display window):
    python3 main.py --headless

  Output profile (default: config.py display_profile = "auto" — composite
  when the Pi's composite output is active):
    python3 main.py --profile composite   # 720x576 analog-VTX layout, 25fps

  Display path (default: config.py display_backend = "auto"):
    python3 main.py --display gpu      # camera video composited by the GPU
                                       # (zero-copy, every camera frame), OSD
                                       # as a separate layer — desktop session
    python3 main.py --display opencv   # OSD drawn into each frame, cv2 window

  Keyboard shortcuts while running:
    Q  — Quit
    S  — Save manual snapshot to output/
    M  — Cycle thermal colour mode
    T  — Toggle thermal PiP overlay
    V  — Cycle main view (RGB → Thermal → Radar)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HOW TO MAKE THIS CODE AUTO-START WHEN THE PI BOOTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A systemd service file (rescue_drone.service) is included in this folder.
  Run these commands ONCE on the Pi to register it:

    # 1. Copy the service file into systemd's folder
    sudo cp rescue_drone.service /etc/systemd/system/

    # 2. Tell systemd to start it automatically on every boot
    sudo systemctl enable rescue_drone

    # 3. Start it right now without rebooting (optional)
    sudo systemctl start rescue_drone

  After that, every time the Pi powers on the detection system starts
  automatically — no SSH, no keyboard, no monitor needed.

  Useful management commands:
    sudo systemctl stop rescue_drone        # Stop it manually
    sudo systemctl restart rescue_drone     # Restart after a code change
    sudo systemctl status rescue_drone      # See if it is running / crashed
    journalctl -u rescue_drone -f           # Watch the live log stream

  To DISABLE auto-start (go back to manual):
    sudo systemctl disable rescue_drone

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 WHAT EACH MODULE DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  sensors/rgb_camera.py       — Reads the Pi Camera 3 (RGB video)
  sensors/thermal_camera.py   — Reads the MLX90640 thermal sensor
  sensors/ld2450_radar.py     — Reads the HLK-LD2450 mmWave radar
  sensors/flight_controller.py— Reads MAVLink telemetry from BotWing F722
                                 (READ-ONLY — never sends data back to FC)

  ml/models.py                — AnomalyDetector (thermal blobs)
                                 YOLOv8 ONNX detector (RGB faces/bodies)
                                 SensorFusion (combines both)
                                 HumanTracker (Kalman tracking with IDs)

  pipeline/detection_pipeline.py — Main brain: runs all sensors + ML every frame
                                    Auto-saves snapshots to output/snapshots/
                                    when a human is confirmed by the tracker

  display/rescue_display.py   — Renders the OSD window with bounding boxes,
                                 telemetry, thermal PiP, and radar scope
  display/osd.py              — Draws the HUD overlay text and telemetry

  config.py                   — All tunable parameters in one place
  main.py                     — Entry point (this file)
  rescue_drone.service        — systemd auto-start service definition

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SNAPSHOT STORAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Automatic snapshots  → output/snapshots/auto_snap_TRK<id>_<frame>.jpg
  Manual snapshots (S) → output/snap_<frame>.jpg

  Auto-snapshots fire once every 5 seconds per unique tracked person.
  Each image is the full 1280x720 OSD frame with bounding box visible.
"""

import os
import sys

# The sensor drivers (MLX90640/Blinka) and onnxruntime live only in ~/venv. Started
# with plain `python` (system Python) the thermal camera showed OFFLINE and YOLO fell
# back to slow cv2.dnn — so re-launch under the venv when it exists.
_VENV = os.path.expanduser("~/venv")
if os.path.exists(os.path.join(_VENV, "bin", "python")) and \
        os.path.realpath(sys.prefix) != os.path.realpath(_VENV):
    _py = os.path.join(_VENV, "bin", "python")
    print(f"[MAIN] Re-launching with {_py} (sensor drivers + onnxruntime live there)", flush=True)
    os.execv(_py, [_py] + sys.argv)

import argparse
import dataclasses
import logging
import threading
import time

from config import Config, BASE_DIR
from pipeline.detection_pipeline import DetectionPipeline
from display.rescue_display import RescueDisplay
from display.screen import ScreenBlankingInhibitor, apply_display_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # Anchored to the repo dir (BASE_DIR), not the process's cwd — see
        # config.py's BASE_DIR comment for why a bare "rescue.log" scatters
        # into $HOME when launched from anywhere but the project folder.
        logging.FileHandler(str(BASE_DIR / "rescue.log")),
    ],
)
logger = logging.getLogger("main")


def parse_args():
    p = argparse.ArgumentParser(description="Rescue Drone v3")
    # Wiring LD2450/2410 Radar: TX→RPi RX(GPIO15/Pin10), RX→RPi TX(GPIO14/Pin8), VCC→5V, GND→GND
    # Value options default to None so config.py stays the single source of
    # truth — they only override it when actually passed on the command line.
    p.add_argument("--mmwave-port", type=str,   default=None,           help="LD2450 UART port")
    # Wiring BotWing F722 FC: FC TX→RPi RX (e.g., UART1/2), FC RX→RPi TX, GND→GND
    p.add_argument("--fc-port",     type=str,   default=None,           help="F722 UART port")
    p.add_argument("--fc-baud",     type=int,   default=None,           help="FC baud rate")
    p.add_argument("--fc-enabled",  action="store_true",                help="Enable FC telemetry")
    p.add_argument("--yolo-model",  type=str,   default=None,           help="YOLO .onnx path")
    p.add_argument("--conf",        type=float, default=None,           help="YOLO confidence threshold")
    p.add_argument("--record",      action="store_true",                help="Save output as video")
    p.add_argument("--headless",    action="store_true",                help="No display window (SSH)")
    p.add_argument("--demo",        action="store_true",                help="Synthetic data — no hardware")
    p.add_argument("--no-fullscreen",action="store_true",               help="Disable fullscreen mode")
    p.add_argument("--display",     choices=["auto", "gpu", "opencv"], default=None,
                   help="Display path (default: config.py display_backend)")
    p.add_argument("--profile",     choices=["auto", "hdmi", "composite"], default=None,
                   help="Output profile (default: config.py display_profile — "
                        "composite when the composite output is active)")
    p.add_argument("--fps",         type=float, default=None,
                   help="Camera/video frame rate (e.g. 15 to halve the CPU load of the "
                        "video and of a remote-desktop viewer like AnyDesk)")
    return p.parse_args()


def choose_display(cfg, requested) -> str:
    """'none' (headless), 'gpu' or 'opencv'."""
    if cfg.headless:
        return "none"
    mode = requested or cfg.display_backend
    if mode == "opencv":
        return "opencv"
    if cfg.demo_mode:
        if mode == "gpu":
            logger.warning("[MAIN] GPU display needs the real camera — using the OpenCV display in demo mode")
        return "opencv"
    from display.gpu_display import available
    ok, why = available()
    if ok:
        return "gpu"
    if mode == "gpu":
        logger.warning(f"[MAIN] GPU display unavailable ({why}) — using the OpenCV display")
    return "opencv"


def opencv_loop(pipeline, display, latest, inf_thread, fps):
    # ── Display loop — at the camera frame rate ───────────────────
    # On every tick we grab the FRESHEST RGB frame straight from
    # RGBCamera (already thread-safe) so video plays smoothly even
    # while the inference thread is mid-YOLO.  Detection overlays
    # are at most one inference cycle stale but visually fine.
    target_dt  = 1.0 / fps

    # Periodic perf log — cheap enough to leave on permanently, and
    # is the actual evidence (not just an assumption) that the
    # display loop isn't waiting on YOLO/thermal/radar/fusion/tracker.
    PERF_LOG_INTERVAL_S = 3.0
    last_perf_log = 0.0

    while inf_thread.is_alive():
        t0 = time.perf_counter()

        fd = latest()

        if fd is not None:
            # Always use the freshest camera frame for smooth video
            fresh_rgb = pipeline.rgb.read()
            if fresh_rgb is not None:
                fd = dataclasses.replace(fd, rgb_frame=fresh_rgb)

            display.render(fd)
            if display.should_quit():
                break

            now = time.time()
            if now - last_perf_log >= PERF_LOG_INTERVAL_S:
                last_perf_log = now
                cam_ts   = pipeline.rgb.capture_ts
                tm       = fd.timing_ms
                logger.info(
                    "[PERF] camera=%.1ffps display=%.1ffps inference=%.1ffps | "
                    "yolo=%.0fms sensors=%.0fms fusion=%.0fms tracker=%.0fms | "
                    "video_frame_age=%.0fms detection_age=%.0fms | blocked_on_ml=NO",
                    pipeline.rgb.fps, display.video_fps, fd.fps,
                    tm.get("yolo_ms", 0.0), tm.get("sensors_ms", 0.0),
                    tm.get("fusion_ms", 0.0), tm.get("tracker_ms", 0.0),
                    (now - cam_ts) * 1000 if cam_ts else 0.0,
                    (now - fd.timestamp) * 1000,
                )

        elapsed  = time.perf_counter() - t0
        sleep_t  = target_dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("  RESCUE DRONE v3 — Human Detection")
    logger.info("  BotWing F722 | MLX90640 | Pi Cam 3 | HLK-LD2450")
    logger.info("=" * 60)

    overrides = {
        "ld2410_port":         args.mmwave_port,
        "fc_port":             args.fc_port,
        "fc_baud":             args.fc_baud,
        "yolo_model_path":     args.yolo_model,
        "yolo_conf_threshold": args.conf,
    }
    cfg = Config(
        fc_enabled=args.fc_enabled,
        record=args.record,
        headless=args.headless,
        demo_mode=args.demo,
        **{k: v for k, v in overrides.items() if v is not None},
    )
    if args.no_fullscreen:
        cfg.fullscreen = False
    if not cfg.headless:
        profile = apply_display_profile(cfg, args.profile)
        if args.fps:
            cfg.rgb_fps = args.fps
        logger.info(f"[MAIN] Output profile: {profile} ({cfg.display_width}x{cfg.display_height} "
                    f"@ {cfg.rgb_fps}fps, OSD style {cfg.osd_style})")

    mode = choose_display(cfg, args.display)
    if mode == "gpu":
        # Display process (camera + GPU video) with the sensors/ML/OSD in a
        # worker process — see display/gpu_display.py for why.
        from display.gpu_display import run_gpu, GpuDisplayUnavailable
        logger.info("[MAIN] Display: gpu")
        try:
            rc = run_gpu(cfg)
            logger.info("Shut down cleanly." if rc == 0 else f"Shut down — worker exit code {rc}.")
            return rc
        except GpuDisplayUnavailable as e:
            logger.warning(f"[MAIN] GPU display failed to start ({e}) — using the OpenCV display")
            mode = "opencv"
    logger.info(f"[MAIN] Display: {mode}")

    pipeline = DetectionPipeline(cfg)
    display  = RescueDisplay(cfg) if mode == "opencv" else None
    screen   = ScreenBlankingInhibitor()
    inf_thread = None
    # Non-zero on a fatal error so systemd's Restart=on-failure restarts us.
    rc = 0

    try:
        logger.info("Initializing all systems...")
        pipeline.initialize()
        logger.info("Ready. Q=quit  S=snapshot")

        if mode == "none":
            # ── Headless: synchronous pipeline loop ──────────────
            for fd in pipeline.run():
                pass

        else:
            # An idle desktop blanks the screen after 10 min — black goggles mid-flight.
            screen.start()

            # ── Inference thread (thermal-gated, ~1-2 fps) ───────
            # Runs YOLO / anomaly / fusion / tracking in the background.
            _state: dict = {"fd": None, "failed": False}
            _lock = threading.Lock()

            def _inference_loop():
                try:
                    for fd in pipeline.run():
                        with _lock:
                            _state["fd"] = fd
                except Exception:
                    logger.exception("[MAIN] Inference thread failed")
                    _state["failed"] = True

            def latest():
                with _lock:
                    return _state["fd"]

            inf_thread = threading.Thread(
                target=_inference_loop, daemon=True, name="inference"
            )
            inf_thread.start()

            opencv_loop(pipeline, display, latest, inf_thread, cfg.rgb_fps)
            if _state["failed"]:
                rc = 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        rc = 1
    finally:
        # Stop and join the inference thread BEFORE tearing anything down —
        # exiting while YOLO is mid-inference aborts the process.
        pipeline.stop()
        if inf_thread is not None:
            inf_thread.join(timeout=5.0)
        pipeline.shutdown()
        if display:
            display.close()
        screen.restore()
        logger.info("Shut down cleanly." if rc == 0 else "Shut down after a fatal error.")
    return rc


if __name__ == "__main__":
    sys.exit(main())

