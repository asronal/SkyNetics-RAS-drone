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

import sys
import argparse
import dataclasses
import logging
import threading
import time

from config import Config
from pipeline.detection_pipeline import DetectionPipeline
from display.rescue_display import RescueDisplay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("rescue.log"),
    ],
)
logger = logging.getLogger("main")


def parse_args():
    p = argparse.ArgumentParser(description="Rescue Drone v3")
    # Wiring LD2450/2410 Radar: TX→RPi RX(GPIO15/Pin10), RX→RPi TX(GPIO14/Pin8), VCC→5V, GND→GND
    p.add_argument("--mmwave-port", type=str,   default="/dev/ttyAMA0", help="LD2450 UART port")
    # Wiring BotWing F722 FC: FC TX→RPi RX (e.g., UART1/2), FC RX→RPi TX, GND→GND
    p.add_argument("--fc-port",     type=str,   default="/dev/ttyAMA1", help="F722 UART port")
    p.add_argument("--fc-baud",     type=int,   default=115200,         help="FC baud rate")
    p.add_argument("--fc-enabled",  action="store_true",                help="Enable FC telemetry")
    p.add_argument("--yolo-model",  type=str,   default="models/rgb_human.onnx")
    p.add_argument("--conf",        type=float, default=0.10,           help="YOLO confidence threshold")
    p.add_argument("--record",      action="store_true",                help="Save output as video")
    p.add_argument("--headless",    action="store_true",                help="No display window (SSH)")
    p.add_argument("--demo",        action="store_true",                help="Synthetic data — no hardware")
    p.add_argument("--no-fullscreen",action="store_true",               help="Disable fullscreen mode")
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("  RESCUE DRONE v3 — Human Detection")
    logger.info("  BotWing F722 | MLX90640 | Pi Cam 3 | HLK-LD2450")
    logger.info("=" * 60)

    cfg = Config(
        ld2410_port=args.mmwave_port,
        fc_port=args.fc_port,
        fc_baud=args.fc_baud,
        fc_enabled=args.fc_enabled,
        yolo_model_path=args.yolo_model,
        yolo_conf_threshold=args.conf,
        record=args.record,
        headless=args.headless,
        demo_mode=args.demo,
    )
    if args.no_fullscreen:
        cfg.fullscreen = False

    pipeline = DetectionPipeline(cfg)
    display  = RescueDisplay(cfg) if not cfg.headless else None

    try:
        logger.info("Initializing all systems...")
        pipeline.initialize()
        logger.info("Ready. Q=quit  S=snapshot")

        if cfg.headless:
            # ── Headless: synchronous pipeline loop ──────────────
            for fd in pipeline.run():
                pass

        else:
            # ── Inference thread (thermal-gated, ~4 fps) ─────────
            # Runs YOLO / anomaly / fusion / tracking in the background.
            _state: dict = {"fd": None}
            _lock = threading.Lock()

            def _inference_loop():
                for fd in pipeline.run():
                    with _lock:
                        _state["fd"] = fd

            inf_thread = threading.Thread(
                target=_inference_loop, daemon=True, name="inference"
            )
            inf_thread.start()

            # ── Display loop — up to 30 fps ───────────────────────
            # On every tick we grab the FRESHEST RGB frame straight from
            # RGBCamera (already thread-safe) so video plays smoothly even
            # while the inference thread is mid-YOLO.  Detection overlays
            # are at most one inference cycle stale but visually fine.
            TARGET_FPS = 30
            target_dt  = 1.0 / TARGET_FPS

            while inf_thread.is_alive():
                t0 = time.perf_counter()

                with _lock:
                    fd = _state["fd"]

                if fd is not None:
                    # Always use the freshest camera frame for smooth video
                    fresh_rgb = pipeline.rgb.read()
                    if fresh_rgb is not None:
                        fd = dataclasses.replace(fd, rgb_frame=fresh_rgb)

                    display.render(fd)
                    if display.should_quit():
                        break

                elapsed  = time.perf_counter() - t0
                sleep_t  = target_dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.exception(f"Fatal: {e}")
    finally:
        pipeline.shutdown()
        if display:
            display.close()
        logger.info("Shut down cleanly.")


if __name__ == "__main__":
    main()

