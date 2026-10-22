"""
Rescue Drone v3 — System Configuration
Hardware: BotWing F722 | MLX90640 | RPi Cam Module 3 | HLK-LD2410C-P
"""

from dataclasses import dataclass
from pathlib import Path

# Anchor every relative output path (logs, snapshots, models, config) to
# THIS file's own directory instead of the process's current working
# directory. Bare relative paths like "output/" or "rescue.log" resolve
# against whatever's CWD when the script is launched — cd into the repo
# and it's fine, but launch via an absolute path from $HOME, a systemd
# WorkingDirectory that doesn't match the actual clone location, or a cron
# job, and everything scatters into that directory instead of the repo.
BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Config:

    # ── MLX90640 Thermal Camera (I2C) ─────────────────────────────
    # 32×24 pixels, 55° FOV, I2C address 0x33
    # Wiring: SDA→Pin3, SCL→Pin5, VCC→3.3V(Pin1), GND→Pin6
    thermal_fps: int = 4
    thermal_min_temp: float = -20.0   # °C cold snow background
    thermal_max_temp: float = 40.0    # °C human body ~37°C
    thermal_colormap: str = "inferno"

    # ── RPi Camera Module 3 (CSI / libcamera) ─────────────────────
    # Sony IMX708, 12MP — captured at 640×360 for display
    # 16:9 to match display_width/display_height (and the IMX708's native
    # 2304x1296 full-frame readout) — keeps the RGB resize in rescue_display
    # a uniform scale instead of stretching a 4:3 capture into a 16:9 canvas.
    # Uses libcamera backend (NOT cv2.VideoCapture)
    # Wiring: Connect ribbon cable to Raspberry Pi 4's primary CSI camera port. Silver contacts facing HDMI ports.
    rgb_width: int = 640
    rgb_height: int = 360
    rgb_fps: int = 30

    # ── HLK-LD2410C-P Presence Radar (UART) ──────────────────────
    # 24GHz FMCW — detects moving + stationary targets
    # Outputs: presence flag, distance (cm), signal strength
    # Does NOT produce point cloud or Doppler velocity
    # Wiring: TX→RPi RX(GPIO15/Pin10), RX→RPi TX(GPIO14/Pin8), VCC→5V, GND→GND
    ld2410_port: str = "/dev/ttyAMA0"   # RPi UART0 (disable BT first)
    ld2410_baud: int = 256000           # LD2410C default baud
    ld2410_max_range_cm: int = 800      # LD2450 max tracking range is 8m (radar scope scale)

    # ── BotWing F722 Flight Controller (MAVLink) ──────────────────
    # Sends detection alerts to FC → triggers LED/buzzer on drone
    # MAVLink uses a second hardware UART on the Pi 4.
    # Wiring: F722 TX → RPi RX (e.g., GPIO1 / Pin 28 for UART2), F722 RX → RPi TX (e.g., GPIO0 / Pin 27 for UART2), GND → RPi GND
    fc_enabled: bool = False            # set True when FC UART wired up
    fc_port: str = "/dev/ttyAMA1"       # RPi UART1/UART2 depending on overlay
    fc_baud: int = 115200              # INAV MAVLink default baud
    fc_heartbeat_interval_sec: float = 5.0  # reconnect retry interval on UART error

    # ── ML: YOLOv8n ONNX ─────────────────────────────────────────
    yolo_model_path: str = str(BASE_DIR / "models" / "rgb_human.onnx")
    yolo_input_size: int = 320          # Must match the locked ONNX export shape (320x320)
    # RPi4 has 4 cores total. Letting ONNX Runtime's thread pool claim all 4
    # during every YOLO call starves the camera/display/sensor threads of
    # CPU for that ~200ms window, which shows up as video stutter in sync
    # with inference even though the threads are architecturally decoupled.
    # Leave at least one core free for everything else.
    # Measured on the Pi 4 with the full pipeline running: detection rate is
    # set by the ~0.5s MLX90640 read, not YOLO — 1 thread gave the same
    # 1.06 inference fps as 2 threads (1.08) while the 30fps display loop
    # went from 19.4fps to 23.1fps (worst frame gap 299ms -> 140ms).
    yolo_intra_threads: int = 1
    # Highest-scoring spurious detection over 548 VisDrone frames was 0.32;
    # clearly visible people score 0.5-0.7. 0.35 rejects every observed
    # false positive without losing any of those true detections.
    yolo_conf_threshold: float = 0.35
    yolo_iou_threshold: float = 0.45
    yolo_every_n_frames: int = 1        # run every frame for maximum responsiveness
    # Preprocessing must match how rgb_human.onnx was actually trained/exported.
    # True (default) = letterbox (pad to preserve aspect ratio) — correct if
    # trained via the standard Ultralytics dataloader.
    # False = plain stretch-resize straight to yolo_input_size, no padding —
    # try this if boxes are consistently offset (usually vertically, on a
    # 16:9 camera feed) under letterbox mode; that symptom matches a model
    # trained with a plain resize instead of letterboxing.
    yolo_letterbox: bool = True
    # Use <model>_int8.onnx (made by tools/quantize_yolo.py) when it exists and
    # is not older than the float model: 251ms vs 292ms per frame on the Pi 4
    # with the same accuracy (coco128 held-out people). Older = stale after a
    # retrain, so the float model is used instead (with a warning).
    yolo_int8: bool = True

    # ── ML: Anomaly Detector ─────────────────────────────────────
    anomaly_enabled: bool = True
    anomaly_min_area_px: int = 150
    anomaly_min_conf: float = 0.30

    # ── Sensor Fusion weights ────────────────────────────────────
    weight_thermal: float = 0.65        # thermal is primary
    weight_radar: float = 0.25          # LD2410 presence confirmation
    weight_anomaly: float = 0.10
    fusion_iou_merge: float = 0.30

    # ── SORT Tracker ─────────────────────────────────────────────
    # Inference is gated by the MLX90640 (one frame per ~520ms at REFRESH_4_HZ),
    # so the pipeline runs at ~1.9 cycles/s, not 4 — ages are counted in cycles.
    # 6 cycles ≈ 3s before a lost track's box disappears.
    tracker_max_age: int = 6
    tracker_min_hits: int = 1
    tracker_iou_thresh: float = 0.25

    # ── Display (VTX Composite Out) ──────────────────────────────
    # Outputting to an analog VTX uses the Pi 4's composite 3.5mm TRRS jack or test pads.
    # Wiring (TRRS Jack): Tip=Audio L, Ring1=Audio R, Ring2=GND, Sleeve=Composite Video.
    # Connect Sleeve to VTX Video-IN, and Ring2 to VTX GND.
    # Enable with 'dtoverlay=vc4-kms-v3d,composite' in /boot/firmware/config.txt (turns HDMI off
    # on a Pi 4) and 'vc4.tv_norm=PAL' or NTSC in cmdline.txt. ('enable_tvout' / 'sdtv_mode' are
    # legacy-firmware settings and have no effect under the KMS driver.)
    display_width: int = 1280
    display_height: int = 720
    fullscreen: bool = True             # Maximize window for composite VTX output
    # "auto": "composite" when the Pi's composite output is active, else "hdmi".
    # "composite" (display/screen.py apply_display_profile) switches to the
    # output's own resolution (720x576 PAL / 720x480 NTSC), 4:3 camera streams
    # (the camera's 16:9 picture is centre-cropped, which also gives PAL's
    # pixel shape exactly), the output's frame rate and the composite OSD style.
    display_profile: str = "auto"
    osd_style: str = "hdmi"             # set by the display profile
    # "gpu": camera video composited by the GPU straight from the camera's
    #        buffers (every camera frame, zero-copy), OSD as a separate layer —
    #        needs a desktop (X11) session. Measured fullscreen on 1080p HDMI
    #        with the full ML pipeline: 30.0fps with 0 frames skipped and ~3
    #        gaps >50ms per 30s at 43% total CPU, vs "opencv" 27fps with ~63
    #        gaps >50ms at 64% CPU.
    # "opencv": OSD drawn into each frame and shown in a cv2 window.
    # "auto": gpu when available, otherwise opencv (always opencv in demo mode).
    display_backend: str = "auto"
    headless: bool = False
    record: bool = False
    output_video: str = str(BASE_DIR / "output" / "recording.mp4")

    # ── Alerts & Snapshots ───────────────────────────────────────
    alert_cooldown_sec: float = 4.0
    snapshot_enabled: bool = True
    snapshot_dir: str = str(BASE_DIR / "output")
    snapshot_cooldown: float = 5.0

    # ── Misc ─────────────────────────────────────────────────────
    demo_mode: bool = False

    def __post_init__(self):
        (BASE_DIR / "output").mkdir(exist_ok=True)
        Path(self.snapshot_dir).mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "models").mkdir(exist_ok=True)
        (BASE_DIR / "config").mkdir(exist_ok=True)
