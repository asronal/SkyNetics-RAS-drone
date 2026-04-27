"""
Rescue Drone v3 — System Configuration
Hardware: BotWing F722 | MLX90640 | RPi Cam Module 3 | HLK-LD2410C-P
"""

from dataclasses import dataclass
from pathlib import Path


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
    # Sony IMX708, 12MP — captured at 640×480 for display
    # Uses libcamera backend (NOT cv2.VideoCapture)
    # Wiring: Connect ribbon cable to Raspberry Pi 4's primary CSI camera port. Silver contacts facing HDMI ports.
    rgb_width: int = 640
    rgb_height: int = 480
    rgb_fps: int = 30

    # ── HLK-LD2410C-P Presence Radar (UART) ──────────────────────
    # 24GHz FMCW — detects moving + stationary targets
    # Outputs: presence flag, distance (cm), signal strength
    # Does NOT produce point cloud or Doppler velocity
    # Wiring: TX→RPi RX(GPIO15/Pin10), RX→RPi TX(GPIO14/Pin8), VCC→5V, GND→GND
    ld2410_port: str = "/dev/ttyAMA0"   # RPi UART0 (disable BT first)
    ld2410_baud: int = 256000           # LD2410C default baud
    ld2410_max_range_cm: int = 600      # configure per environment (max 600cm)

    # ── BotWing F722 Flight Controller (MAVLink) ──────────────────
    # Sends detection alerts to FC → triggers LED/buzzer on drone
    # MAVLink uses a second hardware UART on the Pi 4.
    # Wiring: F722 TX → RPi RX (e.g., GPIO1 / Pin 28 for UART2), F722 RX → RPi TX (e.g., GPIO0 / Pin 27 for UART2), GND → RPi GND
    fc_enabled: bool = False            # set True when FC UART wired up
    fc_port: str = "/dev/ttyAMA1"       # RPi UART1/UART2 depending on overlay
    fc_baud: int = 115200              # INAV MAVLink default baud
    fc_heartbeat_interval_sec: float = 5.0  # reconnect retry interval on UART error

    # ── ML: YOLOv8n ONNX ─────────────────────────────────────────
    yolo_model_path: str = "models/rgb_human.onnx"
    yolo_input_size: int = 320          # Must match the locked ONNX export shape (320x320)
    yolo_conf_threshold: float = 0.10   # Dropped drastically to allow partial face detection
    yolo_iou_threshold: float = 0.45
    yolo_every_n_frames: int = 1        # run every frame for maximum responsiveness

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
    tracker_max_age: int = 12           # frames (3s at 4fps)
    tracker_min_hits: int = 1
    tracker_iou_thresh: float = 0.25

    # ── Display (VTX Composite Out) ──────────────────────────────
    # Outputting to an analog VTX uses the Pi 4's composite 3.5mm TRRS jack or test pads.
    # Wiring (TRRS Jack): Tip=Audio L, Ring1=Audio R, Ring2=GND, Sleeve=Composite Video.
    # Connect Sleeve to VTX Video-IN, and Ring2 to VTX GND.
    # Require '/boot/firmware/config.txt' set 'enable_tvout=1' and 'sdtv_mode=2' (if PAL) or 0 (if NTSC).
    display_width: int = 1280
    display_height: int = 720
    fullscreen: bool = True             # Maximize window for composite VTX output
    headless: bool = False
    record: bool = False
    output_video: str = "output/recording.mp4"

    # ── Alerts & Snapshots ───────────────────────────────────────
    alert_cooldown_sec: float = 4.0
    snapshot_enabled: bool = True
    snapshot_dir: str = "output"
    snapshot_cooldown: float = 5.0

    # ── Misc ─────────────────────────────────────────────────────
    demo_mode: bool = False

    def __post_init__(self):
        Path("output").mkdir(exist_ok=True)
        Path(self.snapshot_dir).mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("config").mkdir(exist_ok=True)
