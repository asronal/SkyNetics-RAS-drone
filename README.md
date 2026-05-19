# 🚁 Rescue Drone v3
**Avalanche & Landslide Autonomous Human Detection System**

![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-C51A4A?style=for-the-badge&logo=raspberry-pi)
![Language](https://img.shields.io/badge/Language-Python%203.9+-3776AB?style=for-the-badge&logo=python)
![AI](https://img.shields.io/badge/AI-YOLOv8%20ONNX-FF6F00?style=for-the-badge&logo=yolo)
![Hardware](https://img.shields.io/badge/Sensors-MLX90640%20%7C%20LD2450%20%7C%20PiCAM3-10B981?style=for-the-badge)

<p align="center">
  <img src="assets/osd_showcase.png" alt="Rescue Drone OSD Interface" width="100%">
</p>

Every second counts when someone is buried under snow or debris. That thought is what pushed us to build this.

**Rescue Drone v3** is something we poured a lot of late nights into — a fully autonomous, multi-sensor aerial system that can find survivors in avalanches and landslides when it's too dangerous or too slow for rescuers to search on foot. It fuses **thermal imaging, mmWave radar, and RGB vision** to detect human presence from above, runs completely offline on a Raspberry Pi 4, and beams a live augmented overlay straight to the pilot's goggles through an analog VTX.

No internet required. No fancy server. Just a drone, some clever sensor fusion, and the hope that it gets there in time.

---

## 👥 The Team

Four people, one shared obsession with getting this thing to actually work. Everyone brought something different to the table.

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/asronal">
        <img src="https://github.com/asronal.png" width="90" height="90" style="border-radius: 50%"><br><br>
        <b>Asronal</b>
      </a><br>
      <sub>Hardware & Integrations</sub>
    </td>
    <td align="center">
      <a href="https://github.com/AnujD21">
        <img src="https://github.com/AnujD21.png" width="90" height="90" style="border-radius: 50%"><br><br>
        <b>Anuj D</b>
      </a><br>
      <sub>Software</sub>
    </td>
    <td align="center">
      <a href="https://github.com/vishal6626">
        <img src="https://github.com/vishal6626.png" width="90" height="90" style="border-radius: 50%"><br><br>
        <b>Vishal</b>
      </a><br>
      <sub>Model Training</sub>
    </td>
    <td align="center">
      <a href="https://github.com/Akilan12335">
        <img src="https://github.com/Akilan12335.png" width="90" height="90" style="border-radius: 50%"><br><br>
        <b>Akilan S</b>
      </a><br>
      <sub>Hardware Assembly</sub>
    </td>
  </tr>
</table>

### 🎥 See It In Action

<p align="center">
  <a href="https://github.com/asronal/SkyNetics-rescue-drone/raw/main/assets/demo.mp4">
    <img src="assets/osd_main.jpg" alt="Click to watch System Demo Video" width="100%">
  </a>
  <br>
  <em>Click above to watch the full system demo</em>
</p>

---

## 🏗️ How We Built It

<p align="center">
  <img src="assets/drone_architecture.png" alt="Drone System Architecture" width="100%">
</p>

One of the things we were most careful about was keeping everything modular. The AI pipeline, the display layer, and the hardware drivers don't know much about each other — which saved us a ton of headaches when a sensor misbehaved or we needed to swap out a model. Each piece does its job and gets out of the way.

```text
rescue_drone_osd_fixed/
├── main.py                    # Where everything kicks off
├── config.py                  # Central hub for all tuning — thresholds, pinouts, sizes
├── rescue_drone.service       # Systemd service so the drone boots straight into mission mode
├── requirements.txt           # Python dependencies
│
├── pipeline/                  
│   └── detection_pipeline.py  # The orchestrator — syncs all sensors & AI every single frame
│
├── display/                   
│   ├── rescue_display.py      # Fullscreen OpenCV UI with PiP overlays
│   └── osd.py                 # Draws telemetry, bounding boxes, and the radar scope
│
├── ml/                        
│   ├── models.py              # YOLO detector, thermal anomaly scanner, SORT tracker, sensor fusion
│   ├── thermal_isolation.py   # Cleans up thermal noise so real heat signatures stand out
│   └── detection.py           # Shared Target dataclass passed through the whole pipeline
│
├── sensors/                   
│   ├── rgb_camera.py          # Pi Cam 3 feed via libcamera
│   ├── thermal_camera.py      # Reads & decodes the MLX90640 over I2C
│   ├── ld2450_radar.py        # Parses the LD2450's UART data stream
│   └── flight_controller.py   # Pulls MAVLink telemetry from the BotWing F722
│
├── models/                    # Drop your .onnx weights here (e.g. rgb_human.onnx)
└── output/                    # Auto-snapshots and mission recordings go here
```

---

## 🔌 Wiring It All Together

Getting the hardware right was honestly half the battle. Here's exactly how everything connects.

### 1. Analog VTX Output (Composite Video)

The Pi 4's 3.5mm TRRS jack carries a composite video signal. Wire it to your VTX and the OSD appears live in the pilot's goggles — no HDMI capture card, no latency, just analog video the way FPV was meant to be.

* **Tip**: Audio Left
* **Ring 1**: Audio Right
* **Ring 2**: GND `➔ VTX Ground`
* **Sleeve**: Video `➔ VTX Video-IN`

> **Important:** The Pi 4 turns composite output off by default. Add these two lines to `/boot/firmware/config.txt` to enable it:
> ```
> enable_tvout=1
> sdtv_mode=2   # PAL — switch to 0 for NTSC regions
> ```

### 2. MLX90640 Thermal Sensor (I2C)

<p align="center">
  <img src="assets/thermal_demo.jpg" alt="Thermal Sensor Demo" width="80%">
</p>

The thermal array is the heart of the survivor detection pipeline. It connects over I2C — dead simple wiring.

* **VCC** `➔` Pin 1 (3.3V)
* **GND** `➔` Pin 6 (GND)
* **SDA** `➔` Pin 3 (GPIO 2, I2C1)
* **SCL** `➔` Pin 5 (GPIO 3, I2C1)

### 3. HLK-LD2450 mmWave Radar (UART0)

* **TX** `➔` Pin 10 (GPIO 15, UART0 RX)
* **RX** `➔` Pin 8  (GPIO 14, UART0 TX)
* **VCC** `➔` Pin 2  (5V)
* **GND** `➔` Pin 14 (GND)

> You'll need to disable onboard Bluetooth to free up UART0. Worth it — the radar adds a whole extra dimension to detection confidence.

### 4. BotWing F722 Flight Controller (UART1 or UART2)

* **F722 TX** `➔` RPi RX (GPIO 1 / Pin 28 for UART2)
* **F722 RX** `➔` RPi TX (GPIO 0 / Pin 27 for UART2)
* **GND** `➔` Shared GND

> In iNav, head to the Ports tab and enable MSP on this UART at 115200 baud.

### 5. Pi Camera Module 3 (CSI)

Ribbon cable into the primary `CAM` port. Silver contacts face the HDMI ports. That's it.

---

## 🛠️ Getting the Pi Ready

We ran this on a freshly imaged Pi 4 with Raspberry Pi OS Lite. Here's the full setup from scratch:

```bash
# 1. Update everything and grab the system-level drivers
sudo apt update
sudo apt install -y python3-opencv python3-picamera2 python3-pip

# 2. Install the Python packages
pip3 install -r requirements.txt
pip3 install onnxruntime filterpy scipy pyserial
pip3 install adafruit-circuitpython-mlx90640 adafruit-blinka

# 3. Lock the CPU to performance mode — don't skip this
#    Without it the Pi throttles under load and the AI pipeline slows to a crawl
sudo apt install cpufrequtils
sudo cpufreq-set -g performance

# 4. Verify everything showed up correctly
i2cdetect -y 1                  # Look for 0x33 — that's the MLX90640
ls /dev/ttyAMA0                 # Should exist if the radar is wired in
libcamera-hello --list-cameras  # Should list IMX708 (Pi Cam 3)
```

---

## 🚀 Launching the System

The default launch goes straight to fullscreen. The display is tuned to fill the composite output without any desktop UI bleeding in — exactly what you want when it's streaming to goggles.

```bash
# Standard flight deployment
python3 main.py

# With live FC telemetry from the flight controller
python3 main.py --fc-enabled

# Windowed mode — great for development over VNC
python3 main.py --no-fullscreen

# Record the full mission to disk
python3 main.py --record

# SSH session, no display output needed
python3 main.py --headless

# No hardware nearby? Synthetic data mode lets you test the full pipeline
python3 main.py --demo
```

### Keyboard Shortcuts

Handy when you're testing over VNC or have a keyboard plugged in:

| Key | Action |
|-----|--------|
| `V` | Cycle main view: RGB → Thermal → Radar |
| `T` | Toggle the thermal picture-in-picture |
| `M` | Cycle thermal isolation mode (Highlight / Silhouette / Contour) |
| `S` | Save a snapshot to `output/` |
| `Q` | Quit |

---

## 📡 Why the Radar Changes Everything

<p align="center">
  <img src="assets/radar_demo.jpg" alt="Radar Tracking Demo" width="80%">
</p>

Adding the **LD2450** was one of those decisions that made the whole system feel significantly more capable. Cameras — thermal or RGB — need line-of-sight. The radar doesn't. It can detect the micro-movement of a person breathing through snow or debris, which is exactly the scenario we're building for.

It also tracks targets through fog, smoke, and whiteout conditions where the cameras are essentially useless.

That said, we're realistic about what it can and can't do. It won't give you a 3D point cloud, it can't image anything, and it's not a replacement for the thermal array. We use it as a **presence confirmation layer** inside `SensorFusion` — when the thermal or RGB detection is borderline, a radar hit tips the confidence scale. When all three agree, you've found your survivor.

---

*Built with a lot of care, more than a few all-nighters, and the hope that this kind of technology actually makes it into the hands of rescue teams someday. If you're working on something similar or want to build on top of this — reach out.*
