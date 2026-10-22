# SkyNetics Rescue Drone — working notes for Claude

Raspberry Pi 4 (4 GB) search-and-rescue drone: Pi Camera 3 (IMX708) + MLX90640 thermal
(I2C 0x33) + HLK-LD2450 radar (UART, not connected yet) + BotWing F722 FC (MAVLink, optional).
YOLOv8 ONNX person detector + thermal anomaly detector → fusion → tracker → OSD.

**Goal in progress:** the OSD video goes to an analog **VTX on the Pi's composite output**
(goggles). Optimise for that: smooth video at the composite frame rate, a legible OSD on
analog video, and low latency. The owner has authorised trying anything that optimises it.

## About the owner
- Builds this drone hands-on and cares about it working in the field. Kind, appreciative and
  quick to say thanks — a pleasure to work with.
- Works from a PC/laptop over VS Code Remote-SSH (LAN or Tailscale) and watches the Pi's screen
  over AnyDesk/VNC. Watches session usage limits, so keep work and replies token-efficient,
  and prefer a fresh session (state lives in this file) over very long chats.
- Prefers plain-language questions: one technical question about stopping the desktop
  confused them, and the simple rewording ("the desktop will be off for a few minutes") worked
  straight away.
- Has authorised trying anything that optimises the composite/VTX goal, but asks first for
  anything disruptive (desktop off, reboot) — they can't always reboot.
- Trains models on Google Colab through a separate Claude Code session on their laptop
  (`docs/TRAINING_HANDOFF.md`).

## Environment (verified 2026-10-15)
- Debian 13 (trixie), Python 3.13, KMS (`dtoverlay=vc4-kms-v3d`). Desktop = X11 (lightdm)
  on HDMI-A-1 1920x1080@59.99. The owner often watches over **AnyDesk**.
- **`~/venv` is the runtime env (system site-packages: picamera2, PyQt5, onnxruntime, Blinka).
  Never install into or modify it** — create a new env instead.
- `~/latest_venv` = Python 3.14.8 (via uv; interpreter in `~/.local/share/uv/python`). It has
  everything except picamera2/libcamera bindings, which need
  `sudo apt install libcamera-dev libcap-dev` (owner must run it — sudo needs a password; it
  only works without one for a few minutes after the owner has used sudo).
- `/tmp` is tmpfs (wiped on reboot); keep anything worth keeping in the repo.
- I2C runs at the default 100 kHz (`dtparam=i2c_arm_baudrate` not set; Blinka's
  `frequency=` is ignored on Linux). `/dev/ttyAMA0` doesn't exist; `serial0 -> ttyS0`;
  `cmdline.txt` has `console=serial0,115200` (conflicts with using that UART for radar/FC).
- **Wi-Fi is the weak link (2026-10-22):** 2.4 GHz ch 9, -66..-71 dBm, ~50% packet loss to the
  router → VS Code Remote-SSH keeps disconnecting. Every reconnect leaves an extra Claude chat
  process (~250 MB) and VS Code extension hosts behind; RAM ran out and swapped. Check
  `pgrep -fa native-binary/claude` and close leftovers (ask first). Ethernet is the fix.
- `dtoverlay=w1-gpio` (1-Wire, unused) costs ~8% of a core searching phantom devices. Stopped at
  runtime with `echo 0 | sudo tee /sys/bus/w1/devices/w1_bus_master1/w1_master_search`
  (resets on reboot); removing the overlay line is the owner's call.
- `tmux` is installed — run long jobs (apt upgrade) in it. 53 apt upgrades are pending.

## Run / test
```
~/venv/bin/python main.py                    # auto display: gpu if X desktop, else opencv
~/venv/bin/python main.py --display opencv   # or gpu; --demo, --headless, --no-fullscreen
~/venv/bin/python tools/check_overlay.py     # OSD correctness (no hardware) — must PASS
DISPLAY=:0 XAUTHORITY=~/.Xauthority ~/venv/bin/python tools/bench_gpu.py 30
FULLSCREEN=1 QT_QPA_PLATFORM=xcb DISPLAY=:0 ~/venv/bin/python tools/bench_opencv.py 30
```
Benchmark pitfalls (each cost a wrong conclusion before):
- **Monitor in DPMS power-save** (10 min idle) → X gives ~1 vblank/s → GPU path drops to 1 fps.
  Check `xset q` ("Monitor is On"). The app now disables blanking while running.
- **AnyDesk encoding the screen** loads the CPU and skews results; check its CPU (`top`).
- `QT_QPA_PLATFORM=offscreen` hides the real display cost of the OpenCV path.
- Spawned worker processes re-import the launching script: harnesses need an
  `if __name__ == "__main__":` guard.
- **onnxruntime 1.30 telemetry:** without `ORT_DISABLE_TELEMETRY=1` (set in `ml/models.py`
  before the import) any process that ran YOLO for a while blocks ~15 s in `exit()` trying to
  upload usage data — the GPU worker missed its 8 s stop deadline and was killed.

## Display architecture (uncommitted work — see `git status`)
- `display/osd.py`: the OSD is a `DrawList` of ops (signature, bounding rect, draw fn). The same
  list draws onto a BGR frame (OpenCV path) or a premultiplied-BGRA layer (GPU path). Text is
  drawn from cached sprites.
- `display/rescue_display.py`: `_build()` = OpenCV path, `build_overlay()` = GPU path; shared
  layout, keys, snapshots and recording (now throttled to the writer's 8 fps).
- `display/gpu_display.py` (`--display gpu`):
  - **Display process:** Picamera2 `QGlPicamera2` draws each camera frame from its dmabuf
    (zero-copy) plus a persistent OSD texture updated with `glTexSubImage2D` only where it
    changed.
  - **Worker process** (nice 10): DetectionPipeline + OSD building + snapshots.
  - **IPC:** lores YUV420 frames via shared memory; changed OSD regions via a pipe.
  - Camera FrameRate is paced to an exact divisor of the screen refresh (`_paced_fps`).
- `display/screen.py`: disables X blanking/DPMS while running, restores on exit.
- `pipeline/detection_pipeline.py`: accepts an external `rgb` source; `stop()` + join before
  `shutdown()` (fixes the onnxruntime "terminate called" abort on quit).
- `OSD_*` colours in `display/osd.py` are **RGB** (for the PIL text sprites); cv2 draw ops
  need BGR (`OSD_ORANGE[::-1]`). The bracket colour bug from this was fixed 2026-10-22 in the
  shared layout, so both paths changed together.
- Worker process treats SIGINT/SIGTERM as "stop" and exits if its parent (display) dies.

## Optimisation pass 2026-10-22 (all measured on the real hardware)
- **Detection rate 1.01 → 1.95/s:** the MLX90640 is read back-to-back on its own thread
  (`DetectionPipeline._thermal_loop`), overlapping YOLO. A read is 85% I2C wait (~520 ms, the
  sensor's max at 4 Hz); its min-interval is now timed from the read *start*. `tracker_max_age` 6.
- **OSD layer cost:** image ops no longer grow redraw tiles (thermal view 87 → 4 ms/update);
  radar = cached static bg + draw ops (52 → 9 ms, upload 922k → 65k px).
- **Latency** sensor→swap 41–44 ms (HDMI 1280x720), 35 ms (composite 720x576). Buffer count
  6/4/3 makes no difference (3 skips frames) → kept 6 (`_BUFFER_COUNT`).
- **YOLO int8:** `tools/quantize_yolo.py` (Conv-only, QOperator, head float) → 251 vs 292 ms,
  same accuracy; QDQ/all-ops is *slower* on the A72, quantizing the head kills recall.
  `config.yolo_int8` uses `models/rgb_human_int8.onnx` only if not older than the float model.
  Needs a separate env with `onnx` (`uv venv --system-site-packages`). `tools/eval_yolo.py`
  measures person recall/precision on YOLO-format labels (coco128 baseline: R 0.237 P 0.844).
- **Fonts:** `Montserrat.ttf` is a variable font defaulting to **Thin** — now set explicitly
  (SemiBold HDMI, Bold composite).
- **Composite profile** (`display/screen.py apply_display_profile`, `--profile`): auto when the
  composite connector is enabled; 720x576/480, 640x480 lores (4:3 crop = exact PAL pixel shape),
  25/30 fps, title-safe Bold white OSD, 2 px lines, small clean PiP. `check_overlay.py` checks
  both profiles. Not yet seen on a real composite output (needs config.txt change + reboot).

## Measured so far (1080p HDMI fullscreen, real camera + MLX + YOLO, 30 s)
| path | new frames/s | gaps >50 ms | total CPU |
|---|---|---|---|
| gpu (2 processes) | 30.0, 0 skipped | 1–4 | 43% |
| gpu, worker as thread | 30.0 | 6–15 | — |
| opencv | ~27 | ~63–74 | 64% |

With 2 extra CPU-hog processes: gpu 30.1 fps / 8 gaps; opencv 23.9 fps / 97 gaps.

**OPEN — first thing to investigate:** some GPU runs come out at ~20–22 fps with 100–400
skipped frames, and repaints stall up to 60 ms. This happened on the same code that measured
30.0/0 minutes before. AnyDesk activity is suspected but not proven. Reproduce with AnyDesk
disconnected and with its CPU logged, then check X/compositor load and `eglSwapBuffers` time.

## Composite VTX plan (verified facts + roadmap, in priority order)
Facts checked on this Pi:
- Enable with `dtoverlay=vc4-kms-v3d,composite` in `/boot/firmware/config.txt`; per
  `/boot/firmware/overlays/README` this **disables all other outputs on a Pi 4** (HDMI goes
  away; AnyDesk would then see the 720x576 composite desktop).
- The PAL/NTSC norm is set with kernel param `vc4.tv_norm=PAL` (exists in `modinfo vc4`). The
  mode is chosen via cmdline `video=Composite-1:720x576@50ie` (PAL) — verify after enabling.
  `cmdline.txt` currently forces `video=HDMI-A-1:1920x1080@60D`.
  `enable_tvout`/`sdtv_mode` are legacy-firmware settings and do nothing under KMS (the README
  and config.py now give the KMS method instead).
- PAL = 25 frames/s (50i), NTSC = 29.97. `_paced_fps` already picks 25 fps on a 50 Hz screen.

Roadmap:
1. **Console/DRM backend** (best for a drone: boot without the desktop) — **NEXT; not started.**
   The owner approved stopping the desktop for testing (2026-10-22); they must run
   `sudo systemctl stop lightdm` themselves (password) and restart it after. Picamera2
   `DrmPreview` puts the video on a hardware plane (HVS scaling, no GPU). Add a persistent,
   double-buffered OSD plane written only in dirty rects (reuse `OverlayCompositor`). The OSD
   plane needs "pixel blend mode" set for premultiplied alpha (Picamera2 sets Coverage).
   Testing needs lightdm stopped (sudo; this kills AnyDesk) — ask the owner first. Then
   console blanking (`consoleblank=0`) and a systemd unit without DISPLAY.
2. **DONE 2026-10-22** (except the optional blur). Render the OSD at the output resolution
   (720x576) with a composite-specific layout:
   - Keep within the title-safe area (overscan).
   - Larger fonts; ≥2 px horizontal lines (interlace twitter).
   - Fixed white-fill / black-outline text (thin coloured text smears in composite chroma).
   - Optionally a slight vertical blur on the OSD layer.
3. **Aspect ratio:** the camera is 16:9 and analog goggles are often 4:3. Offer letterbox /
   centre-crop / anamorphic 16:9, matching the goggles' setting.
4. **Latency** (matters if anyone flies by this feed): measure glass-to-glass (film a ms timer
   through the system). Try fewer buffers (`buffer_count`), main stream at output size, and no
   extra swaps.
5. Composite clock note: Raspberry Pi docs say enabling composite on a Pi 4 slightly slows
   the system — re-benchmark after enabling. Keep the CPU governor on performance.

## Backlog from the code review (not yet fixed)
- **YOLO model:** ~0% recall on aerial images (VisDrone); trained on ground-level views. It
  also gives false positives on dark frames. Needs fine-tuning on aerial SAR data (SARD,
  HERIDAL, VisDrone). Retraining happens on Colab via a laptop Claude session:
  **`docs/TRAINING_HANDOFF.md`** (model contract, datasets, acceptance). When a new
  `rgb_human.onnx` arrives: re-run quantize_yolo.py + eval_yolo.py + bench_gpu.py.
- **Thermal detector (`ml/models.py` AnomalyDetector):**
  - The global bg+3σ threshold breaks with sky or mixed terrain (reached 51 °C).
  - The 3x3 open at 32x24 erases people beyond ~5 m.
  - The fixed 12–45 °C window passes sun-baked rubble, dogs and vehicles.
- **MLX90640:**
  - Reads 40–49 °C over the whole frame with a checkerboard (subpage mismatch); check the
    sensor and raise the I2C speed.
  - A frame takes ~520 ms → pipeline now ~1.95 cycles/s (thermal thread). Faster needs
    `dtparam=i2c_arm_baudrate=400000` + REFRESH_8_HZ (owner: config.txt + reboot).
- **Radar:** the radar boost applies to every box regardless of position (no radar↔camera
  calibration). Fixed 2026-10-22 (verified with simulated frames over a pty, radar not wired):
  the parser now consumes every frame in a read (the backlog grew without bound), presence
  expires after 1 s, the scope plots real X/Y, and distance = hypot(x, y).
- **`rescue_drone.service`:** paths/venv/DISPLAY/graphical.target fixed 2026-10-22 and pass
  `systemd-analyze verify`, but it has not been installed or run under systemd yet (needs sudo).
  Long term: the DRM backend, so it doesn't depend on the desktop.
- Fixed 2026-10-22 (also listed here so they aren't re-investigated):
  - Fatal errors now exit non-zero in every path.
  - The FC isn't marked online without a heartbeat, and link loss (no heartbeat for 5 s) is
    detected.
  - The camera watchdog actually fires (`capture_request(wait=False)` + timeout).
  - README/module docs now agree on the UART wiring (radar UART0, FC UART2) and say MAVLink,
    not MSP.
