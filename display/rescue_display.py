"""
Rescue Display — integrates ThermalIsolator and OSD overlay.

Full-screen : Selectable via `V` key (RGB, Thermal, Radar)
PiP (toggle) : thermal view with human-only isolation in bottom-left

Key controls:
  V — cycle main view (RGB / Thermal / Radar)
  T — toggle Thermal PiP
  M — cycle thermal display mode (highlight / silhouette / contour)
  S — save snapshot
  Q — quit
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional

from pipeline.detection_pipeline import FrameData
from ml.detection import Detection
from ml.thermal_isolation import ThermalIsolator
from display.osd import OSDRenderer

logger = logging.getLogger("display")

BG_DARK   = (15,  12,  10)
BG_PANEL  = (23,  17,  13)
ORANGE    = (0,   140, 255)
BLUE      = (220, 140, 50)
GREEN     = (80,  220, 60)
RED       = (30,  30,  210)
CYAN      = (200, 220, 50)
WHITE     = (235, 235, 235)
GRAY      = (120, 120, 120)
GRAY_DIM  = (65,  65,  65)
BORDER    = (40,  35,  30)
CHIP_BG   = (35,  28,  22)
FONT      = cv2.FONT_HERSHEY_DUPLEX

THERMAL_MODES = ["highlight", "silhouette", "contour"]

def _txt(img, text, pos, scale=0.48, color=WHITE, thickness=1, shadow=True):
    if shadow:
        cv2.putText(img, text, (pos[0]+1, pos[1]+1),
                    FONT, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)

def _chip(img, text, x, y, color, w=None):
    tw, th = cv2.getTextSize(text, FONT, 0.38, 1)[0]
    pw = w if w else tw + 14; ph = th + 8
    cv2.rectangle(img, (x, y), (x+pw, y+ph), CHIP_BG, -1)
    cv2.rectangle(img, (x, y), (x+pw, y+ph), BORDER, 1)
    cv2.putText(img, text, (x+(pw-tw)//2, y+ph-4), FONT, 0.38, color, 1, cv2.LINE_AA)
    return pw + 6

def _corner_box(img, x1, y1, x2, y2, col, t=2):
    cv2.rectangle(img, (x1,y1), (x2,y2), col, 1)
    L = max(10, min(20, (x2-x1)//5))
    for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img,(px,py),(px+dx*L,py),col,t,cv2.LINE_AA)
        cv2.line(img,(px,py),(px,py+dy*L),col,t,cv2.LINE_AA)


class RescueDisplay:
    def __init__(self, cfg):
        self.cfg          = cfg
        self.W            = cfg.display_width
        self.H            = cfg.display_height
        self._WIN         = "Rescue Drone  |  Human Detection"
        self._writer: Optional[cv2.VideoWriter] = None
        self._quit        = False
        self._mode_idx    = 0
        self._show_thermal = True
        self._main_view_idx = 0  # 0: RGB, 1: Thermal, 2: Radar

        self._isolator = ThermalIsolator(cfg)
        
        self._last_snapshot_time = {} # track_id -> timestamp
        self._osd      = OSDRenderer(cfg)

        cv2.namedWindow(self._WIN, cv2.WINDOW_NORMAL)
        if getattr(cfg, 'fullscreen', True):
            cv2.setWindowProperty(self._WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(self._WIN, self.W, self.H)
        if cfg.record:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(cfg.output_video, fourcc, 8, (self.W, self.H))

    def render(self, fd: FrameData):
        import time
        canvas = self._build(fd)
        if self._writer: self._writer.write(canvas)
        cv2.imshow(self._WIN, canvas)
        
        # --- AUTO SNAPSHOT (Mirrors Manual Snap Perfectly) ---
        if self.cfg.snapshot_enabled and getattr(fd, "tracked_humans", None):
            now = time.time()
            for t in fd.tracked_humans:
                last = self._last_snapshot_time.get(t.track_id, 0)
                if now - last >= self.cfg.snapshot_cooldown:
                    p = f"{self.cfg.snapshot_dir}/auto_snap_TRK{t.track_id}_{fd.frame_id:05d}.jpg"
                    cv2.imwrite(p, canvas)
                    logger.info(f"Auto-Snapshot Saved: {p}")
                    self._last_snapshot_time[t.track_id] = now
        # -----------------------------------------------------
        
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self._quit = True
        elif key == ord("s"):
            p = f"output/snap_{fd.frame_id:05d}.jpg"
            cv2.imwrite(p, canvas)
            logger.info(f"Snapshot: {p}")
        elif key == ord("m"):
            self._mode_idx = (self._mode_idx + 1) % len(THERMAL_MODES)
            logger.info(f"Thermal mode: {THERMAL_MODES[self._mode_idx]}")
        elif key == ord("t"):
            self._show_thermal = not self._show_thermal
        elif key == ord("v"):
            self._main_view_idx = (self._main_view_idx + 1) % 3

    def should_quit(self): return self._quit

    def close(self):
        if self._writer: self._writer.release()
        cv2.destroyAllWindows()

    def _build(self, fd: FrameData) -> np.ndarray:
        if self._main_view_idx == 0:
            base_layers = self._get_rgb_base(fd, self.W, self.H)
        elif self._main_view_idx == 1:
            base_layers = self._thermal_pane(fd, self.W, self.H, clean=True)
        else:
            base_layers = self._radar_pane(fd, self.W, self.H)

        sx, sy = self.W/self.W, self.H/self.H # 1.0
        scaled = [
            Detection(
                x1=d.x1*sx, y1=d.y1*sy, x2=d.x2*sx, y2=d.y2*sy,
                confidence=d.confidence, source=d.source,
                track_id=d.track_id, temp_celsius=d.temp_celsius,
            )
            for d in fd.tracked_humans
        ]

        rd = fd.radar
        fc_telemetry = fd.fc_telemetry

        # Render Persistent OSD over ANY view
        canvas = self._osd.render(
            frame          = base_layers,
            dets           = scaled,
            num_humans     = fd.num_humans,
            total_unique   = getattr(fd, "total_unique_humans", 0),
            radar_present  = rd is not None and rd.human_present,
            radar_state    = rd.target_state    if rd else 0,
            radar_dist_cm  = rd.detect_dist_cm  if rd else 0,
            radar_strength = max(rd.moving_energy, rd.static_energy) if rd else 0,
            fps            = fd.fps,
            frame_id       = fd.frame_id,
            recording      = self.cfg.record,
            sensor_ok      = fd.sensor_status,
            yolo_backend   = fd.yolo_backend,
            total_ms       = fd.timing_ms.get("total_ms", 0),
            fc_telemetry   = fc_telemetry,
            anomaly_triggered = getattr(fd, "anomaly_triggered", False),
        )

        # Overlay Thermal PiP if NOT Main View and PiP is enabled
        if self._show_thermal and self._main_view_idx != 1:
            pip_w = 320
            pip_h = 240
            padding = 20
            pip_x = padding
            pip_y = self.H - pip_h - padding
            
            thermal_pip = self._thermal_pane(fd, pip_w, pip_h, clean=False)
            cv2.rectangle(canvas, (pip_x-2, pip_y-2), (pip_x+pip_w+1, pip_y+pip_h+1), BORDER, 2)
            canvas[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = thermal_pip

        return canvas

    # ── Thermal pane ──────────────────────────────────────────────

    def _thermal_pane(self, fd: FrameData, pw: int, ph: int, clean: bool = False) -> np.ndarray:
        mode = THERMAL_MODES[self._mode_idx]

        if fd.thermal_visual is not None and fd.thermal_raw is not None:
            iso, _, _ = self._isolator.process(fd.thermal_raw, fd.thermal_visual, mode)
            pane = cv2.resize(iso, (pw, ph), interpolation=cv2.INTER_LINEAR)
            sw, sh = fd.thermal_visual.shape[1], fd.thermal_visual.shape[0]
        elif fd.thermal_visual is not None:
            pane = cv2.resize(fd.thermal_visual, (pw, ph))
            sw, sh = fd.thermal_visual.shape[1], fd.thermal_visual.shape[0]
        else:
            pane = np.full((ph, pw, 3), BG_DARK, dtype=np.uint8)
            _txt(pane, "THERMAL OFFLINE", (pw//2-60, ph//2), 0.5, GRAY_DIM)
            return pane

        if not clean:
            self._temp_scale(pane, pw, ph)
            
            # Show the dynamic span on PiP
            _txt(pane, f"THERMAL: {mode.upper()}", (5, ph - 10), 0.35, GRAY_DIM, shadow=True)

        return pane

    def _thermal_box(self, pane, det, tw, th, sw, sh):
        sx, sy = tw/max(sw,1), th/max(sh,1)
        x1,y1,x2,y2 = int(det.x1*sx),int(det.y1*sy),int(det.x2*sx),int(det.y2*sy)
        _corner_box(pane, x1, y1, x2, y2, ORANGE, t=1)

    def _temp_scale(self, pane, pw, ph):
        bx,by,bws,bhs = pw-15, 10, 5, ph-20
        for i in range(bhs):
            t = 1.0 - i/bhs
            if t<0.30: s=t/0.30;b=int(80+s*100);g=int(10+s*50);r=int(5+s*25)
            elif t<0.55: s=(t-0.30)/0.25;b=int(180-s*20);g=int(60+s*120);r=20
            elif t<0.75: s=(t-0.55)/0.20;b=int(200-s*180);g=180;r=int(20+s*180)
            elif t<0.90: s=(t-0.75)/0.15;b=20;g=int(180-s*80);r=int(200+s*55)
            else: s=(t-0.90)/0.10;b=int(s*220);g=int(100+s*155);r=255
            cv2.line(pane,(bx,by+i),(bx+bws,by+i),(b,g,r),1)
        cv2.rectangle(pane,(bx,by),(bx+bws,by+bhs),BORDER,1)

        # Dynamic Range T_min / T_max drawing
        t_min = f"{self.cfg.thermal_min_temp:.1f}C"
        t_max = f"{self.cfg.thermal_max_temp:.1f}C"
        _txt(pane, t_max, (bx - 35, by + 10), 0.3, WHITE)
        _txt(pane, t_min, (bx - 35, by + bhs), 0.3, WHITE)


    # ── RGB pane ──────────────────────────────────────────────────

    def _get_rgb_base(self, fd: FrameData, pw: int, ph: int) -> np.ndarray:
        if fd.rgb_frame is not None:
            self._last_rgb_frame = cv2.resize(fd.rgb_frame, (pw, ph))

        if hasattr(self, "_last_rgb_frame") and self._last_rgb_frame is not None:
            base = self._last_rgb_frame
        else:
            base = np.full((ph, pw, 3), BG_DARK, dtype=np.uint8)
            _txt(base, "RGB CAM WARMING UP...", (pw // 2 - 110, ph // 2),
                 0.55, GRAY_DIM, shadow=False)
        
        return base

    # ── Radar pane ────────────────────────────────────────────────

    def _radar_pane(self, fd: FrameData, pw: int, ph: int) -> np.ndarray:
        base = np.full((ph, pw, 3), (10, 20, 15), dtype=np.uint8) # Dark greenish
        cx, cy = pw // 2, ph // 2
        
        # Concentric Rings
        for r in range(100, min(pw, ph) // 2 + 100, 100):
            cv2.circle(base, (cx, cy), r, (0, 80, 0), 1, cv2.LINE_AA)
            
        cv2.line(base, (cx, 0), (cx, ph), (0, 80, 0), 1)
        cv2.line(base, (0, cy), (pw, cy), (0, 80, 0), 1)

        # Sweeping Line
        now = time.time()
        angle = (now % 2.0) / 2.0 * 2 * np.pi
        length = min(pw, ph) // 2
        ex = int(cx + length * np.cos(angle))
        ey = int(cy + length * np.sin(angle))
        cv2.line(base, (cx, cy), (ex, ey), (0, 255, 0), 2, cv2.LINE_AA)

        # Show target
        rd = fd.radar
        if rd and rd.human_present:
            # Distance mapping
            dist_px = (rd.detect_dist_cm / max(1, self.cfg.ld2410_max_range_cm)) * length
            if rd.target_state == 1:   # Moving
                t_angle = angle - 0.2
                tx = int(cx + dist_px * np.cos(t_angle))
                ty = int(cy + dist_px * np.sin(t_angle))
                cv2.circle(base, (tx, ty), 10, (0, 0, 255), -1)
                _txt(base, f"MOVING {rd.detect_dist_cm}cm", (tx+15, ty), 0.5, WHITE)
            elif rd.target_state >= 2: # Stationary
                t_angle = angle + 0.5
                tx = int(cx + dist_px * np.cos(t_angle))
                ty = int(cy + dist_px * np.sin(t_angle))
                cv2.rectangle(base, (tx-8, ty-8), (tx+8, ty+8), (0, 255, 255), -1)
                _txt(base, f"STAT {rd.detect_dist_cm}cm", (tx+15, ty), 0.5, WHITE)
                
        return base
