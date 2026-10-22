# Handoff: retrain the person detector for aerial search-and-rescue

For a Claude Code session on the owner's laptop. The drone code runs on a Raspberry Pi 4.
The Pi can't train models, so training happens on **Google Colab** (free T4 GPU) from
scripts you write here. Written 2026-10-22 by the Claude session working on the Pi.

## Goal
Produce a person detector that works **from the air** and drops into the drone unchanged:
1. Write a Colab notebook plus small helper scripts that fetch and convert aerial person
   datasets, fine-tune YOLOv8n, evaluate it and export ONNX.
2. Hand back one file, `rgb_human.onnx`, that meets the contract below, together with
   the evaluation numbers.

## Why
The current model (`models/rgb_human.onnx`) was trained on ground-level views:
- **~0% recall on VisDrone** aerial images.
- False positives on dark frames.
- On COCO people (coco128, held-out half, conf 0.35): recall 0.237, precision 0.844.

## Model contract — must match `ml/models.py` `YOLODetector` exactly
| Item | Required |
|---|---|
| Architecture | Ultralytics **YOLOv8n**. Inference costs ~290 ms per frame on 1 Pi core; bigger models won't fit. |
| Classes | **1 class, `person` = class 0.** The code reads output row 4 as the person score. |
| ONNX input | One input (any name), **static `[1, 3, 320, 320]`**, float32, **RGB**, 0–1 (pixels / 255). |
| Preprocessing on the Pi | Ultralytics-style **letterbox**: keep aspect, centre pad with value 114. Train with the standard Ultralytics dataloader, which does the same. |
| ONNX output | Raw YOLOv8 head **`[1, 5, 2100]`** (cx, cy, w, h, score) in input-pixel units. **No NMS in the graph**: the Pi does its own NMS. |
| Export | `model.export(format="onnx", imgsz=320, opset=20, dynamic=False, simplify=True, half=False, nms=False)`. The Pi runs onnxruntime **1.30**, and the current model is opset 20. |
| Runtime settings | `yolo_conf_threshold=0.35`, `yolo_iou_threshold=0.45` (config.py). Report results at conf 0.35. |

Input size: aerial people are tiny, and 416 px would help. Only change it if the owner agrees.
It costs ~1.7x the Pi time (~500 ms per frame). That still fits, because YOLO runs alongside
the 520 ms thermal-camera read, but it leaves no headroom. If you do it, also set
`yolo_input_size` in `config.py`. **Train at the same size you export.**

## Datasets (convert all to YOLO format, single class `person` = 0)
- **VisDrone2019-DET** (aerial).
  - Ultralytics' `VisDrone.yaml` downloads and converts it.
  - Map `pedestrian` (0) and `people` (1) to `person`; drop all other classes.
  - Use its **val split (548 images)** as the main aerial test set.
- **SARD** and **HERIDAL**: aerial search-and-rescue person datasets (wilderness, people
  lying or partly hidden).
  - Find the current official or Kaggle/Roboflow sources and check the licences before use.
  - Keep a held-out split of each.
- **Negatives:** images with no people, especially **dark / low-light** frames, to cut the
  false positives. Keep about 10% of training images without labels.
- Optional: a small share (≤ 20%) of COCO `person` images, so ground-level views (people
  near the drone, at take-off) still work.
- Never commit datasets to the repo.

## Suggested recipe (adjust if results say otherwise)
- Start from `yolov8n.pt` (COCO-pretrained, person is already a class). Ask the owner whether
  they still have the `.pt` that `rgb_human.onnx` was exported from; fine-tuning that is
  another option.
- `imgsz=320` (to match the export). Train for about 100 epochs with early stopping and the
  default augmentations, plus mosaic. Batch size to fit the T4.
- Save checkpoints to Google Drive: Colab sessions disconnect.

## Deliverables (in the repo on the laptop, under `training/`)
- `training/train_aerial_person.ipynb`: Colab notebook. Mounts Drive, installs
  `ultralytics`, fetches and converts the data (calling the scripts below), trains,
  evaluates, exports ONNX, and verifies the export (shape `[1,5,2100]`, opset, a test
  inference with onnxruntime).
- `training/prepare_datasets.py`: download, convert and remap to single-class YOLO format.
- `training/check_onnx.py`: asserts the contract above on an exported file.
- A short `training/README.md`: how to run it, and the results table below.

## Acceptance — report these numbers
Run the repo's own evaluator, which uses the same preprocessing, decoding and NMS as the drone,
on the **held-out** sets, for both the old and the new model:
```
python tools/eval_yolo.py --images <val>/images --labels <val>/labels --model models/rgb_human.onnx
python tools/eval_yolo.py --images <val>/images --labels <val>/labels --model <new>.onnx
```
It needs `numpy opencv-python onnxruntime` and the repo's `config.py` and `ml/` (it imports them).
- **Target:** much higher recall on VisDrone/SARD/HERIDAL val people than the old ~0%, at
  conf 0.35.
- Precision must not collapse.
- The coco128 people result should not drop much below the old model's.

## Getting the model to the Pi
- Copy the file to `pi@100.89.36.99` (Tailscale) or `pi@192.168.1.19` (LAN):
  ```
  scp rgb_human.onnx pi@100.89.36.99:/home/pi/SkyNetics-RAS-drone/models/rgb_human.onnx
  ```
  Back up the old model there first (`rgb_human_fp32_old.onnx`).
- The Pi's Wi-Fi is weak (about 50% packet loss was measured), so retry the copy if it stalls.
- On the Pi, the Claude session there will re-run `tools/quantize_yolo.py` (int8 copy; the
  drone ignores an int8 file older than the float one), `tools/eval_yolo.py` and the live
  benchmarks.

## Notes
- The GitHub repo (`asronal/SkyNetics-RAS-drone`) may not have the Pi's latest changes yet.
  If `tools/eval_yolo.py` is missing, ask the owner to copy it, or `ml/` and `config.py`, from the Pi.
- Don't change the output format, class index, input size or preprocessing without updating
  `ml/models.py` and `config.py` to match. The Pi-side code depends on all of them.
