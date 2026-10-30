"""
Quantize the YOLO ONNX model to int8 (static, per-channel weights), calibrated
on real images preprocessed exactly as on the drone (ml.models.YOLODetector.blob).
Re-run after every retrain: the drone ignores an int8 file older than the float
model (config.py yolo_int8).

Default recipe = the fastest measured on the Pi 4 (coco128, 64 calibration /
64 held-out images, 1 thread): only the convolutions quantized (QOperator
format), detect head kept float — 249ms vs 292ms float, same accuracy.
Quantizing everything in QDQ format was SLOWER (350-380ms: the Pi 4's
Cortex-A72 has no int8 dot-product instructions, and each of the 57 SiLU
activations adds quantize/dequantize steps), and quantizing the head too
destroyed detection (recall 0).

Needs the `onnx` package, which the runtime env (~/venv) doesn't have — use a
separate env that can also see the system OpenCV / numpy, e.g.:
    ~/.local/bin/uv venv --system-site-packages ~/quant_venv
    ~/.local/bin/uv pip install --python ~/quant_venv/bin/python onnx onnxruntime==<same as ~/venv>
    ~/quant_venv/bin/python tools/quantize_yolo.py --images DIR [--out models/rgb_human_int8.onnx]

Then compare against the float model with tools/eval_yolo.py (accuracy + speed)
before switching config.py yolo_model_path to the int8 file.
"""
import argparse, glob, os, sys, tempfile
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import logging
logging.disable(logging.WARNING)
import cv2
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from config import Config
from ml.models import YOLODetector

# YOLOv8/YOLO11 head: the DFL box decoding after the last convolutions. It turns
# distributions into pixel coordinates, so int8 rounding there moves boxes.
_POST = ("/dfl/", "/Slice", "/Sub", "/Add_1", "/Add_2", "/Div_1", "/Concat_2", "/Mul_2", "/Concat_3")


class _Images(CalibrationDataReader):
    def __init__(self, blobs, input_name):
        self._it = iter([{input_name: b} for b in blobs])

    def get_next(self):
        return next(self._it, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="calibration images (representative of the camera feed)")
    ap.add_argument("--model", default=None, help="default: config.py yolo_model_path")
    ap.add_argument("--out", default=None, help="default: <model>_int8.onnx")
    ap.add_argument("--n", type=int, default=64, help="number of calibration images")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--method", choices=["minmax", "percentile", "entropy"], default="minmax")
    ap.add_argument("--keep-float", choices=["none", "post", "head"], default="head",
                    help="leave unquantized: nothing, the box decoding, or the whole detect head (default)")
    ap.add_argument("--format", choices=["qoperator", "qdq"], default="qoperator")
    ap.add_argument("--ops", choices=["conv", "all"], default="conv", help="op types to quantize")
    a = ap.parse_args()

    cfg = Config()
    src = a.model or cfg.yolo_model_path
    out = a.out or os.path.splitext(src)[0] + "_int8.onnx"
    det = YOLODetector(cfg)
    model = onnx.load(src)
    input_name = model.graph.input[0].name
    h, w = (d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim[2:4])
    if h and w:
        det.set_input_size(w, h)  # calibrate at the model's own input shape

    paths = sorted(p for p in glob.glob(os.path.join(a.images, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
    paths = paths[a.offset::a.every][:a.n]
    blobs = [det.blob(img) for img in (cv2.imread(p) for p in paths) if img is not None]
    if not blobs:
        sys.exit("no calibration images found")

    # Detect head = the last /model.N/ block (22 in YOLOv8, 23 in YOLO11).
    idx = max(int(n.name.split("/model.")[1].split("/")[0]) for n in model.graph.node if "/model." in n.name)
    head = [n.name for n in model.graph.node if f"/model.{idx}/" in n.name]
    exclude = {"none": [], "head": head,
               "post": [n for n in head if any(k in n for k in _POST)]}[a.keep_float]

    with tempfile.TemporaryDirectory() as tmp:
        pre = os.path.join(tmp, "pre.onnx")
        quant_pre_process(src, pre, skip_symbolic_shape=True)
        quantize_static(
            pre, out, _Images(blobs, input_name),
            quant_format=QuantFormat.QOperator if a.format == "qoperator" else QuantFormat.QDQ,
            op_types_to_quantize=["Conv"] if a.ops == "conv" else None,
            per_channel=True,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            calibrate_method={"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
                              "entropy": CalibrationMethod.Entropy}[a.method],
            nodes_to_exclude=exclude,
        )
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB) | {len(blobs)} calibration images, "
          f"{a.method}, {a.format}, ops: {a.ops}, float kept: {a.keep_float} ({len(exclude)} nodes)")


if __name__ == "__main__":
    main()
