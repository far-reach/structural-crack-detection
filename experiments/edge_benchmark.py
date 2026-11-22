"""Edge budget of each vision model: size, CPU latency, and accuracy after
8-bit quantisation (ONNX Runtime, static quantisation calibrated on the
validation split). Latency is measured on the host CPU, one thread, batch 1;
it is a relative measure between models, not a microcontroller measurement.

Usage: python experiments/edge_benchmark.py --tags convnext_tiny_r224_s0 nano32_r128_kd-..._s0
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspectswarm import vision
from inspectswarm.classes import CODEBRIM_DEFECTS
from inspectswarm.manifest import save_result
from inspectswarm.paths import CACHE, RESULTS, WEIGHTS


def to_input(x_uint8, res):
    return vision.normalise(torch.as_tensor(x_uint8), res).numpy()


class Reader(CalibrationDataReader):
    def __init__(self, x):
        self.it = iter([{"input": x[i:i + 1]} for i in range(len(x))])

    def get_next(self):
        return next(self.it, None)


def session(path, threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run(sess, x, batch=32):
    out = [sess.run(None, {"input": x[i:i + batch]})[0] for i in range(0, len(x), batch)]
    return 1 / (1 + np.exp(-np.concatenate(out)))


def latency_ms(sess, x1, n=60):
    for _ in range(5):
        sess.run(None, {"input": x1})
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {"input": x1})
        t.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(t)), float(np.percentile(t, 90))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    val, test = np.load(CACHE / "codebrim_val.npz"), np.load(CACHE / "codebrim_test.npz")
    out_dir = CACHE / "onnx"
    out_dir.mkdir(exist_ok=True)
    rows = {}
    for tag in a.tags:
        name, res = tag.split("_r")[0], int(tag.split("_r")[1].split("_")[0])
        model = vision.build_model(name, len(CODEBRIM_DEFECTS), pretrained=False)
        model.load_state_dict(torch.load(WEIGHTS / f"codebrim_{tag}.pt", map_location="cpu"))
        model.eval()
        f32, i8 = out_dir / f"{tag}_fp32.onnx", out_dir / f"{tag}_int8.onnx"
        torch.onnx.export(model, torch.zeros(1, 3, res, res), str(f32), input_names=["input"],
                          output_names=["logits"], dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}},
                          opset_version=17, dynamo=False)
        xv, xt, yt = to_input(val["x"], res), to_input(test["x"], res), test["y"].astype(int)
        rng = np.random.default_rng(0)
        quantize_static(str(f32), str(i8), Reader(xv[rng.choice(len(xv), 200, replace=False)]),
                        quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8, per_channel=True)
        row = {"params": vision.count_params(model), "res": res}
        for kind, path in (("fp32", f32), ("int8", i8)):
            s = session(path)
            m = vision.multilabel_metrics(yt, run(s, xt), CODEBRIM_DEFECTS)
            med, p90 = latency_ms(s, xt[:1])
            row[kind] = {"size_kb": path.stat().st_size / 1024, "latency_ms_median": med, "latency_ms_p90": p90,
                         "multi_target_acc": m["multi_target_acc"], "macro_f1": m["macro_f1"],
                         "damage_recall": m["damage_recall"], "false_alarm_rate": m["false_alarm_rate"]}
        # multiply-accumulate count, a hardware-independent cost measure
        try:
            from fvcore.nn import FlopCountAnalysis  # optional

            row["flops"] = int(FlopCountAnalysis(model, torch.zeros(1, 3, res, res)).total())
        except Exception:
            pass
        rows[tag] = row
        print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row["fp32"].items()},
              "| int8", {k: round(v, 3) for k, v in row["int8"].items()})
    save_result(RESULTS / "vision" / "edge_benchmark.json", rows, vars(a))
