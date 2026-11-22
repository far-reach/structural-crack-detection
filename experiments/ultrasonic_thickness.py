"""Thickness from single pulse-echo A-scans, calibrated on one specimen and
tested on the other two.

Calibration (wave speed c and system delay t0) uses Pk050 only. Step 2 of
Pk050 is left out of the calibration: its tabulated thickness (425.0 mm) is
inconsistent with the arrival times of both wave types, see the
`pk050_step2_check` entry of the output.

Usage: python experiments/ultrasonic_thickness.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspectswarm import ultrasonic as U
from inspectswarm.manifest import save_result
from inspectswarm.paths import RESULTS

GATE_US = {"Long": 85.0, "Shear": 110.0}
MAX_US = {"Long": 700.0, "Shear": 900.0}
EDGE_MM = 60.0


def errors(d_est, d_true, sel):
    e = np.abs(d_est - d_true)[sel]
    return {"n": int(sel.sum()), "mae_mm": float(e.mean()), "median_ae_mm": float(np.median(e)),
            "within_10mm": float((e < 10).mean()), "within_5pct": float((e / d_true[sel] < 0.05).mean())}


def neighbour_median(t_pick, k):
    """Median of k picks at adjacent grid points along Y (10 mm apart)."""
    if k == 1:
        return t_pick
    h = k // 2
    stack = [np.roll(t_pick, s, axis=1) for s in range(-h, h + 1)]
    return np.median(np.stack(stack), axis=0)


if __name__ == "__main__":
    out = {}
    for wave in ("Shear", "Long"):
        for rot in ("00", "90"):
            picks, valid = {}, {}
            for s in ("pk050", "pk266", "pk401"):
                v = U.load_volume(s, wave, rot)
                valid[s] = U.valid_mask(v)
                picks[s] = U.pick_backwall(U.envelope(U.bandpass(v, wave)), GATE_US[wave], MAX_US[wave])
            th0, clear0 = U.thickness_map("pk050", EDGE_MM)
            cal = valid["pk050"] & clear0 & (th0 != U.THICKNESS["pk050"][1])
            c, t0 = U.fit_velocity(picks["pk050"][cal], th0[cal])
            step2 = valid["pk050"] & clear0 & (th0 == U.THICKNESS["pk050"][1])
            row = {"c_ms": c, "t0_us": t0,
                   "pk050_step2_check": {"tabulated_mm": U.THICKNESS["pk050"][1],
                                         "estimated_median_mm": float(np.median(U.thickness_from_time(picks["pk050"][step2], c, t0))),
                                         "median_arrival_us": float(np.median(picks["pk050"][step2]))},
                   "pk050_calibration_fit": errors(U.thickness_from_time(picks["pk050"], c, t0), th0, cal)}
            for s in ("pk266", "pk401"):
                th, clear = U.thickness_map(s, EDGE_MM)
                lab, _ = U.object_labels(s)
                plain = valid[s] & clear & (lab == 0)
                over = valid[s] & clear & (lab == 1)
                r = {}
                for k in (1, 3, 5):
                    d = U.thickness_from_time(neighbour_median(picks[s], k), c, t0)
                    r[f"plain_k{k}"] = errors(d, th, plain)
                d = U.thickness_from_time(picks[s], c, t0)
                r["over_embedded_object_k1"] = errors(d, th, over)
                r["per_step_plain_k1"] = {str(t_): errors(d, th, plain & (th == t_)) for t_ in U.THICKNESS[s]}
                # specimen-specific wave speed, to separate picking error from calibration transfer
                c_s, t0_s = U.fit_velocity(picks[s][plain], th[plain])
                r["own_calibration"] = {"c_ms": c_s, "t0_us": t0_s,
                                        **errors(U.thickness_from_time(picks[s], c_s, t0_s), th, plain)}
                row[s] = r
            out[f"{wave}_Rot{rot}"] = row
            print(wave, rot, f"c={c:.0f} t0={t0:.1f}", "pk050 step2 est", round(row["pk050_step2_check"]["estimated_median_mm"], 1),
                  {s: {k: round(v, 3) for k, v in row[s]["plain_k1"].items()} for s in ("pk266", "pk401")})
    save_result(RESULTS / "ultrasonic" / "thickness.json", out, {"gate_us": GATE_US, "max_us": MAX_US, "edge_mm": EDGE_MM})
