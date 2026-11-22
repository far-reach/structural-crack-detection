"""Ultrasonic tier: low-frequency pulse-echo on concrete (BAM open datasets).

Data: Maack et al. 2023 (Pk050, Pk266) and Grohmann et al. 2025 (Pk401),
Harvard Dataverse, CC0. Each volume is (201 X, 81 Y, 4000 samples) at 2 MHz
on a 10 mm grid; grid index times 10 mm is the specimen coordinate.

Ground truth below is taken from the tables and drawings of the two data
articles. Nothing in this module is simulated.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .paths import BAM

FS = 2.0e6          # sampling rate, Hz
GRID_MM = 10.0
VALID_X = (7, 193)  # inclusive index range that holds measurements
VALID_Y = (5, 75)
STEP_EDGES_MM = (500.0, 1000.0, 1500.0)
BAND = {"Long": (25e3, 180e3), "Shear": (15e3, 110e3)}

# Step thickness in mm for X in [0,500), [500,1000), [1000,1500), [1500,2000]
THICKNESS = {
    "pk050": (571.3, 425.0, 330.9, 210.8),
    "pk266": (569.9, 448.2, 329.8, 210.3),
    "pk401": (573.8, 453.4, 333.1, 210.4),
}


@dataclass(frozen=True)
class Target:
    kind: str       # "duct" (runs along Y), "cuboid" or "borehole"
    x: float        # centre, mm
    y: float        # centre, mm (unused for ducts)
    cover: float    # concrete cover to the facing surface, mm
    half_x: float   # half extent in X, mm
    half_y: float   # half extent in Y, mm


TARGETS = {
    "pk050": (),
    "pk266": tuple(Target("duct", x, 400.0, c, 33.5, 1e9)
                   for x, c in ((250, 244.5), (750, 184.6), (1250, 121.4), (1750, 64.4))),
    "pk401": tuple(Target("cuboid", x, y, c, 60.0, 60.0)
                   for x, y, c in ((250, 398.2, 270.0), (750, 398.2, 210.0),
                                   (1250, 498.2, 150.0), (1750, 398.2, 90.0)))
    + tuple(Target("borehole", x, y, 0.0, 7.0, 7.0)
            for x, y in ((368, 359), (506, 365), (647, 367), (827, 366))),
}

_FILES = {"pk050": "Pk050_3D_Dataset_{w}_Rot{r}.npy", "pk266": "Pk266_3D_Dataset_{w}_Rot{r}.npy",
          "pk401": "Pk_401_3D_Dataset_{w}_Wave_Rot{r}.npy"}


def load_volume(specimen: str, wave: str = "Shear", rot: str = "00") -> np.ndarray:
    """(201, 81, 4000) float32 volume, DC removed per A-scan."""
    v = np.load(BAM / specimen / _FILES[specimen].format(w=wave, r=rot), mmap_mode="r")
    v = np.asarray(v, dtype=np.float32)
    return v - v.mean(axis=-1, keepdims=True)


def coords():
    x = np.arange(201) * GRID_MM
    y = np.arange(81) * GRID_MM
    return np.meshgrid(x, y, indexing="ij")


def valid_mask(volume=None) -> np.ndarray:
    m = np.zeros((201, 81), bool)
    m[VALID_X[0]:VALID_X[1] + 1, VALID_Y[0]:VALID_Y[1] + 1] = True
    if volume is not None:
        m &= np.abs(volume).max(-1) > 0
    return m


def thickness_map(specimen: str, edge_margin_mm: float = 0.0):
    """True thickness per grid point, and a mask of points at least
    `edge_margin_mm` away from a step edge (where the back wall is unambiguous)."""
    X, _ = coords()
    step = np.digitize(X, STEP_EDGES_MM)
    t = np.asarray(THICKNESS[specimen])[step]
    clear = np.ones_like(X, bool)
    for e in STEP_EDGES_MM:
        clear &= np.abs(X - e) >= edge_margin_mm
    return t, clear


def object_labels(specimen: str, margin_mm: float = 100.0):
    """Per grid point: 1 above an embedded duct or cuboid, 0 over plain
    concrete, -1 (ignore) within `margin_mm` of an object edge or of a
    grouted borehole. Also returns the cover depth of the object (nan if none)."""
    X, Y = coords()
    lab = np.zeros(X.shape, int)
    cover = np.full(X.shape, np.nan)
    for t in TARGETS[specimen]:
        dx, dy = np.abs(X - t.x), np.abs(Y - t.y)
        inside = (dx <= t.half_x) & (dy <= t.half_y)
        near = (dx <= t.half_x + margin_mm) & (dy <= t.half_y + margin_mm)
        if t.kind == "borehole":
            lab[near & (lab == 0)] = -1
            continue
        lab[near & ~inside & (lab != 1)] = -1
        lab[inside] = 1
        cover[inside] = t.cover
    return lab, cover


def bandpass(volume: np.ndarray, wave: str) -> np.ndarray:
    lo, hi = BAND[wave]
    sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, volume, axis=-1).astype(np.float32)


def envelope(filtered: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(filtered, axis=-1)).astype(np.float32)


# ---------------------------------------------------------------- thickness

def pick_backwall(env: np.ndarray, gate_us: float, max_us: float = 700.0, rel: float = 0.6) -> np.ndarray:
    """Arrival time (microseconds) of the back-wall echo.

    The envelope after the direct-wave gate is weighted by time to offset
    geometric spreading. The pick is the first local maximum that reaches
    `rel` of the weighted maximum, so that a first echo is preferred over
    its own multiple."""
    i0, i1 = int(gate_us * 1e-6 * FS), int(max_us * 1e-6 * FS)
    t = np.arange(i0, i1) / FS * 1e6
    seg = env[..., i0:i1] * t
    is_peak = np.zeros(seg.shape, bool)
    is_peak[..., 1:-1] = (seg[..., 1:-1] >= seg[..., :-2]) & (seg[..., 1:-1] >= seg[..., 2:])
    strong = is_peak & (seg >= rel * seg.max(-1, keepdims=True))
    first = strong.argmax(-1)
    first = np.where(strong.any(-1), first, seg.argmax(-1))
    return t[first]


def fit_velocity(t_us: np.ndarray, thickness_mm: np.ndarray):
    """Fit t = t0 + 2 d / c through the median arrival time of each distinct
    thickness (robust to wrong picks). Returns c (m/s) and t0 (microseconds)."""
    d = np.unique(thickness_mm)
    tm = np.array([np.median(t_us[thickness_mm == v]) for v in d])
    A = np.stack([2 * d, np.ones_like(d)], 1)
    (slope, t0), *_ = np.linalg.lstsq(A, tm, rcond=None)
    return 1e3 / slope, float(t0)


def thickness_from_time(t_us, c_ms: float, t0_us: float):
    return (np.asarray(t_us) - t0_us) * c_ms / 2e3


# ------------------------------------------------------- learned detection

def traces_for_model(filtered: np.ndarray, gate_us: float, n_samples: int = 512, decim: int = 4) -> np.ndarray:
    """Model input, shape (..., 2, n_samples): the band-passed A-scan after the
    direct-wave gate, multiplied by time (to offset geometric spreading),
    decimated to 500 kHz and scaled by its 99th-percentile amplitude; and the
    envelope of the same signal. The direct wave is excluded, so the scale is
    set by echoes from inside the member."""
    i0 = int(gate_us * 1e-6 * FS)
    t = np.arange(filtered.shape[-1], dtype=np.float32) / FS * 1e6
    gained = filtered * t
    env = np.abs(hilbert(gained, axis=-1)).astype(np.float32)
    sl = slice(i0, i0 + n_samples * decim, decim)
    x, e = gained[..., sl], env[..., sl]
    scale = np.maximum(np.percentile(np.abs(x), 99, axis=-1, keepdims=True), 1e-9)
    return np.stack([x / scale, e / scale], axis=-2).astype(np.float32)


def build_ascan_net(width: int = 16):
    """1D CNN for single A-scans (two input channels), 35,233 parameters at width 16."""
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv1d(i, o, 9, 2, 4, bias=False), nn.BatchNorm1d(o), nn.ReLU(inplace=True))

    w = width
    return nn.Sequential(block(2, w), block(w, w), block(w, 2 * w), block(2 * w, 2 * w), block(2 * w, 4 * w),
                         nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(4 * w, 1))


def classical_score(env: np.ndarray, thickness_mm: np.ndarray, c_ms: float, t0_us: float,
                    gate_us: float, guard_us: float = 35.0) -> np.ndarray:
    """Physics baseline for "something reflects before the back wall": the
    largest time-weighted envelope between the direct-wave gate and the
    expected back-wall arrival, divided by the back-wall amplitude. It is
    given the true thickness, so it is an informed (optimistic) baseline."""
    t = np.arange(env.shape[-1]) / FS * 1e6
    t_bw = t0_us + 2e3 * thickness_mm / c_ms
    w = env * t
    before = (t[None, None, :] >= gate_us) & (t[None, None, :] <= (t_bw - guard_us)[..., None])
    at = np.abs(t[None, None, :] - t_bw[..., None]) <= guard_us / 2
    num = np.where(before, w, 0).max(-1)
    den = np.where(at, w, 0).max(-1)
    return num / np.maximum(den, 1e-9)
