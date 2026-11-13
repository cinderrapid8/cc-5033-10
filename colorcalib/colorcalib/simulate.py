"""Deterministic synthetic camera / illumination rendering.

The kit builder and the verification script both use this module, so a rebuild
from the saved parameters reproduces byte-identical inputs.  Every physical
effect is an *explicit*, listed model — never an unexplained fudge factor:

* camera colour mixing     -> fixed 3x3 matrix in linear light
* illuminant / white balance, exposure -> per-channel diagonal gain
* sensor noise             -> Gaussian, seeded
* black level              -> small linear offset before clipping
* specular sheen / clipped channels -> explicit per-patch defects

None of the outputs are meant to be photographic: the manifest lists them as
simulated responses so nobody mistakes them for measured camera data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

from . import colorspace as cs


def derive_seed(*parts: str) -> int:
    """Stable 32-bit seed from meaningful identifiers (no Python hash())."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass(frozen=True)
class CameraProfile:
    name: str
    matrix: np.ndarray          # scene-linear RGB -> camera-linear RGB
    noise_sigma: float
    black_level: float = 0.0

    def forward(self, scene_linear: np.ndarray) -> np.ndarray:
        return np.asarray(scene_linear) @ self.matrix.T


@dataclass(frozen=True)
class Illumination:
    name: str
    gains: tuple[float, float, float]   # per-channel light response + WB
    exposure: float = 1.0
    note: str = ""


def _apply_sensor(
    profile: CameraProfile,
    illumination: Illumination,
    scene_linear: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    cam_lin = profile.forward(scene_linear)
    cam_lin = cam_lin * np.asarray(illumination.gains)[None, :] * illumination.exposure
    cam_lin = cam_lin + profile.black_level
    cam_lin = cam_lin + rng.normal(0.0, profile.noise_sigma, cam_lin.shape)
    return cs.linear_to_srgb(np.clip(cam_lin, 0.0, 1.0))


def render_chart_image(
    profile: CameraProfile,
    illumination: Illumination,
    chart,
    cell: int = 120,
    gap: int = 14,
    background: float = 0.06,
    seed_parts: tuple[str, ...] = ("kit",),
) -> np.ndarray:
    """Render a matte chart as a camera-encoded RGB float image (0..1)."""
    ref_srgb = chart.patch_srgb()
    rows, cols = chart.rows, chart.cols
    H = rows * cell + (rows + 1) * gap
    W = cols * cell + (cols + 1) * gap
    img = np.full((H, W, 3), background, dtype=np.float64)
    rng = np.random.default_rng(
        derive_seed(profile.name, illumination.name, *seed_parts, "chart")
    )
    for idx in range(chart.n_patches):
        r, c = divmod(idx, cols)
        y0, x0 = gap + r * (cell + gap), gap + c * (cell + gap)
        block = np.tile(ref_srgb[idx], (cell, cell, 1))
        img[y0:y0 + cell, x0:x0 + cell] = _apply_sensor(
            profile, illumination, cs.srgb_to_linear(block), rng
        )
    return np.clip(img, 0.0, 1.0)


def chart_corners(rows: int, cols: int, cell: int = 120, gap: int = 14) -> list:
    """Outer chart quad in the rendered image, as [[x,y], ...] TL,TR,BR,BL."""
    H = rows * cell + (rows + 1) * gap
    W = cols * cell + (cols + 1) * gap
    return [
        [gap, gap],
        [W - gap, gap],
        [W - gap, H - gap],
        [gap, H - gap],
    ]


def render_scene_image(
    profile: CameraProfile,
    illumination: Illumination,
    blocks: list[tuple[str, tuple[float, float, float]]],
    block_h: int = 64,
    seed_parts: tuple[str, ...] = ("kit",),
) -> tuple[np.ndarray, list[dict]]:
    """Render vertical colour bands with known reference sRGB values.

    Returns the camera-encoded image and the block geometry/reference list so
    residual evidence can be tied to exact pixel rectangles.
    """
    w = 320
    h = block_h * len(blocks)
    scene = np.zeros((h, w, 3), dtype=np.float64)
    layout: list[dict] = []
    for i, (name, rgb) in enumerate(blocks):
        scene[i * block_h:(i + 1) * block_h] = rgb
        layout.append({
            "name": name,
            "reference_srgb": list(rgb),
            "y0": i * block_h,
            "y1": (i + 1) * block_h,
            "x0": 0,
            "x1": w,
        })
    rng = np.random.default_rng(
        derive_seed(profile.name, illumination.name, *seed_parts, "scene")
    )
    cam_enc = _apply_sensor(profile, illumination, cs.srgb_to_linear(scene), rng)
    return cam_enc, layout


def write_image(path: str, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8),
                       cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def read_image(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
