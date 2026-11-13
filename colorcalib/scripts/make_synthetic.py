"""Generate synthetic chart/product images with a *known* camera transform.

Everything here is simulated, so the fitted mapping can be checked against the
ground-truth 3x3 camera matrix.  The generator deliberately supports the
failure modes the tool must diagnose: white-balance shifts, exposure bias,
channel clipping, specular sheen on some patches, and a misplaced "wrong"
chart variant for detection tests.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from colorcalib.charts import builtin_colorchecker24
from colorcalib.colorspace import (
    linear_to_srgb,
    srgb_to_linear,
    srgb_rgb_to_xyz_d65,
    xyz_d65_to_srgb_rgb,
)

# A deliberately non-diagonal, strictly positive "camera" matrix in
# linear-light space. Row sums are 1 so neutrals stay neutral in level while
# the off-diagonal terms create realistic cross-channel color mixing; all
# entries are positive, so no real color is pushed below zero (clipping then
# only occurs via the explicit clip_patches / exposure scenarios).
TRUE_CAMERA_MATRIX = np.array(
    [
        [0.88, 0.08, 0.04],
        [0.06, 0.87, 0.07],
        [0.05, 0.09, 0.86],
    ]
)


def simulate_camera(
    srgb_encoded: np.ndarray,
    wb_shift=(1.0, 1.0, 1.0),
    exposure=1.0,
    noise_sigma=0.0018,
    clip_above=1.0,
    sheen_patches: set[int] | None = None,
    seed=0,
) -> np.ndarray:
    """Reference sRGB image -> simulated camera sRGB output."""
    rng = np.random.default_rng(seed)
    lin = srgb_to_linear(srgb_encoded)
    cam_lin = lin @ TRUE_CAMERA_MATRIX.T
    cam_lin *= np.asarray(wb_shift)[None, :] * exposure
    if sheen_patches:
        for idx in sheen_patches:
            pass  # applied by caller at patch granularity
    noisy = cam_lin + rng.normal(0, noise_sigma, cam_lin.shape)
    clipped = np.clip(noisy, 0.0, clip_above)
    return linear_to_srgb(clipped)


def render_chart_image(
    out_path: str,
    cell=120,
    wb_shift=(1.0, 1.0, 1.0),
    exposure=1.0,
    clip_patches: set[int] | None = None,
    sheen_patches: set[int] | None = None,
    noise_sigma=0.0018,
    perspective=0.0,
    seed=0,
) -> None:
    chart = builtin_colorchecker24()
    ref_srgb = chart.patch_srgb()
    rows, cols = chart.rows, chart.cols
    gap = 14
    H, W = rows * cell + (rows + 1) * gap, cols * cell + (cols + 1) * gap
    img = np.full((H, W, 3), 0.08)
    rng = np.random.default_rng(seed + 1)

    for idx in range(chart.n_patches):
        r, c = divmod(idx, cols)
        y0, x0 = gap + r * (cell + gap), gap + c * (cell + gap)
        block = np.tile(ref_srgb[idx], (cell, cell, 1))
        lin = srgb_to_linear(block)
        cam_lin = lin @ TRUE_CAMERA_MATRIX.T * np.asarray(wb_shift) * exposure
        if clip_patches and idx in clip_patches:
            # Force one channel fully to the encoding ceiling; real clipping
            # means detail is destroyed, not merely brightened.
            cam_lin[..., 0] = 1.0
            cam_lin = np.clip(cam_lin, 0.0, 1.0)
        if sheen_patches and idx in sheen_patches:
            yy, xx = np.mgrid[0:cell, 0:cell]
            spot = np.exp(-(((xx - cell * 0.3) ** 2 + (yy - cell * 0.3) ** 2)
                            / (2 * (cell * 0.12) ** 2)))
            cam_lin = cam_lin + spot[..., None] * 0.45
        cam_lin += rng.normal(0, noise_sigma, cam_lin.shape)
        img[y0:y0 + cell, x0:x0 + cell] = linear_to_srgb(
            np.clip(cam_lin, 0.0, 1.0)
        )

    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if perspective:
        # A real perspective keystone: the far edge is shorter and shifted,
        # while both heights are preserved, so the grid stays invertible.
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        shift = perspective * W
        dst = np.float32(
            [[shift, 0.03 * H],
             [W - shift, 0.03 * H],
             [W, 0.97 * H],
             [0, 0.97 * H]]
        )
        M = cv2.getPerspectiveTransform(src, dst)
        canvas = np.zeros((int(H * 1.1), int(W * 1.15), 3), np.uint8)
        warped = cv2.warpPerspective(img8, M, (canvas.shape[1], canvas.shape[0]))
        mask = warped.sum(axis=2) > 0
        canvas[mask] = warped[mask]
        img8 = canvas
    cv2.imwrite(out_path, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))


def render_product_image(out_path: str, wb_shift, exposure, seed=3) -> None:
    """A simple synthetic 'product' scene spanning and exceeding chart range."""
    rng = np.random.default_rng(seed)
    h, w = 240, 320
    base = np.zeros((h, w, 3))
    base[:80] = (0.85, 0.2, 0.15)
    base[80:160] = (0.2, 0.55, 0.8)
    base[160:] = (0.9, 0.85, 0.4)
    # some pixels deliberately beyond the chart's calibration domain
    base += rng.normal(0, 0.01, base.shape)
    base = np.clip(base, 0, 1.0)
    cam = simulate_camera(base, wb_shift, exposure, noise_sigma=0.0015, seed=seed)
    cv2.imwrite(out_path, cv2.cvtColor((cam * 255).astype(np.uint8),
                                       cv2.COLOR_RGB2BGR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    scenarios = {
        "nominal": dict(wb_shift=(1.0, 1.0, 1.0), exposure=1.0),
        "warm_wb": dict(wb_shift=(1.22, 1.05, 0.82), exposure=1.0),
        "overexposed": dict(wb_shift=(1.0, 1.0, 1.0), exposure=1.25),
        "underexposed": dict(wb_shift=(1.0, 1.0, 1.0), exposure=0.72),
        "clipped": dict(wb_shift=(1.0, 1.0, 1.0), exposure=1.0,
                        clip_patches={18, 15}),
        "sheen": dict(wb_shift=(1.1, 1.0, 0.9), exposure=1.0,
                      sheen_patches={6, 12}),
        "perspective": dict(wb_shift=(1.15, 1.0, 0.88), exposure=1.05,
                            perspective=0.12),
    }
    for name, kw in scenarios.items():
        render_chart_image(os.path.join(args.outdir, f"chart_{name}.png"), **kw)
        wb = kw.get("wb_shift", (1, 1, 1))
        ex = kw.get("exposure", 1.0)
        render_product_image(
            os.path.join(args.outdir, f"product_{name}.png"), wb, ex
        )
    print(f"wrote {len(scenarios)} scenarios to {args.outdir}")


if __name__ == "__main__":
    main()
