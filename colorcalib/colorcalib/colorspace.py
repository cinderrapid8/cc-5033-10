"""Color-space conversions and color-difference metrics.

Conventions
-----------
* Camera-encoded RGB and sRGB reference values are treated as IEC 61966-2-1
  sRGB encodings (0..1) unless a different EOTF is explicitly supplied.
* All *linear* RGB values are scene-linear, normalized to 0..1.
* XYZ uses CIE 1931 standard observer. White points are given as 2-degree
  illuminant xy chromaticities and converted to XYZ with Y == 1.
* Lab is computed from adapted XYZ under its own illuminant (D50 for the
  reference data; per-image white for camera data when evaluating Lab
  differences under the capture illuminant).

Nothing in this module mixes camera-encoded numbers with reference XYZ/Lab:
the pipeline linearizes and chromatically-adapts before any fitting.
"""

from __future__ import annotations

import numpy as np

# --- Illuminant chromaticities (CIE xy, 2 degree) -------------------------
WHITE_POINTS_XY = {
    "D65": (0.31270, 0.32900),
    "D50": (0.34567, 0.35850),
    "D55": (0.33242, 0.34743),
    "D75": (0.29902, 0.31485),
    "A": (0.44758, 0.40745),
    "E": (1.0 / 3.0, 1.0 / 3.0),
}

# sRGB primaries (IEC 61966-2-1), red/green/blue xy.
_SRGB_PRIMARIES = np.array(
    [[0.6400, 0.3300], [0.3000, 0.6000], [0.1500, 0.0600]]
)


def xy_to_xyz(xy: tuple[float, float]) -> np.ndarray:
    """Return an XYZ white point (Y = 1) for CIE xy chromaticity."""
    x, y = xy
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def white_xyz(name: str) -> np.ndarray:
    return xy_to_xyz(WHITE_POINTS_XY[name.upper()])


# --- sRGB transfer function ------------------------------------------------
def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Decode sRGB-encoded values (0..1) to scene-linear light."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Encode scene-linear values (0..1) as sRGB, clipping negatives first."""
    c = np.asarray(c, dtype=np.float64)
    c = np.maximum(c, 0.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


# --- RGB <-> XYZ -----------------------------------------------------------
def _rgb_to_xyz_matrix(primaries_xy: np.ndarray, wp_xyz: np.ndarray) -> np.ndarray:
    """Build the RGB->XYZ matrix for given primaries and white point.

    Columns of the matrix are the scaled primary XYZ direction vectors; the
    scaling vector S solves P @ S = white (not P^T @ S).
    """
    xy = np.asarray(primaries_xy)
    p = np.array(
        [[xy[i, 0] / xy[i, 1], 1.0, (1 - xy[i, 0] - xy[i, 1]) / xy[i, 1]]
         for i in range(3)]
    ).T  # columns are primary XYZ directions
    s = np.linalg.solve(p, wp_xyz)
    return p * s[np.newaxis, :]  # broadcast scale down each column


SRGB_TO_XYZ_D65 = _rgb_to_xyz_matrix(_SRGB_PRIMARIES, white_xyz("D65"))
XYZ_D65_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ_D65)


def srgb_rgb_to_xyz_d65(rgb_linear: np.ndarray) -> np.ndarray:
    return np.asarray(rgb_linear) @ SRGB_TO_XYZ_D65.T


def xyz_d65_to_srgb_rgb(xyz: np.ndarray) -> np.ndarray:
    return np.asarray(xyz) @ XYZ_D65_TO_SRGB.T


# --- Bradford chromatic adaptation -----------------------------------------
_BRADFORD = np.array(
    [
        [0.8951000, 0.2664000, -0.1614000],
        [-0.7502000, 1.7135000, 0.0367000],
        [0.0389000, -0.0685000, 0.0296000],
    ]
)
_BRADFORD_INV = np.linalg.inv(_BRADFORD)


def chromatic_adaptation_matrix(
    src_white_xyz: np.ndarray, dst_white_xyz: np.ndarray
) -> np.ndarray:
    """Return a 3x3 XYZ->XYZ Bradford CAT matrix between two white points."""
    src = np.asarray(src_xyz := np.asarray(src_white_xyz, dtype=np.float64))
    dst = np.asarray(dst_white_xyz, dtype=np.float64)
    src_cone = _BRADFORD @ src
    dst_cone = _BRADFORD @ dst
    scale = np.diag(dst_cone / src_cone)
    return _BRADFORD_INV @ scale @ _BRADFORD


def adapt_xyz(xyz: np.ndarray, cat: np.ndarray) -> np.ndarray:
    return np.asarray(xyz) @ cat.T


# --- CIE Lab ----------------------------------------------------------------
def _lab_f(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4.0 / 29.0)


def _lab_f_inv(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta, t**3, 3 * delta**2 * (t - 4.0 / 29.0))


def xyz_to_lab(xyz: np.ndarray, white: np.ndarray) -> np.ndarray:
    """XYZ (any illuminant) -> CIE L*a*b* relative to the matching white XYZ."""
    xyz = np.asarray(xyz, dtype=np.float64)
    white = np.asarray(white, dtype=np.float64)
    f = _lab_f(xyz / white)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab: np.ndarray, white: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = lab[..., 1] / 500.0 + fy
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    return _lab_f_inv(f) * np.asarray(white, dtype=np.float64)


# --- CIEDE2000 (Sharma et al. formulation) ---------------------------------
def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 color difference between arrays of L*a*b* triples."""
    lab1 = np.atleast_2d(np.asarray(lab1, dtype=np.float64))
    lab2 = np.atleast_2d(np.asarray(lab2, dtype=np.float64))
    L1, a1, b1 = lab1.T
    L2, a2, b2 = lab2.T

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = (C1 + C2) / 2.0

    G = 0.5 * (1.0 - np.sqrt(C_bar**7 / (C_bar**7 + 25.0**7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = np.zeros_like(dLp)
    both_chroma = (C1p * C2p) > 0
    diff = h2p - h1p
    dhp[both_chroma & (np.abs(diff) <= 180.0)] = diff[both_chroma & (np.abs(diff) <= 180.0)]
    dhp[both_chroma & (diff > 180.0)] = diff[both_chroma & (diff > 180.0)] - 360.0
    dhp[both_chroma & (diff < -180.0)] = diff[both_chroma & (diff < -180.0)] + 360.0

    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    L_bar_p = (L1 + L2) / 2.0
    C_bar_p = (C1p + C2p) / 2.0

    h_bar_p = np.zeros_like(L1)
    mask = both_chroma
    hsum = h1p + h2p
    h_bar_p[mask & (np.abs(h1p[mask] - h2p[mask]) <= 180.0)] = (
        hsum[mask & (np.abs(h1p[mask] - h2p[mask]) <= 180.0)] / 2.0
    )
    m2 = mask & (np.abs(h1p - h2p) > 180.0)
    h_bar_p[m2 & (hsum < 360.0)] = (hsum[m2 & (hsum < 360.0)] + 360.0) / 2.0
    h_bar_p[m2 & (hsum >= 360.0)] = (hsum[m2 & (hsum >= 360.0)] - 360.0) / 2.0
    h_bar_p[~mask] = h1p[~mask] + h2p[~mask]

    T = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0))
    )

    d_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    R_C = 2.0 * np.sqrt(C_bar_p**7 / (C_bar_p**7 + 25.0**7))
    S_L = 1.0 + (0.015 * (L_bar_p - 50.0) ** 2) / np.sqrt(
        20.0 + (L_bar_p - 50.0) ** 2
    )
    S_C = 1.0 + 0.045 * C_bar_p
    S_H = 1.0 + 0.015 * C_bar_p * T
    R_T = -np.sin(np.radians(2.0 * d_theta)) * R_C

    kL = kC = kH = 1.0
    dE = np.sqrt(
        (dLp / (kL * S_L)) ** 2
        + (dCp / (kC * S_C)) ** 2
        + (dHp / (kH * S_H)) ** 2
        + R_T * (dCp / (kC * S_C)) * (dHp / (kH * S_H))
    )
    return dE


# --- Convenience: encoded sRGB -> Lab under a chosen white ------------------
def srgb_to_lab(rgb_encoded: np.ndarray, white_name: str = "D50") -> np.ndarray:
    """Full decode chain: sRGB encoding -> linear -> XYZ(D65) -> Lab(white).

    Reference color charts commonly publish Lab under D50, so the default
    adaptation target is D50. Camera data should instead be compared in the
    same adapted frame as the fitted model (see pipeline module).
    """
    rgb = np.atleast_2d(np.asarray(rgb_encoded, dtype=np.float64))
    lin = srgb_to_linear(rgb)
    xyz_d65 = srgb_rgb_to_xyz_d65(lin)
    if white_name.upper() == "D65":
        xyz_w = xyz_d65
        wp = white_xyz("D65")
    else:
        cat = chromatic_adaptation_matrix(white_xyz("D65"), white_xyz(white_name))
        xyz_w = adapt_xyz(xyz_d65, cat)
        wp = white_xyz(white_name)
    return xyz_to_lab(xyz_w, wp)
