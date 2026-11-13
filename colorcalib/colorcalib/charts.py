"""Reference color-chart definitions.

Two provenance classes are supported:

* built-in classic 24-patch ColorChecker with published D50 L*a*b* reference
  values (post-2014 X-Rite data);
* user-supplied charts via JSON, with explicit reference values and the color
  space they are stated in, so encoded RGB and reference Lab/XYZ are never
  silently mixed.

A user JSON looks like::

    {
      "name": "my-chart",
      "layout": {"rows": 4, "cols": 6, "order": "row-major"},
      "reference_space": {"type": "lab", "illuminant": "D50"},
      "patches": [
        {"name": "dark_skin", "lab": [37.99, 13.56, 14.06]},
        {"name": "blue", "lab": [22.27, 15.69, -34.70]}
      ]
    }

or with sRGB-encoded reference values, in which case the declared white point
and encoding must be given and are converted to D50 Lab on load::

    {
      "reference_space": {"type": "srgb", "illuminant": "D65", "encoding": "sRGB"},
      "patches": [{"name": "r", "rgb": [255, 0, 0]}]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from . import colorspace as cs

# Post-Nov-2014 X-Rite ColorChecker Classic reference values, D50 Lab.
# Order: rows top-to-bottom, each row left-to-right (24 patches, 4x6).
COLORCHECKER_24_LAB_D50: list[tuple[str, tuple[float, float, float]]] = [
    ("dark_skin", (37.99, 13.56, 14.06)),
    ("light_skin", (65.71, 18.13, 17.81)),
    ("blue_sky", (49.93, -4.88, -21.93)),
    ("foliage", (43.14, -13.10, 21.91)),
    ("blue_flower", (55.11, 8.84, -25.40)),
    ("bluish_green", (70.72, -33.40, -0.20)),
    ("orange", (62.66, 36.07, 57.10)),
    ("purplish_blue", (40.02, 10.41, -45.96)),
    ("moderate_red", (51.12, 48.24, 16.25)),
    ("purple", (30.33, 22.98, -21.59)),
    ("yellow_green", (72.53, -23.71, 57.26)),
    ("orange_yellow", (71.94, 19.36, 67.86)),
    ("blue", (22.27, 15.69, -34.70)),
    ("green", (42.10, -31.95, 22.79)),
    ("red", (40.95, 52.25, 25.63)),
    ("yellow", (81.73, 4.04, 79.82)),
    ("magenta", (51.94, 49.99, -14.57)),
    ("cyan", (51.04, -28.63, -28.64)),
    ("white", (96.54, -0.43, 1.19)),
    ("neutral_9", (81.26, -0.64, -0.34)),
    ("neutral_7_5", (66.77, -0.73, -0.50)),
    ("neutral_6_5", (50.87, -0.15, -0.27)),
    ("neutral_5", (35.66, -0.42, -1.23)),
    ("black", (20.46, -0.08, -0.97)),
]


@dataclass
class Chart:
    """A color chart with geometry and reference values in a known space."""

    name: str
    rows: int
    cols: int
    patch_names: list[str]
    lab_d50: np.ndarray  # (N, 3) CIE Lab under D50 — the canonical reference
    reference_space: dict
    raw_reference: list[dict] = field(default_factory=list)

    @property
    def n_patches(self) -> int:
        return len(self.patch_names)

    def grid_index(self, index: int) -> tuple[int, int]:
        """Return (row, col) for a row-major patch index."""
        return divmod(index, self.cols)

    def patch_xyz_d50(self) -> np.ndarray:
        return cs.lab_to_xyz(self.lab_d50, cs.white_xyz("D50"))

    def patch_srgb(self) -> np.ndarray:
        """Reference patches rendered as sRGB-encoded 0..1 values (for UI)."""
        xyz_d50 = self.patch_xyz_d50()
        cat = cs.chromatic_adaptation_matrix(cs.white_xyz("D50"), cs.white_xyz("D65"))
        xyz_d65 = cs.adapt_xyz(xyz_d50, cat)
        lin = cs.xyz_d65_to_srgb_rgb(xyz_d65)
        return np.clip(cs.linear_to_srgb(lin), 0.0, 1.0)


def builtin_colorchecker24() -> Chart:
    names = [n for n, _ in COLORCHECKER_24_LAB_D50]
    lab = np.array([v for _, v in COLORCHECKER_24_LAB_D50], dtype=np.float64)
    return Chart(
        name="xrite-colorchecker-classic-24",
        rows=4,
        cols=6,
        patch_names=names,
        lab_d50=lab,
        reference_space={"type": "lab", "illuminant": "D50", "source": "X-Rite 2014"},
        raw_reference=[{"name": n, "lab": list(v)} for n, v in COLORCHECKER_24_LAB_D50],
    )


def load_chart_json(path: str) -> Chart:
    """Load a user chart JSON and normalize its references to D50 Lab."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    layout = data.get("layout", {})
    rows = int(layout.get("rows", 0))
    cols = int(layout.get("cols", 0))
    patches = data["patches"]
    if rows and cols and rows * cols != len(patches):
        raise ValueError(
            f"layout {rows}x{cols} = {rows * cols} cells but {len(patches)} patches given"
        )
    if not rows:
        rows, cols = 1, len(patches)

    space = data["reference_space"]
    stype = space.get("type", "").lower()
    names: list[str] = []
    labs: list[np.ndarray] = []

    if stype == "lab":
        ill = space.get("illuminant", "D50").upper()
        wp = cs.white_xyz(ill)
        for i, p in enumerate(patches):
            names.append(p.get("name", f"patch_{i:02d}"))
            lab = np.asarray(p["lab"], dtype=np.float64)
            if ill != "D50":
                xyz = cs.lab_to_xyz(lab, wp)
                cat = cs.chromatic_adaptation_matrix(wp, cs.white_xyz("D50"))
                lab = cs.xyz_to_lab(cs.adapt_xyz(xyz, cat), cs.white_xyz("D50"))
            labs.append(lab)
    elif stype in ("srgb", "rgb"):
        encoding = space.get("encoding", "sRGB").lower()
        if encoding != "srgb":
            raise ValueError(f"unsupported RGB encoding '{encoding}' (only sRGB)")
        ill = space.get("illuminant", "D65").upper()
        if ill != "D65":
            raise ValueError("sRGB references must be declared under D65")
        for i, p in enumerate(patches):
            names.append(p.get("name", f"patch_{i:02d}"))
            rgb = np.asarray(p["rgb"], dtype=np.float64)
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            labs.append(cs.srgb_to_lab(np.clip(rgb, 0, 1), "D50")[0])
    else:
        raise ValueError(f"reference_space.type must be 'lab' or 'srgb', got {stype!r}")

    return Chart(
        name=data.get("name", "custom-chart"),
        rows=rows,
        cols=cols,
        patch_names=names,
        lab_d50=np.asarray(labs, dtype=np.float64),
        reference_space=space,
        raw_reference=patches,
    )
