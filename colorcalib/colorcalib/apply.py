"""Batch application of a saved mapping to product images."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .pipeline import apply_mapping_to_rgb, load_mapping

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def apply_batch(
    input_dir: str,
    mapping_path: str,
    output_dir: str,
    report_path: str | None = None,
) -> dict:
    """Apply the mapping to every supported image under ``input_dir``.

    Writes converted images to ``output_dir`` and a per-image extrapolation
    report (pixels outside the calibration range are reported, not hidden).
    """
    mapping = load_mapping(mapping_path)
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for name in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTS:
            continue
        src = os.path.join(input_dir, name)
        bgr = cv2.imread(src, cv2.IMREAD_COLOR)
        if bgr is None:
            rows.append({"image": name, "status": "unreadable"})
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out_rgb, report = apply_mapping_to_rgb(rgb, mapping)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        stem = os.path.splitext(name)[0]
        dst = os.path.join(output_dir, f"{stem}_calibrated{ext}")
        cv2.imwrite(dst, out_bgr)
        row = {"image": name, "status": "ok", "output": os.path.basename(dst)}
        row.update(report)
        rows.append(row)

    summary = {
        "mapping": mapping_path,
        "model": mapping["model"],
        "n_images": len(rows),
        "images": rows,
    }
    if report_path:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return summary
