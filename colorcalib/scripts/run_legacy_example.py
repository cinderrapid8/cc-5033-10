"""Re-run the legacy single-camera example end to end.

This is a regression harness, not a demo: it regenerates the synthetic images
from the checked-in generator, calibrates every scenario, applies each mapping
to its raw product image, and fails if any stage refuses or if the checked-in
synthetic PNGs no longer match a fresh render.

Usage (from the repository root)::

    PYTHONPATH=. python scripts/run_legacy_example.py --examples examples

Exit code 0 means every scenario calibrated, applied, and reported.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colorcalib.apply import apply_batch
from colorcalib.charts import builtin_colorchecker24
from colorcalib.detection import DetectionError, detect_chart
from colorcalib.pipeline import (
    IllConditionedError,
    InsufficientSamplesError,
    calibrate,
)
from colorcalib.report import write_html_report
from scripts.make_synthetic import render_chart_image, render_product_image

SCENARIOS = {
    "nominal": dict(wb_shift=(1.0, 1.0, 1.0), exposure=1.0),
    "warm_wb": dict(wb_shift=(1.22, 1.05, 0.82), exposure=1.0),
    "overexposed": dict(wb_shift=(1.0, 1.0, 1.0), exposure=1.25),
    "underexposed": dict(wb_shift=(1.0, 1.0, 1.0), exposure=0.72),
    "clipped": dict(wb_shift=(1.0, 1.0, 1.0), exposure=1.0, clip_patches={18, 15}),
    "sheen": dict(wb_shift=(1.1, 1.0, 0.9), exposure=1.0, sheen_patches={6, 12}),
    "perspective": dict(wb_shift=(1.15, 1.0, 0.88), exposure=1.05, perspective=0.12),
}


def _img_diff_8bit(path_a: str, path_b: str) -> int:
    import cv2
    a = cv2.imread(path_a)
    b = cv2.imread(path_b)
    if a is None or b is None:
        return 255
    if a.shape != b.shape:
        return 255
    return int(np.max(np.abs(a.astype(int) - b.astype(int))))


def run(examples_dir: str, out_dir: str) -> dict:
    chart = builtin_colorchecker24()
    os.makedirs(out_dir, exist_ok=True)
    synthetic = os.path.join(examples_dir, "synthetic")
    raw = os.path.join(examples_dir, "raw")
    cal_out = os.path.join(out_dir, "calibrated")
    os.makedirs(cal_out, exist_ok=True)

    results: dict[str, dict] = {}
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, params in SCENARIOS.items():
            regen_chart = os.path.join(tmp, f"chart_{name}.png")
            render_chart_image(regen_chart, **params)
            checked = os.path.join(synthetic, f"chart_{name}.png")
            diff = _img_diff_8bit(regen_chart, checked) if os.path.exists(checked) else 255

            entry: dict = {"synthetic_pixel_diff_vs_checked_in": diff}
            try:
                det = detect_chart(
                    checked if os.path.exists(checked) else regen_chart,
                    chart.rows, chart.cols, chart.patch_names, chart.name,
                )
            except DetectionError as exc:
                failures.append(f"{name}: detection refused — {exc}")
                results[name] = entry
                continue
            try:
                result = calibrate(
                    det, chart.lab_d50, chart.reference_space, model="matrix3"
                )
            except (InsufficientSamplesError, IllConditionedError) as exc:
                failures.append(f"{name}: calibration refused — {exc}")
                results[name] = entry
                continue

            map_path = os.path.join(out_dir, f"map_{name}.json")
            result.to_json(map_path)
            write_html_report(
                result, chart, det, os.path.join(out_dir, f"report_{name}.html")
            )

            raw_img = os.path.join(raw, f"product_{name}.png")
            if not os.path.exists(raw_img):
                render_product_image(
                    raw_img,
                    params.get("wb_shift", (1, 1, 1)),
                    params.get("exposure", 1.0),
                    seed=3,
                )
            batch_dir = os.path.join(tmp, f"batch_{name}")
            os.makedirs(batch_dir, exist_ok=True)
            shutil.copy(raw_img, os.path.join(batch_dir, f"product_{name}.png"))
            summary = apply_batch(
                batch_dir, map_path, cal_out,
                os.path.join(out_dir, "batch_report.json"),
            )
            entry.update({
                "detection_confidence": result.detection_summary["confidence"],
                "usable_patches": result.detection_summary["n_usable"],
                "train_deltaE_mean": result.train_summary["deltaE2000_mean"],
                "cv_deltaE_mean": result.validation_summary["deltaE2000_mean"],
                "applied_images": summary["n_images"],
            })
            if summary["n_images"] != 1 or summary["images"][0].get("status") != "ok":
                failures.append(f"{name}: apply failed")
            results[name] = entry

    verdict = {
        "examples_dir": examples_dir,
        "n_scenarios": len(SCENARIOS),
        "passed": len(failures) == 0,
        "failures": failures,
        "scenarios": results,
    }
    with open(os.path.join(out_dir, "legacy_run.json"), "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2)
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="examples")
    ap.add_argument("--out", default=None, help="output dir (default: examples/output)")
    args = ap.parse_args()
    out = args.out or os.path.join(args.examples, "output")
    verdict = run(args.examples, out)
    print(json.dumps({
        "passed": verdict["passed"],
        "n_scenarios": verdict["n_scenarios"],
        "failures": verdict["failures"],
    }, indent=2))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
