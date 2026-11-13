"""Command-line interface for the color calibration tool.

Examples
--------
Calibrate from an auto-detected chart and save a real mapping file::

    python -m colorcalib.cli calibrate --chart-img chart.jpg \\
        --model matrix3 --out mapping.json --report report.html

Apply the mapping to a folder of product images::

    python -m colorcalib.cli apply --images ./raw --mapping mapping.json \\
        --outdir ./calibrated --report batch.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import charts as chart_mod
from .detection import DetectionError, detect_chart, extract_manual
from .pipeline import (
    InsufficientSamplesError,
    IllConditionedError,
    calibrate,
)
from .apply import apply_batch
from .report import write_html_report


def _load_chart(args) -> chart_mod.Chart:
    if args.chart_ref:
        return chart_mod.load_chart_json(args.chart_ref)
    return chart_mod.builtin_colorchecker24()


def cmd_calibrate(args) -> int:
    chart = _load_chart(args)
    try:
        if args.corners:
            corners = np.array(json.loads(args.corners), dtype=np.float32)
            detection = extract_manual(
                args.chart_img, corners, chart.rows, chart.cols,
                chart.patch_names, chart.name,
            )
        else:
            detection = detect_chart(
                args.chart_img, chart.rows, chart.cols,
                chart.patch_names, chart.name,
            )
    except DetectionError as exc:
        print(f"[detection] {exc}", file=sys.stderr)
        return 2

    val_idx = None
    if args.validation_indices:
        val_idx = [int(x) for x in args.validation_indices.split(",")]
    hold_idx = None
    if args.holdout_indices:
        hold_idx = [int(x) for x in args.holdout_indices.split(",")]

    try:
        result = calibrate(
            detection,
            chart.lab_d50,
            chart.reference_space,
            model=args.model,
            validation_indices=val_idx,
            holdout_indices=hold_idx,
            robust=not args.no_robust,
        )
    except (InsufficientSamplesError, IllConditionedError) as exc:
        print(f"[refused] {exc}", file=sys.stderr)
        return 3

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result.to_json(args.out)
    print(f"[ok] mapping written: {args.out}")
    print(json.dumps(result.train_summary, indent=2))
    if result.validation_summary:
        print("validation (held-out patches):")
        print(json.dumps(result.validation_summary, indent=2))
    if result.holdout_summary:
        print("independent holdout (never used by gains, folds, or the fit):")
        print(json.dumps(result.holdout_summary, indent=2))
    for w in result.warnings:
        print(f"[warning] {w}")

    if args.report:
        write_html_report(result, chart, detection, args.report)
        print(f"[ok] report written: {args.report}")
    return 0


def cmd_apply(args) -> int:
    summary = apply_batch(
        args.images, args.mapping, args.outdir, args.report
    )
    flagged = [
        r for r in summary["images"]
        if r.get("fraction_above_calibration_range", 0) > 0.01
    ]
    print(f"[ok] processed {summary['n_images']} image(s) -> {args.outdir}")
    for r in flagged:
        print(
            f"[extrapolation] {r['image']}: "
            f"{r['fraction_above_calibration_range']:.1%} pixels above "
            "calibration range"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="colorcalib", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("calibrate", help="fit a mapping from a chart image")
    c.add_argument("--chart-img", required=True, help="photo of the color chart")
    c.add_argument(
        "--chart-ref",
        help="custom chart reference JSON (default: built-in ColorChecker 24)",
    )
    c.add_argument(
        "--model", choices=["matrix3", "affine3x4", "poly3"], default="matrix3"
    )
    c.add_argument("--out", required=True, help="output mapping JSON path")
    c.add_argument("--report", help="optional HTML report path")
    c.add_argument(
        "--corners",
        help='manual outer corners JSON, e.g. [[x,y],[x,y],[x,y],[x,y]]',
    )
    c.add_argument(
        "--validation-indices",
        help="comma-separated patch indices to force into the holdout set",
    )
    c.add_argument(
        "--holdout-indices",
        help="comma-separated patch indices to EXCLUDE from the fit entirely "
             "(independent generalization check, e.g. a separate chart image)",
    )
    c.add_argument("--no-robust", action="store_true", help="disable IRLS fitting")
    c.set_defaults(func=cmd_calibrate)

    a = sub.add_parser("apply", help="batch-apply a mapping JSON")
    a.add_argument("--images", required=True, help="directory of product images")
    a.add_argument("--mapping", required=True)
    a.add_argument("--outdir", required=True)
    a.add_argument("--report", help="optional batch report JSON path")
    a.set_defaults(func=cmd_apply)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
