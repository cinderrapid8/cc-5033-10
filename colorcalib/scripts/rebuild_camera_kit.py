"""Rebuild mappings from the dual-camera kit and verify the agreed contract.

This is the scripted *cross-camera calibration* path.  It does three things:

1. re-renders every kit image from ``kit_params.json`` and checks the pixels
   match the on-disk kit (so the kit is genuinely reproducible, not a frozen
   screenshot);
2. fits a mapping per camera per illumination from the kit's chart
   observations, then reports **independent holdout residuals** on colours the
   fit never saw;
3. applies those mappings to the scene images and measures per-block ΔE2000,
   tying each residual to the image and pixel rectangle it came from.

A foreign-camera negative control must *fail* the acceptance thresholds; if it
passes, the kit is not discriminating enough and the script errors.

Usage::

    python scripts/rebuild_camera_kit.py --kit camera_kits/dual_camera_kit \\
        --out out/kit_rebuild
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colorcalib import colorspace as cs
from colorcalib.charts import builtin_colorchecker24
from colorcalib.detection import DetectionError, extract_manual
from colorcalib.pipeline import (
    IllConditionedError,
    InsufficientSamplesError,
    apply_coefficients,
    calibrate,
    camera_encoded_to_linear,
    reference_lab_d50_to_linear_srgb,
    scene_white_gains,
)
from colorcalib.simulate import (
    CameraProfile,
    Illumination,
    chart_corners,
    read_image,
    render_chart_image,
    render_scene_image,
)
from scripts.build_camera_kit import load_params, sha256_file


def effective_matrix(result) -> np.ndarray:
    """Forward camera mapping implied by a fitted result (test reconstruction).

    The fitted coefficients map camera-linear -> reference-linear, so they
    should recover the *inverse* camera matrix; the white-balance gains are a
    diagonal that absorbs the illumination and are reported separately.
    """
    coef = np.asarray(result.coefficients)
    coef = coef / coef[1, 1]
    inv = np.linalg.inv(coef)
    return inv / inv[1, 1]


def predict_with_target_wb(result, detection, reference_linear, neutral_idx):
    """Apply a fitted *color matrix* under the target image's own lighting.

    The cross-camera contract is: the mapping carries the sensor→reference
    colour transform, while white balance is re-derived from the target image's
    neutral patches. This lets one sensor's mapping transfer to another
    illumination of the *same* sensor, but not to a different sensor.
    """
    cam_lin = camera_encoded_to_linear(detection.adopted_rgb)
    gains = scene_white_gains(cam_lin, reference_linear, neutral_idx)
    lin_wb = cam_lin * gains[None, :]
    pred = apply_coefficients(
        lin_wb, np.asarray(result.coefficients), result.model
    )
    return cs.srgb_to_lab(cs.linear_to_srgb(np.clip(pred, 0.0, 1.0)))


def measure_scene(result, scene_rgb: np.ndarray, blocks: list[dict]) -> list[dict]:
    """Per-block ΔE2000 residual of the mapped scene versus its reference."""
    img_lab = cs.srgb_to_lab(
        (np.clip(scene_rgb, 0, 1) * 255).round() / 255
    )
    # Apply the mapping pixel-wise, then convert to Lab.
    lin = camera_encoded_to_linear(scene_rgb.reshape(-1, 3))
    lin_wb = lin * np.asarray(result.white_gains)[None, :]
    pred_lin = apply_coefficients(lin_wb, np.asarray(result.coefficients), result.model)
    out_enc = cs.linear_to_srgb(np.clip(pred_lin, 0.0, 1.0))
    out_lab = cs.srgb_to_lab(out_enc).reshape(scene_rgb.shape[0], scene_rgb.shape[1], 3)
    del img_lab
    rows = []
    for b in blocks:
        y0, y1, x0, x1 = b["y0"], b["y1"], b["x0"], b["x1"]
        ref_lab = cs.srgb_to_lab(np.array([b["reference_srgb"]]), "D50")[0]
        region = out_lab[y0:y1, x0:x1].reshape(-1, 3)
        de = cs.delta_e_2000(region, np.tile(ref_lab, (len(region), 1)))
        rows.append({
            "block": b["name"],
            "reference_srgb": b["reference_srgb"],
            "pixel_rect": [y0, y1, x0, x1],
            "deltaE2000_mean": float(np.mean(de)),
            "deltaE2000_p95": float(np.percentile(de, 95)),
        })
    return rows


def rebuild(kit_dir: str, out_dir: str) -> dict:
    params, params_sha = load_params(kit_dir)
    manifest_path = os.path.join(kit_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest["provenance"]["params_sha256"] != params_sha:
        raise SystemExit(
            "kit_params.json changed since the manifest was built; run "
            "scripts/build_camera_kit.py first"
        )

    chart = builtin_colorchecker24()
    geom = params["render_geometry"]
    cell, gap = int(geom["cell_px"]), int(geom["gap_px"])
    block_h = int(geom.get("scene_block_height_px", 64))
    corners = np.asarray(chart_corners(chart.rows, chart.cols, cell, gap),
                         dtype=np.float32)
    blocks = [(b["name"], tuple(float(v) for v in b["rgb"]))
              for b in params["scene_blocks"]]
    holdout = params["chart"]["holdout_indices"]
    accept = params["acceptance"]

    cams = {
        name: CameraProfile(
            name, np.asarray(spec["matrix"], dtype=np.float64),
            float(spec.get("noise_sigma", 0.0015)),
            float(spec.get("black_level", 0.0)),
        )
        for name, spec in params["camera_profiles"].items()
    }
    ills = {
        name: Illumination(name, tuple(float(g) for g in spec["gains"]),
                           float(spec.get("exposure", 1.0)))
        for name, spec in params["illuminations"].items()
    }

    os.makedirs(out_dir, exist_ok=True)
    source_check = []
    for rec in manifest["files"]:
        path = os.path.join(kit_dir, rec["file"])
        source_check.append({
            "file": rec["file"],
            "sha256_on_disk": sha256_file(path),
            "sha256_manifest": rec["sha256"],
            "matches_manifest": sha256_file(path) == rec["sha256"],
        })

    cameras_out: dict[str, dict] = {}
    failures: list[str] = []
    for cam_name in ("cam_a", "cam_b"):
        cam = cams[cam_name]
        cam_out = {"label": params["camera_profiles"][cam_name].get("label", ""),
                   "illuminations": {}}
        truth = cam.matrix / cam.matrix[1, 1]
        for ill_name in params["imaged_illuminations"]:
            ill = ills[ill_name]
            rec_file = f"charts/chart_{cam_name}_{ill_name}.png"
            scene_file = f"scenes/scene_{cam_name}_{ill_name}.png"
            chart_path = os.path.join(kit_dir, rec_file)
            scene_path = os.path.join(kit_dir, scene_file)

            # 1. reproducibility: re-render, quantize identically to PNG, compare
            regen = render_chart_image(
                cam, ill, chart, cell, gap,
                seed_parts=(params["kit_name"], cam_name, ill_name),
            )
            regen_scene, _ = render_scene_image(
                cam, ill, blocks, block_h,
                seed_parts=(params["kit_name"], cam_name, ill_name),
            )
            on_disk = read_image(chart_path)
            scene_on_disk = read_image(scene_path)
            q = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8) / 255.0
            repro = {
                "chart_max_abs_pixel_diff_8bit": int(np.max(np.abs(
                    (q(regen) * 255).astype(int) - (on_disk * 255).astype(int)
                ))),
                "scene_max_abs_pixel_diff_8bit": int(np.max(np.abs(
                    (q(regen_scene) * 255).astype(int) - (scene_on_disk * 255).astype(int)
                ))),
            }

            try:
                det = extract_manual(
                    chart_path, corners, chart.rows, chart.cols,
                    chart.patch_names, chart.name,
                )
                result = calibrate(
                    det, chart.lab_d50, chart.reference_space,
                    model="matrix3", holdout_indices=holdout,
                )
            except (InsufficientSamplesError, IllConditionedError, DetectionError) as exc:
                failures.append(f"{cam_name}/{ill_name}: refused — {exc}")
                continue

            eff = effective_matrix(result)
            matrix_err = float(np.max(np.abs(eff - truth)))
            scene_rows = measure_scene(result, scene_on_disk,
                                       manifest_scene_blocks(manifest, scene_file))
            max_scene_de = max(r["deltaE2000_mean"] for r in scene_rows)
            cam_out["illuminations"][ill_name] = {
                "chart_file": rec_file,
                "scene_file": scene_file,
                "reproducibility": repro,
                "n_usable": result.detection_summary["n_usable"],
                "train": result.train_summary,
                "cv": result.validation_summary,
                "holdout": result.holdout_summary,
                "holdout_patch_names": params["chart"]["holdout_patch_names"],
                "effective_matrix_recovered": eff.round(6).tolist(),
                "truth_matrix": truth.round(6).tolist(),
                "matrix_max_abs_error": matrix_err,
                "scene_residuals": scene_rows,
                "scene_max_deltaE2000": max_scene_de,
                "white_gains": result.white_gains,
            }
            # contract check
            if result.validation_summary["n_patches"] == 0:
                failures.append(f"{cam_name}/{ill_name}: no CV patches")
            if result.train_summary["deltaE2000_mean"] > accept["max_train_deltaE_mean"]:
                failures.append(
                    f"{cam_name}/{ill_name}: train ΔE "
                    f"{result.train_summary['deltaE2000_mean']:.2f} > "
                    f"{accept['max_train_deltaE_mean']}"
                )
            if result.holdout_summary["deltaE2000_mean"] > accept["max_holdout_deltaE_mean"]:
                failures.append(
                    f"{cam_name}/{ill_name}: holdout ΔE "
                    f"{result.holdout_summary['deltaE2000_mean']:.2f} > "
                    f"{accept['max_holdout_deltaE_mean']}"
                )
            if result.holdout_summary["deltaE2000_max"] > accept["max_absolute_holdout_deltaE"]:
                failures.append(
                    f"{cam_name}/{ill_name}: worst holdout patch ΔE "
                    f"{result.holdout_summary['deltaE2000_max']:.2f} > "
                    f"{accept['max_absolute_holdout_deltaE']}"
                )
            if matrix_err > accept["matrix_recovery_rel_tol"]:
                failures.append(
                    f"{cam_name}/{ill_name}: recovered matrix off by "
                    f"{matrix_err:.3f} > {accept['matrix_recovery_rel_tol']}"
                )
            if result.detection_summary["n_usable"] < accept["min_usable_patches"]:
                failures.append(
                    f"{cam_name}/{ill_name}: only "
                    f"{result.detection_summary['n_usable']} usable patches"
                )
        cameras_out[cam_name] = cam_out

    # Positive control: a cam_a mapping fitted under neutral must still predict
    # cam_a colours under a different illumination (holding the sensor fixed).
    reference_linear = reference_lab_d50_to_linear_srgb(chart.lab_d50)
    neutral_idx = params["chart"]["neutral_indices"]
    lighting_check = holdout_across_illumination(
        kit_dir, params, chart, corners, cams, ills, blocks, block_h,
        reference_linear, neutral_idx,
    )
    if lighting_check["cross_illumination_mean"] > accept["max_holdout_deltaE_mean"]:
        failures.append(
            "same-camera lighting generalization failed: cross-illumination "
            f"holdout ΔE {lighting_check['cross_illumination_mean']:.2f} > "
            f"{accept['max_holdout_deltaE_mean']}"
        )

    # Negative control: the same cam_a mapping must misfit the foreign camera.
    control = negative_control(
        kit_dir, params, chart, corners, cams, ills,
        reference_linear=reference_linear, neutral_idx=neutral_idx,
    )
    if control["foreign_fullchart_deltaE_mean"] <= accept["max_holdout_deltaE_mean"]:
        failures.append(
            "negative control passed: foreign-camera full-chart ΔE "
            f"{control['foreign_fullchart_deltaE_mean']:.2f} <= "
            f"{accept['max_holdout_deltaE_mean']} — kit does not reject an "
            "unrelated camera"
        )

    source_ok = all(r["matches_manifest"] for r in source_check)
    if accept.get("require_source_reproducible") and not source_ok:
        failures.append("kit files on disk do not match their recorded SHA-256")

    verdict = {
        "schema": "colorcalib.kit_verification/v1",
        "kit_name": params["kit_name"],
        "params_sha256": params_sha,
        "source_reproducible": source_ok,
        "source_checks": source_check,
        "cameras": cameras_out,
        "lighting_generalization": lighting_check,
        "negative_control": control,
        "acceptance": accept,
        "passed": len(failures) == 0,
        "failures": failures,
    }
    with open(os.path.join(out_dir, "kit_verification.json"), "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2)
    return verdict


def manifest_scene_blocks(manifest: dict, scene_file: str) -> list[dict]:
    for rec in manifest["files"]:
        if rec["file"] == scene_file:
            return rec["blocks"]
    raise KeyError(scene_file)


def holdout_across_illumination(kit_dir, params, chart, corners, cams, ills,
                                blocks, block_h, reference_linear, neutral_idx):
    """Fit on cam_a neutral; verify holdout colours on cam_a warm."""
    val_ill = ills[params["holdout_validation_illumination"]]
    fit_path = os.path.join(kit_dir, "charts", "chart_cam_a_neutral.png")
    val_path = os.path.join(kit_dir, "charts", f"chart_cam_a_{val_ill.name}.png")
    det_fit = extract_manual(fit_path, corners, chart.rows, chart.cols,
                             chart.patch_names, chart.name)
    det_val = extract_manual(val_path, corners, chart.rows, chart.cols,
                             chart.patch_names, chart.name)
    holdout = params["chart"]["holdout_indices"]
    result = calibrate(det_fit, chart.lab_d50, chart.reference_space,
                       model="matrix3", holdout_indices=holdout)
    pred_lab = predict_with_target_wb(
        result, det_val, reference_linear, neutral_idx
    )
    de = cs.delta_e_2000(pred_lab[holdout], chart.lab_d50[holdout])
    return {
        "fit_chart": "charts/chart_cam_a_neutral.png",
        "validation_chart": f"charts/chart_cam_a_{val_ill.name}.png",
        "holdout_patch_names": [chart.patch_names[i] for i in holdout],
        "cross_illumination_deltaE2000": [round(float(v), 3) for v in de],
        "cross_illumination_mean": round(float(np.mean(de)), 3),
        "cross_illumination_max": round(float(np.max(de)), 3),
        "within_illumination_holdout_mean": result.holdout_summary["deltaE2000_mean"],
    }


def negative_control(kit_dir, params, chart, corners, cams, ills, *,
                     reference_linear, neutral_idx):
    """Fit cam_a neutral, then apply it to the foreign camera's chart."""
    fit_path = os.path.join(kit_dir, "charts", "chart_cam_a_neutral.png")
    foreign_path = os.path.join(kit_dir, "control", "chart_foreign_cam_neutral.png")
    det_fit = extract_manual(fit_path, corners, chart.rows, chart.cols,
                             chart.patch_names, chart.name)
    det_foreign = extract_manual(foreign_path, corners, chart.rows, chart.cols,
                                 chart.patch_names, chart.name)
    holdout = params["chart"]["holdout_indices"]
    result = calibrate(det_fit, chart.lab_d50, chart.reference_space,
                       model="matrix3", holdout_indices=holdout)
    pred_lab = predict_with_target_wb(
        result, det_foreign, reference_linear, neutral_idx
    )
    de_all = cs.delta_e_2000(pred_lab, chart.lab_d50)
    de = cs.delta_e_2000(pred_lab[holdout], chart.lab_d50[holdout])
    return {
        "fit_chart": "charts/chart_cam_a_neutral.png",
        "control_chart": "control/chart_foreign_cam_neutral.png",
        "foreign_fullchart_deltaE_mean": round(float(np.mean(de_all)), 3),
        "foreign_fullchart_deltaE_max": round(float(np.max(de_all)), 3),
        "foreign_holdout_deltaE2000": [round(float(v), 3) for v in de],
        "foreign_holdout_deltaE_mean": round(float(np.mean(de)), 3),
        "foreign_holdout_deltaE_max": round(float(np.max(de)), 3),
        "reject_rule": "full-chart mean ΔE must exceed max_holdout_deltaE_mean",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    verdict = rebuild(args.kit, args.out)
    print(json.dumps({
        "passed": verdict["passed"],
        "failures": verdict["failures"],
        "negative_control_fullchart_deltaE":
            verdict["negative_control"]["foreign_fullchart_deltaE_mean"],
        "lighting_deltaE":
            verdict["lighting_generalization"]["cross_illumination_mean"],
    }, indent=2))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
