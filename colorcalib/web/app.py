"""Flask web interface for chart calibration and batch application.

Run with::

    python -m web.app --host 127.0.0.1 --port 5000

The interface deliberately exposes the numerical evidence (per-patch adopted
values, fit and cross-validated ΔE2000, residuals, diagnostics) and the actual
mapping coefficients. It offers no saturation/hue sliders: calibration is a
fitted color transform with stated error, not subjective grading.
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import uuid

import cv2
import numpy as np
from flask import (
    Flask, Response, jsonify, render_template, request, send_file, url_for,
)

from colorcalib import colorspace as cs
from colorcalib.charts import builtin_colorchecker24, load_chart_json
from colorcalib.detection import (
    DetectionError, detect_chart, extract_manual,
)
from colorcalib.pipeline import (
    IllConditionedError, InsufficientSamplesError, apply_coefficients,
    calibrate, camera_encoded_to_linear,
    reference_lab_d50_to_linear_srgb, scene_white_gains,
)
from colorcalib.apply import apply_batch
from colorcalib.report import write_html_report

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

# In-process job store (single-user local tool). Each job holds uploaded files
# and the calibration result keyed by an opaque id.
JOBS: dict[str, dict] = {}
WORK_ROOT = tempfile.mkdtemp(prefix="colorcalib_web_")

# Camera kits live next to the package. A kit is any directory containing a
# manifest.json plus a kit_params.json; loading is read-only.
KIT_ROOT = os.environ.get(
    "COLORCALIB_KIT_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "camera_kits"),
)
_KIT_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _save_upload(file_storage, subdir: str) -> str:
    job_dir = os.path.join(WORK_ROOT, subdir)
    os.makedirs(job_dir, exist_ok=True)
    name = file_storage.filename or "upload.png"
    safe = os.path.basename(name).replace(os.sep, "_")
    path = os.path.join(job_dir, safe)
    file_storage.save(path)
    return path


# ---------------------------------------------------------------------------
# Camera-kit loading (read-only, from the repository's camera_kits/*)
# ---------------------------------------------------------------------------
def _kit_dirs() -> list[str]:
    if not os.path.isdir(KIT_ROOT):
        return []
    return sorted(
        d for d in os.listdir(KIT_ROOT)
        if _KIT_NAME_RE.match(d)
        and os.path.isdir(os.path.join(KIT_ROOT, d))
        and os.path.exists(os.path.join(KIT_ROOT, d, "manifest.json"))
    )


def _load_kit(kit_name: str) -> dict:
    if not _KIT_NAME_RE.match(kit_name or ""):
        raise KeyError("invalid kit name")
    kit_dir = os.path.join(KIT_ROOT, kit_name)
    manifest_path = os.path.join(kit_dir, "manifest.json")
    params_path = os.path.join(kit_dir, "kit_params.json")
    if not (os.path.isfile(manifest_path) and os.path.isfile(params_path)):
        raise KeyError(f"kit {kit_name!r} not found")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(params_path, "r", encoding="utf-8") as fh:
        params = json.load(fh)
    return {"dir": kit_dir, "manifest": manifest, "params": params}


def _kit_png_data_uri(kit_dir: str, rel_file: str, max_w: int = 900) -> str:
    path = os.path.join(kit_dir, rel_file)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return ""
    if bgr.shape[1] > max_w:
        scale = max_w / bgr.shape[1]
        bgr = cv2.resize(bgr, (max_w, int(bgr.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _kit_files(manifest: dict, role: str | None = None) -> list[dict]:
    files = manifest.get("files", [])
    if role is None:
        return files
    return [f for f in files if f.get("role") == role]


@app.route("/kits")
def list_kits():
    kits = []
    for name in _kit_dirs():
        try:
            kit = _load_kit(name)
        except (KeyError, json.JSONDecodeError):
            continue
        manifest = kit["manifest"]
        params = kit["params"]
        kits.append({
            "name": name,
            "kit_name": manifest.get("kit_name", name),
            "kit_class": manifest.get("kit_class"),
            "simulation_note": manifest.get("simulation_note"),
            "cameras": list(params.get("camera_profiles", {}).keys()),
            "illuminations": list(params.get("imaged_illuminations", [])),
            "holdout_patch_names": params.get("chart", {}).get("holdout_patch_names", []),
            "n_files": len(manifest.get("files", [])),
        })
    return jsonify({"kits": kits, "kit_root": KIT_ROOT})


@app.route("/kits/reference_swatches")
def kit_reference_swatches():
    """Built-in chart swatches as sRGB, for rendering the holdout selector."""
    chart = builtin_colorchecker24()
    return jsonify({
        "chart_name": chart.name,
        "names": chart.patch_names,
        "srgb": chart.patch_srgb().round(4).tolist(),
    })


@app.route("/kits/<kit_name>")
def kit_detail(kit_name: str):
    try:
        kit = _load_kit(kit_name)
    except (KeyError, json.JSONDecodeError) as exc:
        return jsonify({"error": str(exc)}), 404
    manifest = kit["manifest"]
    charts = _kit_files(manifest, "chart_observation")
    scenes = _kit_files(manifest, "scene_observation")
    control = _kit_files(manifest, "negative_control_chart")
    chart_entries = [{
        "file": f["file"], "camera": f["camera"], "illumination": f["illumination"],
    } for f in charts]
    scene_entries = [{
        "file": f["file"], "camera": f["camera"], "illumination": f["illumination"],
        "blocks": f.get("blocks", []),
    } for f in scenes]
    control_entries = [{
        "file": f["file"], "camera": f["camera"], "illumination": f["illumination"],
    } for f in control]
    return jsonify({
        "name": kit_name,
        "manifest": {
            "kit_name": manifest.get("kit_name"),
            "kit_class": manifest.get("kit_class"),
            "simulation_note": manifest.get("simulation_note"),
            "provenance": manifest.get("provenance"),
            "acceptance": manifest.get("acceptance"),
            "chart": manifest.get("chart"),
        },
        "charts": chart_entries,
        "scenes": scene_entries,
        "control": control_entries,
        "chart_preview": _kit_png_data_uri(kit["dir"], chart_entries[0]["file"])
        if chart_entries else None,
    })


@app.route("/kits/<kit_name>/preview")
def kit_preview(kit_name: str):
    """Serve a kit image as a data URI (path is constrained to the manifest)."""
    try:
        kit = _load_kit(kit_name)
    except (KeyError, json.JSONDecodeError) as exc:
        return jsonify({"error": str(exc)}), 404
    rel = request.args.get("file", "")
    allowed = {f["file"] for f in kit["manifest"].get("files", [])}
    if rel not in allowed:
        return jsonify({"error": "file not part of this kit"}), 400
    return jsonify({"file": rel, "preview": _kit_png_data_uri(kit["dir"], rel)})


def _kit_chart_corners(params: dict) -> np.ndarray | None:
    corners = params.get("render_geometry", {}).get("chart_corners_xy")
    if not corners:
        return None
    return np.asarray(corners, dtype=np.float32)


@app.route("/kits/<kit_name>/cross_calibrate", methods=["POST"])
def kit_cross_calibrate(kit_name: str):
    """Fit a mapping on one camera, apply it to another, report residuals.

    The calibration kernel is the single source of truth: this route never
    hard-codes coefficients. It fits with ``calibrate`` (holdout patches
    excluded), then applies the fitted *colour matrix* under the target image's
    own white balance — the documented cross-camera contract.
    """
    try:
        kit = _load_kit(kit_name)
    except (KeyError, json.JSONDecodeError) as exc:
        return jsonify({"error": str(exc)}), 404
    params, manifest = kit["params"], kit["manifest"]
    kit_dir = kit["dir"]

    chart = builtin_colorchecker24()
    holdout = params.get("chart", {}).get("holdout_indices", [])
    neutral_idx = params.get("chart", {}).get("neutral_indices", [])

    src_file = request.form.get("source_chart", "")
    tgt_file = request.form.get("target_chart", "")
    allowed = {f["file"] for f in manifest.get("files", [])
               if f.get("role") == "chart_observation"}
    if src_file not in allowed or tgt_file not in allowed:
        return jsonify({"error": "source/target chart not in this kit"}), 400

    model = request.form.get("model", "matrix3")
    if model not in ("matrix3", "affine3x4", "poly3"):
        return jsonify({"error": f"unknown model {model}"}), 400

    # Patch selection: the UI may pass a subset of holdout indices to withhold
    # from the fit; those patches then serve as the independent check.
    selected_raw = request.form.get("holdout_indices", "").strip()
    if selected_raw:
        try:
            holdout = [int(x) for x in selected_raw.split(",") if x.strip()]
        except ValueError:
            return jsonify({"error": "holdout_indices must be integers"}), 400
    holdout = [i for i in holdout if 0 <= i < chart.n_patches]

    corners = _kit_chart_corners(params)
    mode = request.form.get("corners_mode", "kit")
    corners_raw = request.form.get("corners", "").strip()
    if mode == "manual" and corners_raw:
        try:
            corners = np.asarray(json.loads(corners_raw), dtype=np.float32)
        except (ValueError, TypeError):
            return jsonify({"error": "corners must be 4 [x,y] pairs"}), 400
    if corners is None:
        return jsonify({"error": "kit has no registered chart corners; supply manual corners"}), 422

    src_path = os.path.join(kit_dir, src_file)
    tgt_path = os.path.join(kit_dir, tgt_file)
    try:
        det_src = extract_manual(
            src_path, corners, chart.rows, chart.cols,
            chart.patch_names, chart.name,
        )
        det_tgt = extract_manual(
            tgt_path, corners, chart.rows, chart.cols,
            chart.patch_names, chart.name,
        )
    except FileNotFoundError as exc:
        return jsonify({"error": f"unreadable image: {exc}"}), 400

    try:
        result = calibrate(
            det_src, chart.lab_d50, chart.reference_space,
            model=model, holdout_indices=holdout,
        )
    except (InsufficientSamplesError, IllConditionedError) as exc:
        return jsonify({"error": str(exc), "refused": True}), 422

    # Apply the fitted matrix under the *target* image's own white balance.
    # The same guard rail as the fit path is used: if the target has too few
    # trustworthy neutrals it returns unity gains AND a warning, which is
    # surfaced below rather than hidden behind default values.
    ref_lin = reference_lab_d50_to_linear_srgb(chart.lab_d50)
    cam_lin = camera_encoded_to_linear(det_tgt.adopted_rgb)
    tgt_notes: list[str] = []
    tgt_gains = scene_white_gains(cam_lin, ref_lin, neutral_idx, tgt_notes)
    tgt_wb = cam_lin * tgt_gains[None, :]
    pred_lin = apply_coefficients(
        tgt_wb, np.asarray(result.coefficients), result.model
    )
    pred_lab = cs.srgb_to_lab(cs.linear_to_srgb(np.clip(pred_lin, 0.0, 1.0)))
    de = cs.delta_e_2000(pred_lab, chart.lab_d50)
    rgb_resid = pred_lin - ref_lin

    holdout_summary = result.holdout_summary
    target_rows = []
    for i, name in enumerate(chart.patch_names):
        target_rows.append({
            "index": i,
            "name": name,
            "role": ("holdout" if i in holdout else "fit+apply"),
            "adopted_srgb": det_tgt.adopted_rgb[i].round(5).tolist(),
            "reference_lab_d50": chart.lab_d50[i].round(3).tolist(),
            "predicted_lab_d50": pred_lab[i].round(3).tolist(),
            "delta_e2000": round(float(de[i]), 3),
            "residual_linear_rgb": rgb_resid[i].round(6).tolist(),
        })

    # Scene sample: apply the same mapping to the target camera's scene image
    # and measure each colour block against its reference.
    scene_file = tgt_file.replace("charts/chart_", "scenes/scene_")
    scene_evidence = _kit_scene_evidence(
        kit_dir, scene_file, manifest, result, tgt_gains
    )
    scene_preview = _kit_png_data_uri(kit_dir, scene_file)

    # Source and target chart previews side by side.
    job_id = uuid.uuid4().hex[:12]
    map_path = os.path.join(WORK_ROOT, job_id, "mapping.json")
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    result.to_json(map_path)
    JOBS[job_id] = {"mapping_path": map_path, "result": result, "chart": chart}

    payload = {
        "job_id": job_id,
        "kit_name": params.get("kit_name", kit_name),
        "simulated_response": True,
        "source": {"file": src_file, "preview": _kit_png_data_uri(kit_dir, src_file)},
        "target": {"file": tgt_file, "preview": _kit_png_data_uri(kit_dir, tgt_file)},
        "scene": {"file": scene_file, "preview": scene_preview,
                  "blocks": scene_evidence},
        "model": result.model,
        "holdout_indices": holdout,
        "holdout_patch_names": [chart.patch_names[i] for i in holdout],
        "source_fit": {
            "train": result.train_summary,
            "cv": result.validation_summary,
            "holdout": holdout_summary,
            "white_gains": result.white_gains,
            "coefficients": result.coefficients,
            "feature_names": result.feature_names,
            "detection": result.detection_summary,
            "warnings": result.warnings,
        },
        "target_warnings": tgt_notes,
        "target_white_gains": np.round(tgt_gains, 4).tolist(),
        "target_patch_residuals": target_rows,
        "target_delta_e2000": {
            "mean": round(float(np.mean(de)), 3),
            "max": round(float(np.max(de)), 3),
            "holdout_mean": (
                round(float(np.mean(de[holdout])), 3) if holdout else None
            ),
        },
        "controls": {
            "fit_patch_count": result.detection_summary["n_fitted"],
            "holdout_patch_count": len(holdout),
        },
        "download_mapping": url_for("download_mapping", job_id=job_id),
    }
    # Controlled failure: if the target is a different camera the residual is
    # large; surface that explicitly rather than returning a "nice" image.
    if payload["target_delta_e2000"]["mean"] > 8.0:
        payload["transfer_assessment"] = "mismatch"
    elif payload["target_delta_e2000"]["mean"] > 4.0:
        payload["transfer_assessment"] = "weak"
    else:
        payload["transfer_assessment"] = "consistent"
    # Same camera, different lighting should stay within the target bound. If
    # the numbers say otherwise, feed that back rather than trusting the label.
    same_camera = (
        _kit_camera_of(manifest, src_file) == _kit_camera_of(manifest, tgt_file)
    )
    payload["same_camera"] = same_camera
    if same_camera and payload["transfer_assessment"] != "consistent":
        payload["transfer_assessment"] = "mismatch"
        payload["target_warnings"] = list(payload["target_warnings"]) + [
            "the same camera under a different illumination did not meet the "
            "transfer bound — the fit or the target white balance is suspect"
        ]
    return jsonify(payload)


def _kit_camera_of(manifest: dict, rel_file: str) -> str:
    for rec in manifest.get("files", []):
        if rec["file"] == rel_file:
            return rec.get("camera", "")
    return ""


def _kit_scene_evidence(kit_dir, scene_file, manifest, result, gains):
    """Per-block ΔE2000 of the mapped scene, with exact pixel rectangles."""
    if not scene_file or not os.path.exists(os.path.join(kit_dir, scene_file)):
        return []
    blocks = None
    for rec in manifest.get("files", []):
        if rec["file"] == scene_file:
            blocks = rec.get("blocks")
            break
    if not blocks:
        return []
    import numpy as np
    bgr = cv2.imread(os.path.join(kit_dir, scene_file), cv2.IMREAD_COLOR)
    if bgr is None:
        return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    lin = camera_encoded_to_linear(rgb.reshape(-1, 3))
    lin_wb = lin * np.asarray(gains)[None, :]
    pred_lin = apply_coefficients(
        lin_wb, np.asarray(result.coefficients), result.model
    )
    out_lab = cs.srgb_to_lab(
        cs.linear_to_srgb(np.clip(pred_lin, 0.0, 1.0))
    ).reshape(rgb.shape[0], rgb.shape[1], 3)
    evidence = []
    for b in blocks:
        y0, y1, x0, x1 = b["y0"], b["y1"], b["x0"], b["x1"]
        ref_lab = cs.srgb_to_lab(np.array([b["reference_srgb"]]), "D50")[0]
        region = out_lab[y0:y1, x0:x1].reshape(-1, 3)
        block_de = cs.delta_e_2000(region, np.tile(ref_lab, (len(region), 1)))
        evidence.append({
            "block": b["name"],
            "reference_srgb": b["reference_srgb"],
            "pixel_rect_y0y1x0x1": [y0, y1, x0, x1],
            "delta_e2000_mean": round(float(np.mean(block_de)), 3),
            "delta_e2000_p95": round(float(np.percentile(block_de, 95)), 3),
        })
    return evidence


@app.route("/")
def index() -> str:
    return render_template("index.html", models=["matrix3", "affine3x4", "poly3"])


@app.route("/calibrate", methods=["POST"])
def calibrate_route():
    job_id = uuid.uuid4().hex[:12]
    chart_file = request.files.get("chart")
    if chart_file is None:
        return jsonify({"error": "no chart image uploaded"}), 400
    chart_path = _save_upload(chart_file, job_id)

    ref_file = request.files.get("reference")
    if ref_file and ref_file.filename:
        ref_path = _save_upload(ref_file, job_id)
        try:
            chart = load_chart_json(ref_path)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return jsonify({"error": f"invalid reference JSON: {exc}"}), 400
    else:
        chart = builtin_colorchecker24()

    model = request.form.get("model", "matrix3")
    if model not in ("matrix3", "affine3x4", "poly3"):
        return jsonify({"error": f"unknown model {model}"}), 400

    corners_raw = request.form.get("corners", "").strip()
    try:
        if corners_raw:
            corners = np.array(json.loads(corners_raw), dtype=np.float32)
            detection = extract_manual(
                chart_path, corners, chart.rows, chart.cols,
                chart.patch_names, chart.name,
            )
        else:
            detection = detect_chart(
                chart_path, chart.rows, chart.cols,
                chart.patch_names, chart.name,
            )
    except FileNotFoundError as exc:
        return jsonify({"error": f"unreadable image file: {exc}"}), 400
    except DetectionError as exc:
        return jsonify({"error": str(exc), "needs_manual_corners": True}), 422

    try:
        result = calibrate(
            detection, chart.lab_d50, chart.reference_space, model=model
        )
    except (InsufficientSamplesError, IllConditionedError) as exc:
        return jsonify({"error": str(exc), "refused": True}), 422

    map_path = os.path.join(WORK_ROOT, job_id, "mapping.json")
    result.to_json(map_path)
    report_path = os.path.join(WORK_ROOT, job_id, "report.html")
    write_html_report(result, chart, detection, report_path)

    # Encode the unwarped chart for an inline diagnostic preview.
    ok, buf = cv2.imencode(".jpg", detection.unwarped, [cv2.IMWRITE_JPEG_QUALITY, 80])
    preview_b64 = None
    if ok:
        import base64
        preview_b64 = "data:image/jpeg;base64," + base64.b64encode(
            buf.tobytes()
        ).decode("ascii")

    JOBS[job_id] = {
        "mapping_path": map_path,
        "report_path": report_path,
        "result": result,
        "chart": chart,
    }
    payload = {
        "job_id": job_id,
        "download_mapping": url_for("download_mapping", job_id=job_id),
        "download_report": url_for("download_report", job_id=job_id),
        "preview": preview_b64,
        "summary": {
            "model": result.model,
            "detection": result.detection_summary,
            "train": result.train_summary,
            "validation": result.validation_summary,
            "warnings": result.warnings,
            "white_gains": result.white_gains,
            "coefficients": result.coefficients,
            "feature_names": result.feature_names,
        },
        "patches": result.patch_records,
        "reference_srgb": chart.patch_srgb().round(4).tolist(),
    }
    return jsonify(payload)


@app.route("/jobs/<job_id>/mapping.json")
def download_mapping(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return Response("unknown job", status=404)
    return send_file(
        job["mapping_path"], as_attachment=True, download_name="color_mapping.json"
    )


@app.route("/jobs/<job_id>/report.html")
def download_report(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return Response("unknown job", status=404)
    return send_file(job["report_path"], as_attachment=False)


@app.route("/apply", methods=["POST"])
def apply_route():
    job_id = request.form.get("job_id", "")
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "calibrate before applying"}), 400
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "no product images uploaded"}), 400
    in_dir = os.path.join(WORK_ROOT, job_id, "batch_in")
    out_dir = os.path.join(WORK_ROOT, job_id, "batch_out")
    os.makedirs(in_dir, exist_ok=True)
    for f in files:
        f.save(os.path.join(in_dir, os.path.basename(f.filename)))
    report_path = os.path.join(WORK_ROOT, job_id, "batch_report.json")
    summary = apply_batch(
        in_dir, job["mapping_path"], out_dir, report_path
    )
    return jsonify(summary)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
