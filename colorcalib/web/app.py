"""Flask web interface for chart calibration and batch application.

Run with::

    python -m web.app --host 127.0.0.1 --port 5000

The interface deliberately exposes the numerical evidence (per-patch adopted
values, fit and cross-validated ΔE2000, residuals, diagnostics) and the actual
mapping coefficients. It offers no saturation/hue sliders: calibration is a
fitted color transform with stated error, not subjective grading.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import uuid

import cv2
import numpy as np
from flask import (
    Flask, Response, jsonify, render_template, request, send_file, url_for,
)

from colorcalib.charts import builtin_colorchecker24, load_chart_json
from colorcalib.detection import (
    DetectionError, detect_chart, extract_manual,
)
from colorcalib.pipeline import (
    IllConditionedError, InsufficientSamplesError, calibrate,
)
from colorcalib.apply import apply_batch
from colorcalib.report import write_html_report

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

# In-process job store (single-user local tool). Each job holds uploaded files
# and the calibration result keyed by an opaque id.
JOBS: dict[str, dict] = {}
WORK_ROOT = tempfile.mkdtemp(prefix="colorcalib_web_")


def _save_upload(file_storage, subdir: str) -> str:
    job_dir = os.path.join(WORK_ROOT, subdir)
    os.makedirs(job_dir, exist_ok=True)
    name = file_storage.filename or "upload.png"
    safe = os.path.basename(name).replace(os.sep, "_")
    path = os.path.join(job_dir, safe)
    file_storage.save(path)
    return path


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
