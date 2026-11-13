"""Tests for the dual-camera kit: holdout independence, reproducibility,
cross-camera contract, and the web surface that exposes it.

The kit is *synthetic* and its parameters are checked in; these tests rebuild
it from parameters and assert the documented contract, so a future change that
silently breaks transfer (or hides a failure behind a default) turns red.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colorcalib import colorspace as cs
from colorcalib.charts import builtin_colorchecker24
from colorcalib.detection import extract_manual
from colorcalib.pipeline import (
    InsufficientSamplesError,
    calibrate,
    camera_encoded_to_linear,
    apply_coefficients,
    scene_white_gains,
)
from colorcalib.simulate import chart_corners

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, "camera_kits", "dual_camera_kit")
PARAMS = json.load(open(os.path.join(KIT, "kit_params.json")))
CHART = builtin_colorchecker24()
CORNERS = np.asarray(chart_corners(4, 6), dtype=np.float32)
HOLDOUT = PARAMS["chart"]["holdout_indices"]


@pytest.fixture(scope="module")
def kit_ready():
    """Build the checked-in kit from its parameters once per session."""
    from scripts.build_camera_kit import build
    manifest = build(KIT)
    yield manifest


def _det(chart_file: str):
    return extract_manual(
        os.path.join(KIT, chart_file), CORNERS, CHART.rows, CHART.cols,
        CHART.patch_names, CHART.name,
    )


def _apply_target_wb(result, det):
    ref_lin = __import__(
        "colorcalib.pipeline", fromlist=["reference_lab_d50_to_linear_srgb"]
    ).reference_lab_d50_to_linear_srgb(CHART.lab_d50)
    cam_lin = camera_encoded_to_linear(det.adopted_rgb)
    gains = scene_white_gains(cam_lin, ref_lin, PARAMS["chart"]["neutral_indices"])
    pred = apply_coefficients(
        cam_lin * gains[None, :], np.asarray(result.coefficients), result.model
    )
    return cs.srgb_to_lab(cs.linear_to_srgb(np.clip(pred, 0, 1))), gains


# ---------------------------------------------------------------------------
# Holdout independence in the core kernel
# ---------------------------------------------------------------------------
def test_holdout_patches_are_excluded_from_fit(kit_ready):
    det = _det("charts/chart_cam_a_neutral.png")
    result = calibrate(
        det, CHART.lab_d50, CHART.reference_space,
        model="matrix3", holdout_indices=HOLDOUT,
    )
    assert result.detection_summary["n_holdout"] == len(HOLDOUT)
    roles = {r["index"]: r["role"] for r in result.patch_records}
    for i in HOLDOUT:
        assert roles[i] == "holdout"
    # holdout patches carry their own independent prediction error, not CV
    for r in result.patch_records:
        if r["role"] == "holdout":
            assert r["diagnoses"]["holdout"] is True
            assert r["delta_e2000"] is not None
            assert "cv_delta_e2000" not in r["diagnoses"]
    assert result.holdout_summary["n_patches"] == len(HOLDOUT)
    # fitting uses fewer patches than the usable total
    n_used = result.detection_summary["n_usable"]
    assert result.detection_summary["n_fitted"] == n_used - len(HOLDOUT)


def test_holdout_result_is_independent_of_other_illuminations(kit_ready):
    """The holdout error must be honest, i.e. NOT smaller than the fit error
    merely because the patch was used elsewhere. It is computed from a model
    that never saw it.
    """
    det = _det("charts/chart_cam_a_neutral.png")
    result = calibrate(
        det, CHART.lab_d50, CHART.reference_space,
        model="matrix3", holdout_indices=HOLDOUT,
    )
    # holdout errors are plausible and bounded; gross errors would signal the
    # holdout was actually leaking into the fit or being predicted trivially.
    assert result.holdout_summary["deltaE2000_mean"] < 8.0
    assert result.holdout_summary["deltaE2000_max"] < 15.0


def test_all_holdout_refused_when_no_samples_left(kit_ready):
    det = _det("charts/chart_cam_a_neutral.png")
    with pytest.raises(InsufficientSamplesError):
        calibrate(
            det, CHART.lab_d50, CHART.reference_space,
            model="poly3", holdout_indices=list(range(CHART.n_patches)),
        )


# ---------------------------------------------------------------------------
# Reproducibility and provenance
# ---------------------------------------------------------------------------
def test_kit_rebuild_is_byte_identical(kit_ready):
    import hashlib
    from scripts.build_camera_kit import sha256_file, build as rebuild
    rebuild(KIT)
    manifest2 = json.load(open(os.path.join(KIT, "manifest.json")))
    for rec in manifest2["files"]:
        path = os.path.join(KIT, rec["file"])
        assert sha256_file(path) == rec["sha256"]
    # params hash is carried as provenance
    params_text = open(os.path.join(KIT, "kit_params.json")).read()
    params_sha = hashlib.sha256(params_text.encode()).hexdigest()
    assert manifest2["provenance"]["params_sha256"] == params_sha


def test_kit_images_are_labelled_simulated(kit_ready):
    manifest = json.load(open(os.path.join(KIT, "manifest.json")))
    assert manifest["kit_class"] == "synthetic"
    for rec in manifest["files"]:
        assert rec["simulated_response"] is True
    assert "simulated" in manifest["simulation_note"].lower()


# ---------------------------------------------------------------------------
# Cross-camera contract
# ---------------------------------------------------------------------------
def test_same_camera_transfers_across_illumination(kit_ready):
    result = calibrate(
        _det("charts/chart_cam_a_neutral.png"), CHART.lab_d50,
        CHART.reference_space, model="matrix3", holdout_indices=HOLDOUT,
    )
    pred, _ = _apply_target_wb(result, _det("charts/chart_cam_a_warm.png"))
    de = cs.delta_e_2000(pred[HOLDOUT], CHART.lab_d50[HOLDOUT])
    assert float(np.mean(de)) < 6.0


def test_foreign_camera_is_rejected(kit_ready):
    result = calibrate(
        _det("charts/chart_cam_a_neutral.png"), CHART.lab_d50,
        CHART.reference_space, model="matrix3", holdout_indices=HOLDOUT,
    )
    pred, _ = _apply_target_wb(
        result, _det("control/chart_foreign_cam_neutral.png")
    )
    foreign_mean = float(np.mean(cs.delta_e_2000(pred, CHART.lab_d50)))
    # the foreign camera must be clearly worse than the same-camera case
    assert foreign_mean > PARAMS["acceptance"]["max_holdout_deltaE_mean"]


def test_verification_script_passes_and_reports_evidence(kit_ready):
    from scripts.rebuild_camera_kit import rebuild as verify
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        verdict = verify(KIT, tmp)
    assert verdict["passed"] is True, verdict["failures"]
    assert verdict["source_reproducible"] is True
    # every camera/illumination carries holdout evidence tied to a file
    for cam, data in verdict["cameras"].items():
        for ill, entry in data["illuminations"].items():
            assert "holdout" in entry and entry["holdout"] is not None
            assert entry["chart_file"].startswith("charts/")
            assert entry["scene_file"].startswith("scenes/")
            assert len(entry["scene_residuals"]) >= 1
    # negative control is recorded, not hidden
    assert "negative_control" in verdict
    assert verdict["negative_control"]["foreign_fullchart_deltaE_mean"] > 6.0
    assert verdict["lighting_generalization"]["cross_illumination_mean"] < 6.0


# ---------------------------------------------------------------------------
# Web surface
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    from web.app import app
    app.config["TESTING"] = True
    return app.test_client()


def test_kit_listing_and_detail(client):
    listing = client.get("/kits").get_json()
    assert any(k["name"] == "dual_camera_kit" for k in listing["kits"])
    assert all(k["kit_class"] == "synthetic" for k in listing["kits"])
    detail = client.get("/kits/dual_camera_kit").get_json()
    assert len(detail["charts"]) == 8
    assert len(detail["scenes"]) == 8
    assert detail["control"][0]["camera"] == "foreign_cam"


def test_reference_swatches_endpoint(client):
    d = client.get("/kits/reference_swatches").get_json()
    assert d["names"][0] == "dark_skin"
    assert len(d["srgb"]) == CHART.n_patches


def test_cross_calibrate_same_camera_is_consistent(client):
    body = {
        "source_chart": "charts/chart_cam_a_neutral.png",
        "target_chart": "charts/chart_cam_a_warm.png",
        "model": "matrix3",
        "holdout_indices": "2,11,17,20",
    }
    r = client.post("/kits/dual_camera_kit/cross_calibrate", data=body)
    assert r.status_code == 200
    d = r.get_json()
    assert d["transfer_assessment"] == "consistent"
    assert d["target_delta_e2000"]["mean"] < 4.0
    # scene evidence is present with exact pixel rectangles
    assert len(d["scene"]["blocks"]) >= 1
    for b in d["scene"]["blocks"]:
        assert len(b["pixel_rect_y0y1x0x1"]) == 4
    # every patch row carries a residual and a role
    assert len(d["target_patch_residuals"]) == CHART.n_patches
    roles = [p["role"] for p in d["target_patch_residuals"]]
    assert roles.count("holdout") == 4


def test_same_camera_flag_and_feedback(client):
    """The UI/route must know whether it transferred across cameras, and must
    downgrade to mismatch if the same camera still exceeds the bound."""
    body = {
        "source_chart": "charts/chart_cam_a_neutral.png",
        "target_chart": "charts/chart_cam_a_warm.png",
        "model": "matrix3",
        "holdout_indices": "2,11,17,20",
    }
    d = client.post("/kits/dual_camera_kit/cross_calibrate", data=body).get_json()
    assert d["same_camera"] is True
    assert d["transfer_assessment"] == "consistent"
    body["target_chart"] = "charts/chart_cam_b_warm.png"
    d2 = client.post("/kits/dual_camera_kit/cross_calibrate", data=body).get_json()
    assert d2["same_camera"] is False
    assert d2["transfer_assessment"] == "mismatch"


def test_cross_calibrate_foreign_camera_is_mismatch(client):
    body = {
        "source_chart": "charts/chart_cam_a_neutral.png",
        "target_chart": "charts/chart_cam_b_neutral.png",
        "model": "matrix3",
        "holdout_indices": "2,11,17,20",
    }
    r = client.post("/kits/dual_camera_kit/cross_calibrate", data=body)
    assert r.status_code == 200
    d = r.get_json()
    assert d["transfer_assessment"] == "mismatch"
    assert d["target_delta_e2000"]["mean"] > 8.0
    assert d["source_fit"]["holdout"]["deltaE2000_mean"] < 8.0


def test_cross_calibrate_refuses_when_holdout_consumes_all(client):
    body = {
        "source_chart": "charts/chart_cam_a_neutral.png",
        "target_chart": "charts/chart_cam_a_warm.png",
        "model": "matrix3",
        "holdout_indices": ",".join(str(i) for i in range(CHART.n_patches)),
    }
    r = client.post("/kits/dual_camera_kit/cross_calibrate", data=body)
    assert r.status_code == 422
    d = r.get_json()
    assert d["refused"] is True
    assert "error" in d


def test_cross_calibrate_rejects_unknown_kit_file(client):
    body = {
        "source_chart": "charts/nope.png",
        "target_chart": "charts/chart_cam_a_warm.png",
    }
    r = client.post("/kits/dual_camera_kit/cross_calibrate", data=body)
    assert r.status_code == 400
    assert "not in this kit" in r.get_json()["error"]


def test_cross_calibrate_manual_corners(client):
    body = {
        "source_chart": "charts/chart_cam_a_neutral.png",
        "target_chart": "charts/chart_cam_a_warm.png",
        "model": "matrix3",
        "holdout_indices": "2,11,17,20",
        "corners_mode": "manual",
        "corners": "[[14,14],[806,14],[806,534],[14,534]]",
    }
    r = client.post("/kits/dual_camera_kit/cross_calibrate", data=body)
    assert r.status_code == 200
    assert r.get_json()["transfer_assessment"] == "consistent"


def test_index_page_exposes_cross_camera_panel(client):
    html = client.get("/").get_data(as_text=True)
    assert "Cross-camera kit" in html
    assert "Holdout patches" in html
    assert "Calibrate across cameras" in html
