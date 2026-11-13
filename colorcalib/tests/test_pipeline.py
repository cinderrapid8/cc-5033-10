"""End-to-end and unit tests using synthetic data with a known camera matrix."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colorcalib.charts import builtin_colorchecker24
from colorcalib.colorspace import (
    chromatic_adaptation_matrix,
    delta_e_2000,
    linear_to_srgb,
    srgb_to_linear,
    srgb_rgb_to_xyz_d65,
    white_xyz,
    xyz_d65_to_srgb_rgb,
    xyz_to_lab,
)
from colorcalib.detection import detect_chart, extract_manual, DetectionError
from colorcalib.pipeline import (
    CalibrationResult,
    InsufficientSamplesError,
    calibrate,
    apply_mapping_to_rgb,
    camera_encoded_to_linear,
    reference_lab_d50_to_linear_srgb,
)
from colorcalib.apply import apply_batch

import pytest

from scripts.make_synthetic import (
    TRUE_CAMERA_MATRIX,
    render_chart_image,
    render_product_image,
)

CHART = builtin_colorchecker24()


@pytest.fixture(scope="module")
def sim_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("sim")
    render_chart_image(str(d / "nominal.png"))
    render_chart_image(
        str(d / "warm.png"), wb_shift=(1.22, 1.05, 0.82), exposure=1.05
    )
    render_chart_image(
        str(d / "clipped.png"), clip_patches={18, 15}, exposure=1.0
    )
    render_chart_image(
        str(d / "sheen.png"), sheen_patches={6, 12}, wb_shift=(1.1, 1.0, 0.9)
    )
    render_chart_image(
        str(d / "persp.png"), perspective=0.12, wb_shift=(1.15, 1.0, 0.88)
    )
    return d


# ---------------------------------------------------------------------------
def test_srgb_roundtrip():
    x = np.linspace(0, 1, 50)
    out = linear_to_srgb(srgb_to_linear(x))
    assert np.max(np.abs(out - x)) < 1e-12


def test_bradford_white_preservation():
    cat = chromatic_adaptation_matrix(white_xyz("D65"), white_xyz("D50"))
    mapped = cat @ white_xyz("D65")
    assert np.allclose(mapped, white_xyz("D50"), atol=1e-10)


def test_rgb_xyz_roundtrip_and_d50_white():
    lin = np.array([[0.5, 0.3, 0.7]])
    xyz = srgb_rgb_to_xyz_d65(lin)
    back = xyz_d65_to_srgb_rgb(xyz)
    assert np.allclose(back, lin, atol=1e-12)
    # D65 white in sRGB must give the D65 XYZ white point.
    assert np.allclose(srgb_rgb_to_xyz_d65(np.ones((1, 3)))[0],
                       white_xyz("D65"), atol=1e-10)


def test_srgb_xyz_matches_iec_standard():
    from colorcalib.colorspace import SRGB_TO_XYZ_D65
    expected = np.array(
        [[0.4123908, 0.3575843, 0.1804808],
         [0.2126390, 0.7151687, 0.0721923],
         [0.0193308, 0.1191948, 0.9505322]]
    )
    assert np.allclose(SRGB_TO_XYZ_D65, expected, atol=1e-6)
    # A pure encoded red must stay on the red primary axis (no cross channel).
    lin_red = srgb_to_linear(np.array([[1.0, 0.0, 0.0]]))
    xyz_red = srgb_rgb_to_xyz_d65(lin_red)[0]
    assert xyz_red[1] > 0 and xyz_red[0] > xyz_red[1] > xyz_red[2]


def test_deltae_known_values():
    # Identical colors -> 0; a known textbook pair (Sharma dataset 1).
    lab1 = np.array([[50.0, 2.6772, -79.7751]])
    lab2 = np.array([[50.0, 0.0, -82.7485]])
    de = delta_e_2000(lab1, lab2)
    assert abs(de[0] - 2.0425) < 0.01
    assert delta_e_2000(lab1, lab1)[0] < 1e-12


def test_reference_frame_is_d50_lab_consistent():
    lin = reference_lab_d50_to_linear_srgb(CHART.lab_d50)
    # L*=96.54 white patch maps to a near-white neutral (~0.914 linear),
    # not to 1.0: the reference is a real measured patch, not a perfect white.
    white = lin[CHART.patch_names.index("white")]
    assert np.ptp(white) < 0.005  # neutral: R≈G≈B
    assert 0.85 < white.mean() < 0.96
    # black patch stays dark and neutral in the same linear frame.
    black = lin[CHART.patch_names.index("black")]
    assert black.mean() < 0.04 and np.ptp(black) < 0.005


# ---------------------------------------------------------------------------
def test_auto_detection_nominal(sim_dir):
    det = detect_chart(
        str(sim_dir / "nominal.png"), CHART.rows, CHART.cols, CHART.patch_names
    )
    assert det.confidence > 0.5
    assert len(det.diagnoses) == 24
    assert det.usable_mask.sum() >= 22


def test_manual_corners_perspective(sim_dir):
    img_path = str(sim_dir / "persp.png")
    # Use the true outer quadrilateral from auto-detection as the manual
    # corner input: manual mode must reproduce the same quality, which is the
    # documented fallback when automatic detection fails on cluttered shots.
    auto = detect_chart(img_path, CHART.rows, CHART.cols, CHART.patch_names)
    corners = auto.corners
    det = extract_manual(img_path, corners, CHART.rows, CHART.cols,
                         CHART.patch_names)
    assert det.detection_method == "manual"
    assert det.usable_mask.sum() >= 18
    # manual and automatic sampling of the same quad agree closely
    diff = np.abs(det.adopted_rgb - auto.adopted_rgb).max()
    assert diff < 0.02


def test_clipped_patches_rejected(sim_dir):
    det = detect_chart(
        str(sim_dir / "clipped.png"), CHART.rows, CHART.cols, CHART.patch_names
    )
    for idx in (18, 15):
        assert det.diagnoses[idx].clipped
        assert not det.diagnoses[idx].usable
        assert "channel_clipped" in det.diagnoses[idx].reject_reasons


def test_sheen_patches_rejected(sim_dir):
    det = detect_chart(
        str(sim_dir / "sheen.png"), CHART.rows, CHART.cols, CHART.patch_names
    )
    rejected = {i for i, d in enumerate(det.diagnoses) if not d.usable}
    assert {6, 12} & rejected


# ---------------------------------------------------------------------------
def _fit(sim_dir, fname, model="matrix3", **kw):
    det = detect_chart(
        str(sim_dir / fname), CHART.rows, CHART.cols, CHART.patch_names
    )
    return calibrate(det, CHART.lab_d50, CHART.reference_space,
                     model=model, **kw), det


def test_matrix_fit_recovers_known_transform(sim_dir):
    result, det = _fit(sim_dir, "nominal.png", "matrix3")
    # Fitted chain = inverse(camera matrix) up to WB diagonal; reconstruct the
    # effective forward camera mapping and compare against TRUE_CAMERA_MATRIX.
    fit_coef = np.asarray(result.coefficients)
    gains = np.asarray(result.white_gains)
    forward = np.linalg.inv(fit_coef @ np.diag(gains))
    forward /= forward[1, 1]  # scale ambiguity on the green channel
    truth = TRUE_CAMERA_MATRIX / TRUE_CAMERA_MATRIX[1, 1]
    assert np.allclose(forward, truth, atol=0.06), (forward, truth)
    assert result.validation_summary["deltaE2000_mean"] < 2.0
    assert result.train_summary["deltaE2000_mean"] < 1.5

def test_train_validation_disjoint(sim_dir):
    result, _ = _fit(sim_dir, "nominal.png")
    roles = {r["role"] for r in result.patch_records if r["role"] != "excluded"}
    assert "train+cv" in roles
    # every usable patch carries both a fit residual and an honest CV error
    cv_de = [r["diagnoses"]["cv_delta_e2000"]
             for r in result.patch_records if r["role"] != "excluded"]
    assert len(cv_de) >= 20
    fit_de = [r["delta_e2000"]
              for r in result.patch_records if r["role"] != "excluded"]
    # independent cross-validated errors stay in a credible range
    assert np.mean(cv_de) < 3.0
    assert np.mean(fit_de) <= np.mean(cv_de) + 0.5
    assert result.validation_summary["method"].endswith("cross-validation")


def test_warm_white_balance_is_undone(sim_dir):
    result, det = _fit(sim_dir, "warm.png")
    gains = np.asarray(result.white_gains)
    # warm shift raised red; the fitted red gain must compensate downward-ish
    # relative behavior, and final errors must remain small.
    assert result.validation_summary["deltaE2000_mean"] < 2.5
    assert gains[0] != 1.0 or gains[2] != 1.0


def test_affine_and_poly_models(sim_dir):
    for model in ("affine3x4", "poly3"):
        result, _ = _fit(sim_dir, "nominal.png", model)
        assert len(result.coefficients) == {"affine3x4": 4, "poly3": 10}[model]
        assert result.validation_summary["deltaE2000_mean"] < 3.0


def test_insufficient_samples_refused(sim_dir):
    det = detect_chart(
        str(sim_dir / "nominal.png"), CHART.rows, CHART.cols, CHART.patch_names
    )
    # Force-reject most patches; usable_mask derives from each diagnosis flag.
    for d in det.diagnoses[6:]:
        d.rejected = True
        d.reject_reasons.append("forced")
    assert det.usable_mask.sum() == 6
    with pytest.raises(InsufficientSamplesError):
        calibrate(det, CHART.lab_d50, CHART.reference_space, model="affine3x4")
    # matrix3 can still fit exactly 6, but poly3 must refuse.
    with pytest.raises(InsufficientSamplesError):
        calibrate(det, CHART.lab_d50, CHART.reference_space, model="poly3")


def test_irls_downweights_outlier(sim_dir):
    det = detect_chart(
        str(sim_dir / "nominal.png"), CHART.rows, CHART.cols, CHART.patch_names
    )
    # Catastrophic patch corruption: replace the measured color with an
    # unrelated hue, as happens when a patch is mis-identified or contaminated.
    target = 4
    det.diagnoses[target].rgb_median = np.array([0.05, 0.85, 0.1])
    result_robust = calibrate(det, CHART.lab_d50, CHART.reference_space,
                              model="matrix3")
    weight_map = {r["index"]: r["irls_weight"]
                  for r in result_robust.patch_records
                  if r["role"] != "excluded"}
    # the catastrophic patch is hard-rejected from the production fit
    assert weight_map[target] < 0.05

    # The honest CV error of the corrupted patch itself is large (unavoidable:
    # the measured color is simply wrong). What robustness must guarantee is
    # that CLEAN patches keep the same accuracy as an uncontaminated fit.
    clean_det = detect_chart(
        str(sim_dir / "nominal.png"), CHART.rows, CHART.cols, CHART.patch_names
    )
    clean_res = calibrate(clean_det, CHART.lab_d50, CHART.reference_space,
                          model="matrix3")
    clean_cv = {r["index"]: r["diagnoses"]["cv_delta_e2000"]
                for r in clean_res.patch_records if r["role"] != "excluded"}
    dirty_cv = {r["index"]: r["diagnoses"]["cv_delta_e2000"]
                for r in result_robust.patch_records if r["role"] != "excluded"}
    others = [i for i in clean_cv if i != target]
    clean_mean = np.mean([clean_cv[i] for i in others])
    dirty_mean = np.mean([dirty_cv[i] for i in others])
    assert abs(dirty_mean - clean_mean) < 1.0
    # production coefficients on clean data are barely moved by the bad patch
    c_dirty = np.array(result_robust.coefficients)
    c_clean = np.array(clean_res.coefficients)
    assert np.abs(c_dirty - c_clean).max() < 0.05


def test_mapping_roundtrip_and_extrapolation_report(sim_dir, tmp_path):
    result, det = _fit(sim_dir, "nominal.png")
    mp = tmp_path / "map.json"
    result.to_json(str(mp))
    render_product_image(str(tmp_path / "prod.png"), (1.0, 1.0, 1.0), 1.0)
    import cv2
    bgr = cv2.imread(str(tmp_path / "prod.png"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mapping = json.loads(mp.read_text())
    out, report = apply_mapping_to_rgb(rgb, mapping)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert 0.0 <= report["fraction_above_calibration_range"] <= 1.0

    # Perceptual check: for each product band, the calibrated image must be
    # closer (mean ΔE2000) to the intended reference color than the raw camera
    # output. The reference colors are the exact bases fed to the simulator.
    from colorcalib.colorspace import srgb_to_lab, delta_e_2000
    bases = [(0, 80, (0.85, 0.2, 0.15)),
             (80, 160, (0.2, 0.55, 0.8)),
             (160, 240, (0.9, 0.85, 0.4))]
    raw_lab = srgb_to_lab(rgb.reshape(-1, 3) / 255.0).reshape(240, 320, 3)
    out_lab = srgb_to_lab(out.reshape(-1, 3) / 255.0).reshape(240, 320, 3)
    for y0, y1, base in bases:
        ref_lab = srgb_to_lab(np.array([base]))[0]
        raw_de = delta_e_2000(raw_lab[y0:y1].reshape(-1, 3),
                              np.tile(ref_lab, ((y1 - y0) * 320, 1))).mean()
        new_de = delta_e_2000(out_lab[y0:y1].reshape(-1, 3),
                              np.tile(ref_lab, ((y1 - y0) * 320, 1))).mean()
        assert new_de < raw_de, (base, raw_de, new_de)


def test_batch_application(sim_dir, tmp_path):
    result, _ = _fit(sim_dir, "nominal.png")
    mp = tmp_path / "map.json"
    result.to_json(str(mp))
    indir = tmp_path / "in"
    outdir = tmp_path / "out"
    indir.mkdir()
    render_product_image(str(indir / "a.png"), (1.2, 1.0, 0.85), 1.1)
    render_product_image(str(indir / "b.png"), (0.95, 1.0, 1.1), 0.9)
    summary = apply_batch(str(indir), str(mp), str(outdir),
                          str(tmp_path / "batch.json"))
    assert summary["n_images"] == 2
    assert all(r["status"] == "ok" for r in summary["images"])
    assert len(list(outdir.iterdir())) == 2


def test_patch_records_explain_every_value(sim_dir):
    result, _ = _fit(sim_dir, "nominal.png")
    for rec in result.patch_records:
        assert len(rec["adopted_srgb"]) == 3
        assert len(rec["reference_lab_d50"]) == 3
        assert rec["role"] in {"train", "validation", "train+cv", "excluded"}
        if rec["role"] != "excluded":
            assert rec["delta_e2000"] is not None
            assert rec["residual_linear_rgb"] is not None
            assert "cv_delta_e2000" in rec["diagnoses"]
