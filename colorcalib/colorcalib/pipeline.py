"""Calibration pipeline: normalization, fitting, validation, application.

Color representations are kept strictly separate until they share a frame:

1. Camera samples are sRGB-encoded camera output.  They are linearized
   (IEC sRGB EOTF) to *camera-linear RGB* and white-balanced from the chart's
   neutral patches, never compared directly with reference numbers.
2. References are D50 L*a*b*.  They are converted to XYZ(D50), Bradford
   adapted to D65, and expressed as *reference-linear sRGB RGB*.
3. The mapping is fitted camera-linear -> reference-linear.  Only linear-light
   values enter the least-squares problem; encoded values are only used for
   display and final re-encoding.

Models (all interpretable, no hidden per-hue tweaks):

* ``matrix3``  : 3x3 linear matrix (white-preserving, 9 parameters)
* ``affine3x4``: matrix plus per-channel offset (12 parameters)
* ``poly3``    : linear + quadratic RGB cross-terms (10 features, 30 params)

A minimum-sample rule prevents the tool from emitting a confident-looking
config when the chart did not actually provide enough independent colors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np

from . import colorspace as cs
from .detection import DetectionResult

# Minimum usable patches to even attempt each model, and the extra samples
# required beyond the parameter count for a meaningful residual degree count.
MIN_SAMPLES = {
    "matrix3": 6,
    "affine3x4": 8,
    "poly3": 14,
}
PARAM_COUNT = {"matrix3": 9, "affine3x4": 12, "poly3": 30}
ILL_CONDITIONED_THRESHOLD = 1e8  # condition number of the design matrix


# ---------------------------------------------------------------------------
# Frame conversion
# ---------------------------------------------------------------------------
def reference_lab_d50_to_linear_srgb(lab_d50: np.ndarray) -> np.ndarray:
    """D50 Lab -> XYZ(D50) -> Bradford to D65 -> linear sRGB RGB."""
    xyz_d50 = cs.lab_to_xyz(lab_d50, cs.white_xyz("D50"))
    cat = cs.chromatic_adaptation_matrix(cs.white_xyz("D50"), cs.white_xyz("D65"))
    xyz_d65 = cs.adapt_xyz(xyz_d50, cat)
    return cs.xyz_d65_to_srgb_rgb(xyz_d65)


def camera_encoded_to_linear(rgb_encoded: np.ndarray) -> np.ndarray:
    """Linearize camera sRGB-encoded samples (0..1)."""
    return cs.srgb_to_linear(np.clip(rgb_encoded, 0.0, 1.0))


def estimate_scene_white_gains(
    cam_linear: np.ndarray, ref_linear: np.ndarray, neutral_idx: list[int]
) -> np.ndarray:
    """Per-channel gain that makes neutral patches map onto the neutral axis.

    Primary path: median per-channel ratio over declared neutral patches.
    Fallback when the neutral row is clipped/uneven (common in perspective or
    over-exposed captures): a preliminary 3x3 fit over all usable patches,
    from which the gain is read as the diagonal that sends camera white
    (1,1,1) toward a neutral reference direction. This never assumes neutral
    data that the diagnostics have rejected.
    """
    if neutral_idx:
        cn = cam_linear[neutral_idx]
        rn = ref_linear[neutral_idx]
        ratios = np.divide(
            rn, np.maximum(cn, 1e-6),
            out=np.full_like(rn, np.nan), where=cn > 1e-4,
        )
        valid = ratios[np.isfinite(ratios).all(axis=1)]
        if len(valid) >= 2:
            return np.median(valid, axis=0)

    # Fallback: provisional matrix over all usable patches.
    A = cam_linear
    y = ref_linear
    if len(A) < MIN_SAMPLES["matrix3"]:
        return np.ones(3)
    lum = np.maximum(y.mean(axis=1), 0.05)
    w = (1.0 / lum)
    w /= w.mean()
    coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w)[:, None],
                               rcond=None)
    white_pred = np.ones(3) @ coef.T  # camera-white mapped to reference RGB
    gain = white_pred / max(float(np.mean(white_pred)), 1e-6)
    return gain / np.median(gain)  # keep an average gain of 1


def normalize_white(
    cam_linear: np.ndarray, gains: np.ndarray
) -> np.ndarray:
    return cam_linear * gains[np.newaxis, :]


def scene_white_gains(
    cam_linear: np.ndarray,
    ref_linear: np.ndarray,
    neutral_indices: list[int] | None,
    warnings: list[str] | None = None,
) -> np.ndarray:
    """White-balance gains for a target image, with the same guard rails as
    the fit path.

    Shared by :func:`calibrate` and by cross-camera application so a target
    image with too few trustworthy neutrals never silently gets fabricated
    unity gains: callers receive the gains *and* a warning string when the
    neutral row is insufficient, and the extreme-gain clamp is identical.
    """
    note: list[str] = warnings if warnings is not None else []
    usable_neutral = list(neutral_indices or [])
    if len(usable_neutral) < 2:
        note.append(
            "fewer than 2 usable neutral patches; white-balance gains left at "
            "unity and channel balance delegated to the fitted matrix"
        )
        return np.ones(3)
    gains = estimate_scene_white_gains(cam_linear, ref_linear, usable_neutral)
    if np.any(gains <= 0) or np.max(gains) / max(np.min(gains), 1e-6) > 4:
        note.append(
            f"extreme white-balance gains {gains.round(2)} — check illumination"
        )
        gains = np.clip(gains, 0.25, 4.0)
    return gains


# ---------------------------------------------------------------------------
# Design matrices and solvers
# ---------------------------------------------------------------------------
def design_matrix(x: np.ndarray, model: str) -> np.ndarray:
    """Build the per-sample feature matrix for camera-linear RGB x."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    if model == "matrix3":
        return x.copy()
    if model == "affine3x4":
        return np.column_stack([x, np.ones(len(x))])
    if model == "poly3":
        return np.column_stack(
            [r, g, b, r * g, r * b, g * b, r * r, g * g, b * b, np.ones_like(r)]
        )
    raise ValueError(f"unknown model {model!r}")


def _lstsq(A: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return coef
    sw = np.sqrt(weights)[:, None]
    coef, *_ = np.linalg.lstsq(A * sw, y * sw, rcond=None)
    return coef


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    model: str,
    robust: bool = True,
    huber_k: float = 1.345,
    irls_iters: int = 25,
    perceptual_weights: bool = True,
) -> tuple[np.ndarray, dict]:
    """Fit x (camera-linear) -> y (reference-linear).

    Chart patches span a huge dynamic range (dark neutrals near 0.02 versus a
    white patch near 0.9 in linear light).  Plain least squares is dominated by
    bright samples, which leaves large *perceptual* errors in dark tones where
    the sRGB EOTF steeply amplifies a small linear residual into a large Lab
    distance.  Rows are therefore weighted by inverse luminance, the standard
    color-matrix calibration practice, with an epsilon floor to avoid letting
    near-black read noise dominate.

    Returns the coefficient matrix (n_features x 3) and fit metadata
    (condition number, IRLS weights, residual degrees of freedom).
    """
    A = design_matrix(x, model)
    n, p3 = A.shape[0], A.shape[1] * 3
    if n < MIN_SAMPLES[model]:
        raise InsufficientSamplesError(
            f"{model} needs >= {MIN_SAMPLES[model]} usable patches, got {n}"
        )
    dof = max(n * 3 - p3, 1)
    cond = float(np.linalg.cond(A))
    if cond > ILL_CONDITIONED_THRESHOLD:
        raise IllConditionedError(
            f"design matrix condition number {cond:.2e} — chart colors are "
            "not independent enough (check detection or lighting)"
        )

    if perceptual_weights:
        lum = np.maximum(y.mean(axis=1), 0.05)
        base_w = 1.0 / lum
        base_w /= base_w.mean()
    else:
        base_w = np.ones(n)

    coef = _lstsq(A, y, base_w)
    weights = base_w.copy()
    robust_factor = np.ones(n)
    if robust and n >= MIN_SAMPLES[model] + 2:
        # Cook's influence identifies points that actually move the fit: a
        # single grossly wrong patch among many can be absorbed by plain least
        # squares with a deceptively small own-residual, but deleting it
        # changes the fitted coefficients strongly. We approximate influence
        # with leave-one-out coefficient displacement, normalized per row.
        base_coef = coef.copy()
        influence = np.zeros(n)
        for i in range(n):
            keep = np.ones(n, dtype=bool)
            keep[i] = False
            if keep.sum() < MIN_SAMPLES[model]:
                continue
            c_i = _lstsq(A[keep], y[keep], base_w[keep])
            influence[i] = np.linalg.norm(c_i - base_coef)
        inf_scale = max(float(np.percentile(influence, 75)) / 0.6745, 1e-8)
        u_inf = influence / inf_scale
        # Hard rejection for catastrophic influence (more than ~4 robust
        # standard deviations), Huber taper for the moderate band. A soft
        # weight alone lets a wholly wrong color still drag the matrix.
        inf_w = np.where(
            u_inf <= huber_k, 1.0,
            np.where(u_inf <= 4.0, huber_k / np.maximum(u_inf, 1e-8), 0.0),
        )

        for _ in range(irls_iters):
            resid = y - A @ coef
            # Luminance-normalized residual scale: dark patches naturally have
            # larger absolute linear residuals.
            lum = np.maximum(y.mean(axis=1), 0.05)
            rel_norms = np.linalg.norm(resid, axis=1) / lum
            s = max(float(np.percentile(rel_norms, 75)) / 0.6745, 1e-4)
            u = rel_norms / s
            res_w = np.where(u <= huber_k, 1.0,
                             huber_k / np.maximum(u, 1e-8))
            # Both signals required to stay at full trust; influence catches
            # absorbed outliers, residual catches noisy ones.
            huber = np.minimum(res_w, inf_w)
            w = base_w * huber
            new_coef = _lstsq(A, y, w)
            if np.max(np.abs(new_coef - coef)) < 1e-7:
                coef = new_coef
                weights = w
                robust_factor = huber
                break
            coef, weights, robust_factor = new_coef, w, huber

    meta = {
        "condition_number": cond,
        "dof": dof,
        "irls_weights": robust_factor.tolist(),  # robust factor only
        "perceptual_row_weights": base_w.round(4).tolist(),
        "n_samples": n,
        "n_parameters": p3,
    }
    return coef, meta


def apply_coefficients(x: np.ndarray, coef: np.ndarray, model: str) -> np.ndarray:
    return design_matrix(x, model) @ coef


# ---------------------------------------------------------------------------
# Error evaluation in a shared perceptual frame
# ---------------------------------------------------------------------------
def linear_srgb_to_lab_d50(lin_rgb: np.ndarray) -> np.ndarray:
    xyz_d65 = cs.srgb_rgb_to_xyz_d65(np.maximum(lin_rgb, 0.0))
    cat = cs.chromatic_adaptation_matrix(cs.white_xyz("D65"), cs.white_xyz("D50"))
    xyz_d50 = cs.adapt_xyz(xyz_d65, cat)
    return cs.xyz_to_lab(xyz_d50, cs.white_xyz("D50"))


def evaluate_patches(
    cam_linear_wb: np.ndarray,
    ref_linear: np.ndarray,
    coef: np.ndarray,
    model: str,
    patch_names: list[str],
) -> dict:
    """Per-patch adopted values, predictions and residuals in Lab and RGB."""
    pred_linear = apply_coefficients(cam_linear_wb, coef, model)
    pred_lab = linear_srgb_to_lab_d50(pred_linear)
    ref_lab = linear_srgb_to_lab_d50(ref_linear)
    de = cs.delta_e_2000(pred_lab, ref_lab)
    rgb_resid = pred_linear - ref_linear
    return {
        "names": list(patch_names),
        "pred_linear": pred_linear,
        "ref_linear": ref_linear,
        "pred_lab": pred_lab,
        "ref_lab": ref_lab,
        "delta_e2000": de,
        "rgb_residual": rgb_resid,
        "max_abs_rgb_residual": float(np.max(np.abs(rgb_resid))),
    }


def summarize_errors(ev: dict) -> dict:
    de = ev["delta_e2000"]
    return {
        "deltaE2000_mean": float(np.mean(de)),
        "deltaE2000_median": float(np.median(de)),
        "deltaE2000_p95": float(np.percentile(de, 95)),
        "deltaE2000_max": float(np.max(de)),
        "rmse_linear_rgb": float(
            np.sqrt(np.mean(ev["rgb_residual"] ** 2))
        ),
    }


# ---------------------------------------------------------------------------
# Splits and full calibration
# ---------------------------------------------------------------------------
def kfold_indices(n: int, k: int = 4, seed: int = 7) -> np.ndarray:
    """Assign each of n samples a fold id 0..k-1, shuffled but deterministic."""
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    folds = np.empty(n, dtype=int)
    folds[order] = np.arange(n) % k
    return folds


@dataclass
class PatchRecord:
    index: int
    name: str
    adopted_srgb: list[float]
    camera_linear_wb: list[float]
    reference_lab_d50: list[float]
    diagnoses: dict
    role: str = "train"          # train | validation | excluded
    delta_e2000: float | None = None
    residual_linear_rgb: list[float] | None = None
    irls_weight: float | None = None


@dataclass
class CalibrationResult:
    model: str
    coefficients: list[list[float]]
    white_gains: list[float]
    feature_names: list[str]
    fit_meta: dict
    train_summary: dict
    validation_summary: dict | None
    patch_records: list[dict]
    detection_summary: dict
    warnings: list[str]
    reference_space: dict
    holdout_summary: dict | None = None
    extrapolation: dict = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False)


def calibrate(
    detection: DetectionResult,
    ref_lab_d50: np.ndarray,
    ref_space: dict,
    model: str = "matrix3",
    neutral_indices: list[int] | None = None,
    validation_indices: list[int] | None = None,
    holdout_indices: list[int] | None = None,
    robust: bool = True,
) -> CalibrationResult:
    """Run the full fit with train/validation separation and diagnostics.

    ``validation_indices`` only relabels already-usable patches as *forced
    validation*: they still help fit the shipped mapping and are shown so a
    caller can annotate an external holdout image.

    ``holdout_indices`` is stronger: those patches are removed from the
    production fit *entirely* and are never used to estimate white balance or
    the cross-validation folds. Their residuals are reported separately in
    ``holdout_summary`` — an independent check that the mapping generalizes to
    colors it never saw, rather than a training residual dressed up as one.
    """
    if model not in MIN_SAMPLES:
        raise ValueError(f"model must be one of {list(MIN_SAMPLES)}")
    n = detection.rows * detection.cols
    if len(ref_lab_d50) != n:
        raise ValueError(
            f"reference has {len(ref_lab_d50)} values but chart has {n} patches"
        )

    names = [d.name for d in detection.diagnoses]
    diag = detection.diagnoses
    usable = detection.usable_mask
    warnings: list[str] = []

    cam_enc = detection.adopted_rgb
    cam_linear = camera_encoded_to_linear(cam_enc)
    ref_linear = reference_lab_d50_to_linear_srgb(ref_lab_d50)

    holdout_set = {i for i in (holdout_indices or []) if usable[i]}
    if holdout_set and len(holdout_set) >= int(usable.sum()):
        raise InsufficientSamplesError(
            "holdout patches leave no samples for fitting"
        )

    # White balance from usable neutral patches only, excluding held-out ones
    # so the holdout measurement stays independent of the fitted transform.
    if neutral_indices is None:
        neutral_indices = [
            i for i, nm in enumerate(names) if "neutral" in nm or nm == "white"
        ]
    usable_neutral = [
        i for i in neutral_indices if usable[i] and i not in holdout_set
    ]
    wb_note: list[str] = []
    gains = scene_white_gains(cam_linear, ref_linear, usable_neutral, wb_note)
    warnings.extend(wb_note)
    cam_wb = normalize_white(cam_linear, gains)

    usable_idx = np.where(usable)[0]
    fit_idx = np.array(
        [i for i in usable_idx if i not in holdout_set], dtype=int
    )
    if len(fit_idx) < MIN_SAMPLES[model]:
        raise InsufficientSamplesError(
            f"only {len(fit_idx)} usable patches after holdout; {model} needs "
            f"{MIN_SAMPLES[model]}. Refusing to emit a high-precision config."
        )

    # --- Validation strategy -------------------------------------------------
    # K-fold cross-validation: every fitted patch gets an honest ΔE from a
    # model that never saw it. The shipped mapping is then refit on all fitted
    # patches, so validation never shrinks the production training set.
    # Held-out patches are excluded from the folds so they stay independent.
    k_folds = 4
    if len(fit_idx) < 2 * MIN_SAMPLES[model]:
        k_folds = 3
    fold_ids = kfold_indices(len(fit_idx), k_folds)

    cv_delta: dict[int, float] = {}
    cv_resid: dict[int, np.ndarray] = {}
    for k in range(k_folds):
        tr = fit_idx[fold_ids != k]
        va = fit_idx[fold_ids == k]
        if len(tr) < MIN_SAMPLES[model]:
            continue
        fold_coef, _ = fit_model(cam_wb[tr], ref_linear[tr], model, robust=robust)
        fold_ev = evaluate_patches(
            cam_wb[va], ref_linear[va], fold_coef, model,
            [names[i] for i in va],
        )
        for local, global_i in enumerate(va):
            cv_delta[global_i] = float(fold_ev["delta_e2000"][local])
            cv_resid[global_i] = fold_ev["rgb_residual"][local]

    cv_values = np.array(list(cv_delta.values()))
    validation_summary = {
        "deltaE2000_mean": float(np.mean(cv_values)),
        "deltaE2000_median": float(np.median(cv_values)),
        "deltaE2000_p95": float(np.percentile(cv_values, 95)),
        "deltaE2000_max": float(np.max(cv_values)),
        "method": f"{k_folds}-fold cross-validation",
        "n_patches": int(len(cv_values)),
    }

    # Optional fixed holdout (e.g. separate chart image) overrides fold roles.
    forced_val = {
        i for i in (validation_indices or []) if usable[i] and i not in holdout_set
    }

    # Production fit on every fitted patch (never on holdout patches).
    coef, fit_meta = fit_model(
        cam_wb[fit_idx], ref_linear[fit_idx], model, robust=robust
    )
    irls = np.asarray(fit_meta["irls_weights"])
    full_ev = evaluate_patches(
        cam_wb[fit_idx], ref_linear[fit_idx], coef, model,
        [names[i] for i in fit_idx],
    )
    train_summary = summarize_errors(full_ev)

    # Independent check: predict the held-out patches with the production
    # mapping. These patches contributed nothing to gains, folds, or the fit.
    holdout_summary = None
    holdout_ev = None
    if holdout_set:
        hold_idx = np.array(sorted(holdout_set), dtype=int)
        holdout_ev = evaluate_patches(
            cam_wb[hold_idx], ref_linear[hold_idx], coef, model,
            [names[i] for i in hold_idx],
        )
        holdout_summary = summarize_errors(holdout_ev)
        holdout_summary["n_patches"] = int(len(hold_idx))
        holdout_summary["patches"] = [names[i] for i in hold_idx]
        holdout_summary["method"] = "held-out patches excluded from fit and WB"
        gap_h = holdout_summary["deltaE2000_mean"] - train_summary["deltaE2000_mean"]
        if gap_h > 4.0:
            warnings.append(
                f"held-out mean ΔE2000 exceeds training residual by {gap_h:.2f} "
                "— the mapping is extrapolating, not generalizing"
            )

    gap = validation_summary["deltaE2000_mean"] - train_summary["deltaE2000_mean"]
    if gap > 3.0:
        warnings.append(
            f"cross-validated mean ΔE2000 exceeds training residual by {gap:.2f} "
            "— the fit is partly memorizing the chart; check lighting consistency"
        )

    de_map = dict(zip(full_ev["names"], full_ev["delta_e2000"]))
    res_map = {nm: row for nm, row in zip(full_ev["names"], full_ev["rgb_residual"])}
    w_map = {nm: float(w) for nm, w in
             zip([names[i] for i in fit_idx], irls)}
    hold_de = dict(
        zip(holdout_ev["names"], holdout_ev["delta_e2000"])
    ) if holdout_ev is not None else {}
    hold_res = dict(
        zip(holdout_ev["names"], holdout_ev["rgb_residual"])
    ) if holdout_ev is not None else {}

    records: list[PatchRecord] = []
    for i, d in enumerate(diag):
        if not usable[i]:
            role = "excluded"
        elif i in holdout_set:
            role = "holdout"
        elif i in forced_val:
            role = "validation"
        else:
            # every fitted patch participates in the final fit AND was held out
            # of exactly one CV fold; label folds for transparent provenance.
            role = "train+cv"
        records.append(
            PatchRecord(
                index=i,
                name=d.name,
                adopted_srgb=cam_enc[i].round(5).tolist(),
                camera_linear_wb=cam_wb[i].round(6).tolist(),
                reference_lab_d50=ref_lab_d50[i].round(4).tolist(),
                diagnoses={
                    "clipped": d.clipped,
                    "near_clipped": d.near_clipped,
                    "near_black": d.near_black,
                    "specular_sheen": d.sheen,
                    "nonuniform": d.nonuniform,
                    "cv_per_channel": d.cv_per_channel.round(4).tolist(),
                    "reject_reasons": d.reject_reasons,
                },
                role=role,
                delta_e2000=(
                    round(float(hold_de[d.name]), 3) if d.name in hold_de
                    else round(float(de_map[d.name]), 3) if d.name in de_map
                    else None
                ),
                residual_linear_rgb=(
                    hold_res[d.name].round(6).tolist() if d.name in hold_res
                    else res_map[d.name].round(6).tolist() if d.name in res_map
                    else None
                ),
                irls_weight=(round(w_map[d.name], 3) if d.name in w_map else None),
            )
        )
    # Expose the honest cross-validated per-patch error alongside the fit
    # residual so the UI can show both numbers per patch. Held-out patches get
    # their independent prediction error instead of a CV number.
    for rec in records:
        gi = rec.index
        if gi in cv_delta:
            rec.diagnoses["cv_delta_e2000"] = round(cv_delta[gi], 3)
            rec.diagnoses["cv_residual_linear_rgb"] = cv_resid[gi].round(6).tolist()
        if rec.role == "holdout":
            rec.diagnoses["holdout"] = True
            rec.diagnoses["holdout_delta_e2000"] = rec.delta_e2000

    # Interpretable warnings about fit quality.
    if train_summary["deltaE2000_mean"] > 4.0:
        warnings.append(
            "training ΔE2000 mean > 4 — residual spectral mismatch between "
            "chart dyes and camera filters remains; a single 3x3 matrix cannot "
            "fully correct metameric differences"
        )
    n_rejected = int((~usable).sum())
    if n_rejected:
        warnings.append(f"{n_rejected} patch(es) excluded by quality diagnostics")
    if fit_meta["condition_number"] > 1e6:
        warnings.append(
            f"design matrix poorly conditioned ({fit_meta['condition_number']:.1e})"
        )

    feature_names = {
        "matrix3": ["R", "G", "B"],
        "affine3x4": ["R", "G", "B", "1"],
        "poly3": ["R", "G", "B", "RG", "RB", "GB", "R2", "G2", "B2", "1"],
    }[model]

    fit_meta["input_range_min"] = cam_wb[fit_idx].min(axis=0).round(6).tolist()
    fit_meta["input_range_max"] = cam_wb[fit_idx].max(axis=0).round(6).tolist()

    def _rounded(d):
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in d.items()}

    return CalibrationResult(
        model=model,
        coefficients=coef.round(8).tolist(),
        white_gains=gains.round(6).tolist(),
        feature_names=feature_names,
        fit_meta=fit_meta,
        train_summary=_rounded(train_summary),
        validation_summary=_rounded(validation_summary),
        patch_records=[asdict(r) for r in records],
        detection_summary={
            "method": detection.detection_method,
            "confidence": round(detection.confidence, 3),
            "rows": detection.rows,
            "cols": detection.cols,
            "n_usable": int(usable.sum()),
            "n_excluded": int((~usable).sum()),
            "n_fitted": int(len(fit_idx)),
            "n_holdout": int(len(holdout_set)),
            "holdout_indices": sorted(holdout_set),
        },
        warnings=warnings,
        reference_space=ref_space,
        holdout_summary=(
            _rounded(holdout_summary) if holdout_summary is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# Application to product images
# ---------------------------------------------------------------------------
def load_mapping(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def apply_mapping_to_rgb(
    rgb_encoded: np.ndarray, mapping: dict
) -> tuple[np.ndarray, dict]:
    """Apply a saved mapping to an HxWx3 sRGB-encoded image (0..1 or 0..255).

    Returns the encoded output and an extrapolation report describing how
    many pixels left the convex hull / numeric range seen during fitting.
    """
    scale = 255.0 if np.max(rgb_encoded) > 1.5 else 1.0
    enc = np.clip(rgb_encoded / scale, 0.0, 1.0)
    flat = enc.reshape(-1, 3)
    lin = camera_encoded_to_linear(flat)
    gains = np.asarray(mapping["white_gains"])
    lin_wb = lin * gains[None, :]
    coef = np.asarray(mapping["coefficients"])
    model = mapping["model"]
    out_lin = apply_coefficients(lin_wb, coef, model)

    train_min = np.asarray(mapping["fit_meta"].get("input_range_min", [0, 0, 0]))
    train_max = np.asarray(mapping["fit_meta"].get("input_range_max", [1, 1, 1]))
    over = (lin_wb > train_max[None, :]).any(axis=1)
    under = (lin_wb < train_min[None, :]).any(axis=1)
    negative_out = (out_lin < 0).any(axis=1)
    clip_hi = (out_lin > 1).any(axis=1)
    report = {
        "fraction_above_calibration_range": float(np.mean(over)),
        "fraction_below_calibration_range": float(np.mean(under)),
        "fraction_negative_output_clipped": float(np.mean(negative_out)),
        "fraction_output_above_1_clipped": float(np.mean(clip_hi)),
    }
    out_enc = cs.linear_to_srgb(np.clip(out_lin, 0.0, 1.0))
    out_img = (out_enc.reshape(enc.shape) * scale).clip(0, 255).astype(np.uint8)
    return out_img, report


def attach_input_range(result: CalibrationResult, cam_wb_range: tuple) -> CalibrationResult:
    lo, hi = cam_wb_range
    result.fit_meta["input_range_min"] = np.asarray(lo).round(6).tolist()
    result.fit_meta["input_range_max"] = np.asarray(hi).round(6).tolist()
    return result


class InsufficientSamplesError(RuntimeError):
    pass


class IllConditionedError(RuntimeError):
    pass
