# dual_camera_kit — fictional dual-camera reference kit

This directory is a **camera calibration material pack**: two simulated
camera sensors observing one reference chart under several illuminations, with
a deliberately different "foreign" camera included as a negative control.

Nothing here is a photograph. Every image is a **simulated response** generated
deterministically from `kit_params.json`; the manifest records that fact per
file (`"simulated_response": true`). The kit exists so cross-camera calibration
can be exercised, rebuilt, and verified without real hardware, and so the
reported colour residuals can be tied to exact files and pixel rectangles.

## What is in the pack

| Path | Contents |
| --- | --- |
| `kit_params.json` | The single source of truth: camera matrices, illumination gains, noise, chart geometry, holdout patches, acceptance thresholds, and the provenance hash of itself. |
| `manifest.json` | Generated inventory: every image file, its role, camera, illumination, SHA-256, and (for scenes) the pixel rectangles of each colour block. Provenance (params SHA-256, builder) is recorded here. |
| `charts/` | Chart observations: `chart_<camera>_<illumination>.png` for cameras `cam_a`, `cam_b` × illuminations `neutral`, `warm`, `cool`, `dim`. |
| `scenes/` | Product-like colour-band scenes under the same camera/illumination pairs, used to show mapped cross-camera samples. |
| `control/` | `foreign_cam` observations. A mapping fitted on `cam_a` must **fail** on these; they are the negative control. |

The reference chart is the built-in 24-patch ColorChecker (post-2014 X-Rite D50
Lab). The kit declares four **holdout patches** — `blue_sky`, `orange_yellow`,
`cyan`, `neutral_7_5` — that are excluded from the fit and from white-balance
estimation, and whose residuals are the independent generalization check.

## Simulation models (why each difference exists)

The manifest's `simulation_models` list corresponds one-to-one with the code in
`colorcalib/simulate.py`:

- **camera colour mixing** — a fixed 3x3 matrix in linear light per camera
  (`camera_profiles[*].matrix`). `cam_a` and `cam_b` differ in cross-talk;
  `foreign_cam` is deliberately far from both.
- **illumination / white balance** — per-channel diagonal gains plus exposure.
  `warm` and `cool` shift the neutral axis; `dim` is the same neutral light at
  lower exposure.
- **sensor noise** — seeded Gaussian, `noise_sigma` per camera.
- **black level** — a linear offset added before clipping.
- **patches** — matte, no texture; chart border and background are synthetic.

No effect is a hidden fudge factor: change a parameter, rebuild, and the images
change accordingly.

## Rebuilding (must be reproducible)

```bash
# from the repository root
PYTHONPATH=. python scripts/build_camera_kit.py --kit camera_kits/dual_camera_kit
```

Re-running the builder reproduces byte-identical PNGs (the verifier and the
test suite both check this). The builder refuses nothing silently: it writes
the manifest with per-file hashes, and if `kit_params.json` drifts the
verification step below stops with an error instead of reusing stale files.

## Verifying the agreed contract

```bash
PYTHONPATH=. python scripts/rebuild_camera_kit.py \
    --kit camera_kits/dual_camera_kit --out <some/out/dir>
```

Exit code 0 means every check passed. The script writes
`kit_verification.json` containing, per camera/illumination:

- re-render pixel-diff vs. the on-disk files (must be 0/255),
- fitted train / cross-validated / **holdout** ΔE2000,
- the recovered camera matrix vs. the true matrix,
- **scene residuals** with the exact pixel rectangle each came from.

The agreement checked is:

1. **Source reproducibility** — checked-in pixels match a fresh render and
   their recorded SHA-256.
2. **Same camera generalizes across lighting** — fit on `cam_a` neutral,
   predict `cam_a` warm's holdout colours, mean ΔE ≤ acceptance threshold.
3. **A different camera is rejected** — the same mapping on `foreign_cam`
   leaves full-chart mean ΔE *above* the threshold. If the negation of this
   fails, the script exits non-zero: the kit would not be discriminating.

Defaults live in `acceptance` in `kit_params.json`; the verifier reads them and
never invents a passing value. Change the kit and the thresholds you must meet
change with it.

## Cross-camera contract (what "consistent" means)

A fitted mapping carries the **sensor → reference colour transform** (the
matrix). White balance is a property of the illumination, so when a mapping is
applied to a target image the target image's own neutral patches re-derive its
gains. This is why the same camera transfers across illuminations, while a
different camera does not. The web UI performs exactly this and labels the
outcome `consistent` / `weak` / `mismatch` from measured ΔE — it does not
override the number with a default.

## Starting the interface

The kit is loaded read-only through the existing image-calibration web UI:

```bash
PYTHONPATH=. python -m web.app --host 127.0.0.1 --port 5000
```

Then open <http://127.0.0.1:5000> and use the **"Cross-camera kit"** panel:

1. pick a kit (only `dual_camera_kit` exists here),
2. choose a **source chart** (the mapping is fitted on it) and a **target
   chart** (a different camera/illumination is pre-selected so the first run is
   a real transfer),
3. click swatches in the **Holdout patches** strip to choose which patches are
   excluded from the fit; the kit's declared holdout set is pre-selected,
4. optionally paste **manual chart corners** if the geometry were not known,
5. click **Calibrate across cameras**.

The preview area shows the source chart, the target chart, and a mapped
**cross-camera sample** (the target scene) with per-block ΔE2000 and the pixel
rectangle it was measured over. The table below the previews lists every
target-patch colour residual, with holdout rows highlighted. Controlled
failures still surface: a foreign target reports `mismatch`, and a refused
kernel returns the kernel's message rather than a blank success.

## Reproducing the legacy single-camera example

The original single-camera example still runs end to end after these changes:

```bash
PYTHONPATH=. python scripts/run_legacy_example.py --examples examples
```

It regenerates all seven synthetic scenarios, calibrates each, applies the
mapping to the raw product image, and fails if any stage refuses or if the
checked-in PNGs no longer match a fresh render. Exit code 0 means all seven
scenarios calibrated and applied.
