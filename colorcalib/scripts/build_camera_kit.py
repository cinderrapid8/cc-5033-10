"""Build the dual-camera reference kit from ``kit_params.json``.

Run from the repository root::

    python scripts/build_camera_kit.py --kit camera_kits/dual_camera_kit

Every image is rendered deterministically from the parameters, so re-running
this script reproduces identical pixels (the verifier checks this).  The
manifest records what was produced, each file's role, and the SHA-256 of the
parameters and of each image; it never claims these are photographs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colorcalib.charts import builtin_colorchecker24, load_chart_json
from colorcalib.simulate import (
    CameraProfile,
    Illumination,
    chart_corners,
    render_chart_image,
    render_scene_image,
    write_image,
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_params(kit_dir: str) -> tuple[dict, str]:
    path = os.path.join(kit_dir, "kit_params.json")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    return json.loads(text), sha256_text(text)


def build(kit_dir: str) -> dict:
    params, params_sha = load_params(kit_dir)
    chart = (
        load_chart_json(params["chart"]["path"])
        if params["chart"].get("path")
        else builtin_colorchecker24()
    )
    geom = params.get("render_geometry", {})
    cell = int(geom.get("cell_px", 120))
    gap = int(geom.get("gap_px", 14))
    block_h = int(geom.get("scene_block_height_px", 64))
    corners = chart_corners(chart.rows, chart.cols, cell, gap)

    cams = {
        name: CameraProfile(
            name,
            np.asarray(spec["matrix"], dtype=np.float64),
            float(spec.get("noise_sigma", 0.0015)),
            float(spec.get("black_level", 0.0)),
        )
        for name, spec in params["camera_profiles"].items()
    }
    ills = {
        name: Illumination(
            name,
            tuple(float(g) for g in spec["gains"]),
            float(spec.get("exposure", 1.0)),
            spec.get("label", ""),
        )
        for name, spec in params["illuminations"].items()
    }
    blocks = [
        (b["name"], tuple(float(v) for v in b["rgb"])) for b in params["scene_blocks"]
    ]

    produced: list[dict] = []
    chart_dir = os.path.join(kit_dir, "charts")
    scene_dir = os.path.join(kit_dir, "scenes")
    control_dir = os.path.join(kit_dir, "control")
    for d in (chart_dir, scene_dir, control_dir):
        os.makedirs(d, exist_ok=True)

    def _emit(dirpath, filename, rgb, role, camera, illumination, extra=None):
        path = os.path.join(dirpath, filename)
        write_image(path, rgb)
        rec = {
            "file": os.path.relpath(path, kit_dir),
            "role": role,
            "camera": camera,
            "illumination": illumination,
            "simulated_response": True,
            "sha256": sha256_file(path),
        }
        if extra:
            rec.update(extra)
        produced.append(rec)
        return rec

    for cam_name in ("cam_a", "cam_b"):
        for ill_name in params["imaged_illuminations"]:
            cam, ill = cams[cam_name], ills[ill_name]
            chart_img = render_chart_image(
                cam, ill, chart, cell, gap,
                seed_parts=(params["kit_name"], cam_name, ill_name),
            )
            _emit(
                chart_dir, f"chart_{cam_name}_{ill_name}.png", chart_img,
                "chart_observation", cam_name, ill_name,
                {"chart_corners_xy": corners},
            )
            scene_img, layout = render_scene_image(
                cam, ill, blocks, block_h,
                seed_parts=(params["kit_name"], cam_name, ill_name),
            )
            _emit(
                scene_dir, f"scene_{cam_name}_{ill_name}.png", scene_img,
                "scene_observation", cam_name, ill_name,
                {"blocks": layout},
            )

    # Negative control: a *foreign* camera under the reference illumination.
    # It is evidence that the kit refuses a transfer that does not belong.
    foreign = cams["foreign_cam"]
    ref_ill = ills["neutral"]
    chart_img = render_chart_image(
        foreign, ref_ill, chart, cell, gap,
        seed_parts=(params["kit_name"], "foreign_cam", "neutral"),
    )
    _emit(control_dir, "chart_foreign_cam_neutral.png", chart_img,
          "negative_control_chart", "foreign_cam", "neutral",
          {"chart_corners_xy": corners})
    scene_img, layout = render_scene_image(
        foreign, ref_ill, blocks, block_h,
        seed_parts=(params["kit_name"], "foreign_cam", "neutral"),
    )
    _emit(control_dir, "scene_foreign_cam_neutral.png", scene_img,
          "negative_control_scene", "foreign_cam", "neutral",
          {"blocks": layout})

    manifest = {
        "schema": "colorcalib.camera_kit_manifest/v1",
        "kit_name": params["kit_name"],
        "kit_class": "synthetic",
        "simulation_note": params["simulation_note"],
        "provenance": {
            "params_file": "kit_params.json",
            "params_sha256": params_sha,
            "builder": "scripts/build_camera_kit.py",
            "version": "1",
        },
        "chart": params["chart"],
        "camera_profiles": {
            name: {
                "label": spec.get("label", ""),
                "matrix": spec["matrix"],
                "noise_sigma": spec.get("noise_sigma", 0.0015),
                "black_level": spec.get("black_level", 0.0),
            }
            for name, spec in params["camera_profiles"].items()
        },
        "illuminations": params["illuminations"],
        "imaged_illuminations": params["imaged_illuminations"],
        "holdout_validation_illumination": params.get("holdout_validation_illumination"),
        "acceptance": params["acceptance"],
        "files": produced,
    }
    manifest_path = os.path.join(kit_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, help="kit directory containing kit_params.json")
    args = ap.parse_args()
    manifest = build(args.kit)
    print(f"[ok] built {len(manifest['files'])} image(s) for kit "
          f"'{manifest['kit_name']}'")
    print(f"[ok] manifest: {os.path.join(args.kit, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
