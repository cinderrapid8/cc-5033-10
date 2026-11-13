"""Self-contained HTML report: per-patch adopted values, residuals, diagnostics."""

from __future__ import annotations

import html
import json

import numpy as np

from .charts import Chart
from .detection import DetectionResult
from .pipeline import CalibrationResult, linear_srgb_to_lab_d50


def _swatch(rgb01) -> str:
    r, g, b = [int(round(float(v) * 255)) for v in np.clip(rgb01, 0, 1)]
    return (
        f'<span class="sw" style="background:rgb({r},{g},{b})" '
        f'title="sRGB {r},{g},{b}"></span>'
    )


def write_html_report(
    result: CalibrationResult,
    chart: Chart,
    detection: DetectionResult,
    path: str,
) -> None:
    ref_srgb = chart.patch_srgb()
    diag_map = {d.index: d for d in detection.diagnoses}

    rows_html = []
    for rec in result.patch_records:
        i = rec["index"]
        d = diag_map[i]
        badges = []
        for reason in rec["diagnoses"]["reject_reasons"]:
            badges.append(f'<span class="badge bad">{html.escape(reason)}</span>')
        if not badges:
            badges.append('<span class="badge ok">clean</span>')
        role_cls = {
            "train": "role-train",
            "train+cv": "role-train",
            "validation": "role-val",
            "excluded": "role-excl",
        }[rec["role"]]
        de = rec["delta_e2000"]
        de_str = "—" if de is None else f"{de:.2f}"
        cv_de = rec["diagnoses"].get("cv_delta_e2000")
        cv_str = "—" if cv_de is None else f"{cv_de:.2f}"
        resid = rec["residual_linear_rgb"]
        resid_str = "—" if resid is None else ", ".join(f"{v:+.4f}" for v in resid)
        iw = "—" if rec["irls_weight"] is None else f"{rec['irls_weight']:.2f}"
        rows_html.append(
            f"""<tr class="{role_cls}">
              <td>{i}</td>
              <td>{html.escape(rec['name'])}</td>
              <td>{_swatch(rec['adopted_srgb'])}</td>
              <td>{_swatch(ref_srgb[i])}</td>
              <td class="num">{', '.join(f'{v:.3f}' for v in rec['adopted_srgb'])}</td>
              <td class="num">{', '.join(f'{v:.1f}' for v in rec['reference_lab_d50'])}</td>
              <td class="num">{de_str}</td>
              <td class="num">{cv_str}</td>
              <td class="num">{resid_str}</td>
              <td class="num">{iw}</td>
              <td>{''.join(badges)}</td>
              <td><span class="role {role_cls}">{rec['role']}</span></td>
            </tr>"""
        )

    coef = np.asarray(result.coefficients)
    coef_rows = "".join(
        "<tr>" + "".join(f"<td class='num'>{v:.5f}</td>" for v in row) + "</tr>"
        for row in coef
    )
    feat = ", ".join(result.feature_names)
    vs = result.validation_summary or {}
    warnings_html = "".join(
        f"<li>{html.escape(w)}</li>" for w in result.warnings
    ) or "<li>none</li>"

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Color calibration report — {html.escape(chart.name)}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color:#222; }}
 h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 28px; }}
 table {{ border-collapse: collapse; font-size: 12px; }}
 th, td {{ border: 1px solid #ccc; padding: 4px 7px; text-align: left; }}
 th {{ background: #f2f2f2; position: sticky; top: 0; }}
 td.num {{ font-variant-numeric: tabular-nums; }}
 .sw {{ display:inline-block; width:22px; height:22px; border:1px solid #999; vertical-align:middle; }}
 .badge {{ display:inline-block; padding:1px 6px; border-radius:8px; font-size:11px; margin:1px; }}
 .badge.bad {{ background:#f8d7da; color:#842029; }}
 .badge.ok {{ background:#d1e7dd; color:#0f5132; }}
 .role {{ padding:1px 6px; border-radius:6px; font-size:11px; }}
 .role-train {{ background:#e2e8ff; }} .role-val {{ background:#fff3cd; }}
 .role-excl {{ background:#f8d7da; }}
 tr.role-excl {{ opacity: .55; }}
 .cards {{ display:flex; gap:14px; flex-wrap:wrap; }}
 .card {{ border:1px solid #bbb; border-radius:8px; padding:10px 14px; min-width:170px; }}
 .card .v {{ font-size:22px; font-weight:600; }}
 code {{ background:#f4f4f4; padding:1px 4px; }}
</style></head><body>
<h1>Cross-camera color calibration report</h1>
<p>Chart: <b>{html.escape(chart.name)}</b> &middot; detection:
 <code>{result.detection_summary['method']}</code>
 (confidence {result.detection_summary['confidence']}) &middot;
 model: <code>{result.model}</code> &middot;
 usable patches: {result.detection_summary['n_usable']} /
 {result.detection_summary['rows'] * result.detection_summary['cols']}</p>

<h2>Error summary (CIEDE2000, separate train and held-out validation)</h2>
<div class="cards">
 {''.join(f'<div class="card"><div>{k}</div><div class="v">{v}</div></div>'
          for k, v in result.train_summary.items())}
</div>
<h3>Validation (patches never seen by the fit)</h3>
<div class="cards">
 {''.join(f'<div class="card"><div>{k}</div><div class="v">{v}</div></div>'
          for k, v in vs.items()) or '<p>no validation split</p>'}
</div>

<h2>Interpretable mapping (camera-linear RGB → reference-linear sRGB)</h2>
<p>Features: <code>{feat}</code>; white-balance gains (per channel):
 <code>{np.round(result.white_gains, 4).tolist()}</code>;
 design condition number: {result.fit_meta['condition_number']:.2e};
 fit DoF: {result.fit_meta['dof']}</p>
<table><tbody>{coef_rows}</tbody></table>

<h2>Warnings and honest limitations</h2>
<ul>{warnings_html}</ul>

<h2>Per-patch adopted values and residuals</h2>
<p>ΔE2000 (fit) is the production model's training residual on the full usable
set; <b>ΔE2000 (CV)</b> is each patch's honest cross-validated error from a
model that never saw it. IRLS weight 0 means influence-based hard rejection.</p>
<table>
<thead><tr><th>#</th><th>patch</th><th>adopted</th><th>reference</th>
<th>adopted sRGB (0..1)</th><th>ref Lab D50</th><th>ΔE2000 (fit)</th>
<th>ΔE2000 (CV)</th><th>linear RGB residual</th><th>IRLS weight</th>
<th>diagnostics</th><th>role</th></tr></thead>
<tbody>{''.join(rows_html)}</tbody></table>

<h2>Raw mapping JSON</h2>
<pre style="max-height:300px;overflow:auto;background:#f7f7f7;padding:10px;font-size:11px;">
{html.escape(json.dumps({
    'model': result.model,
    'white_gains': result.white_gains,
    'coefficients': result.coefficients,
    'feature_names': result.feature_names,
    'fit_meta': result.fit_meta,
    'reference_space': result.reference_space,
}, indent=2))}
</pre>
</body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
