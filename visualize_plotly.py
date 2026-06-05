#!/usr/bin/env python3
"""
Interactive Plotly-based antipsychotic episode timeline visualization.

Output: self-contained HTML with patient dropdown + Prev/Next navigation.

Three visual layers per patient (back to front):
  1. Coloured boxes       — predicted episodes (colour per drug)
  2. Shaded bands         — polypharmacy periods (orange = transition overlap,
                            red = concurrent therapy); hover for details
  3. Scatter dots         — raw prescription records (colour per drug)

Usage:
    python visualize_plotly.py                       # auto-selects 20 patients
    python visualize_plotly.py --patient_ids 12345 67890  # specific patients
    python visualize_plotly.py --n 50                # top-N patients
"""

import argparse
import json
import os

import pandas as pd
import plotly.graph_objects as go
import plotly.colors

DATA_DIR = "data"
RESULTS_DIR = "results"

# Only load these columns from the prescription CSV
APS_NEEDED = {"drug_name", "patient_id", "date"}

PALETTE = plotly.colors.qualitative.Plotly   # 10 distinct colours

# Visual style for each polypharmacy type
POLY_STYLE = {
    "transition_overlap": {
        "fill":   "rgba(255,165,0,0.22)",
        "border": "rgba(210,120,0,0.85)",
        "label":  "Transition Overlap",
    },
    "concurrent_therapy": {
        "fill":   "rgba(220,53,69,0.18)",
        "border": "rgba(180,30,45,0.80)",
        "label":  "Concurrent Therapy",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_aps() -> pd.DataFrame:
    """Load prescription data from CSV."""
    path = os.path.join(DATA_DIR, "antipsychotic_prescriptions.csv")
    # usecols callable strips BOM and spaces from header names
    aps = pd.read_csv(
        path,
        usecols=lambda c: c.strip().lstrip("﻿") in APS_NEEDED,
        low_memory=False,
    )
    aps.columns = aps.columns.str.strip().str.lstrip("﻿")
    aps["date"] = pd.to_datetime(aps["date"], errors="coerce")
    aps = aps.dropna(subset=["date"])
    aps["drug_name"] = aps["drug_name"].str.lower().str.strip()
    aps["drug_label"] = aps["drug_name"]
    return aps[["patient_id", "drug_label", "date"]]


def load_predictions() -> pd.DataFrame:
    """Load predicted episode results."""
    pred = pd.read_csv(os.path.join(RESULTS_DIR, "episodes.csv"))
    pred.columns = pred.columns.str.strip()
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
    pred["end_date"] = pd.to_datetime(pred["end_date"], errors="coerce")
    return pred


def load_polypharmacy() -> dict:
    """Load polypharmacy detection results. Returns {} if not available."""
    path = os.path.join(RESULTS_DIR, "episodes_polypharmacy.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    df["period_end"]   = pd.to_datetime(df["period_end"],   errors="coerce")
    return {
        pid: group.to_dict("records")
        for pid, group in df.groupby("patient_id")
    }


# ---------------------------------------------------------------------------
# Per-patient figure builder
# ---------------------------------------------------------------------------

def _fmt_med(med: str) -> str:
    """Human-readable medication label (strip route suffix if present)."""
    return med.replace("_Oral", "").replace("_LAI", "").replace("_oral", "").replace("_lai", "").title()


def _norm_med(med: str) -> str:
    """Strip optional route suffix (_oral / _lai) from a medication name."""
    return med.lower().replace("_oral", "").replace("_lai", "").strip()


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #RRGGBB hex colour to an rgba() CSS string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def make_patient_figure(
    patient_id: int,
    aps: pd.DataFrame,
    pred: pd.DataFrame,
    polypharmacy: list,
) -> tuple:
    """Build an interactive Plotly figure for one patient plus panel HTML.

    Visual layers (back to front):
      1. Coloured boxes  — predicted episodes (colour per drug)
      2. Shaded bands    — polypharmacy periods with hover tooltips
      3. Scatter dots    — raw prescription records (colour per drug)

    Returns (fig, panel_html).
    """
    dat = aps[aps["patient_id"] == patient_id].copy().sort_values("date")

    if dat.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No prescription records for patient {patient_id}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        return fig, ""

    # Ordered medication list from raw data (first appearance = y-axis order)
    med_order = dat.drop_duplicates("drug_label", keep="first")["drug_label"].tolist()
    med_to_y: dict = {med: i for i, med in enumerate(med_order)}

    # Extend with any medications that appear only in predicted episodes
    p_pred = pred[pred["patient_id"] == patient_id]
    for _, row in p_pred.iterrows():
        med_norm = _norm_med(row["medication"])
        if med_norm not in med_to_y:
            med_to_y[med_norm] = len(med_to_y)
            med_order.append(med_norm)

    shapes = []
    traces = []

    # ------------------------------------------------------------------ #
    # Layer 1 — Predicted episodes (boxes, below everything)             #
    # ------------------------------------------------------------------ #
    for _, row in p_pred.iterrows():
        med = row["medication"]
        med_norm = _norm_med(med)
        y_pos = med_to_y[med_norm]

        base_hex = PALETTE[y_pos % len(PALETTE)]
        bar_color = _hex_to_rgba(base_hex, 0.30)
        border_color = _hex_to_rgba(base_hex, 0.85)

        s = row["date"]
        e = row["end_date"]
        if pd.isna(s):
            continue

        shapes.append(dict(
            type="rect",
            x0=s, x1=e if pd.notna(e) else dat["date"].max(),
            y0=y_pos - 0.38, y1=y_pos + 0.38,
            fillcolor=bar_color,
            line=dict(width=1.5, color=border_color),
            layer="below",
        ))

    # ------------------------------------------------------------------ #
    # Layer 2 — Polypharmacy periods (shaded bands + invisible hover)    #
    # ------------------------------------------------------------------ #
    poly_types_shown = set()

    for period in polypharmacy:
        ptype = period.get("type", "")
        style = POLY_STYLE.get(ptype)
        if style is None:
            continue

        p_start = pd.Timestamp(period["period_start"])
        p_end   = pd.Timestamp(period["period_end"])

        y0 = med_to_y.get(_norm_med(period["primary_med"]))
        y1 = med_to_y.get(_norm_med(period["secondary_med"]))

        if y0 is None and y1 is None:
            continue
        if y0 is None:
            y0 = y1
        if y1 is None:
            y1 = y0

        shapes.append(dict(
            type="rect",
            x0=p_start, x1=p_end,
            y0=min(y0, y1) - 0.45, y1=max(y0, y1) + 0.45,
            fillcolor=style["fill"],
            line=dict(width=1.5, color=style["border"], dash="dot"),
            layer="below",
        ))

        # Invisible scatter point at band centre — carries the hover tooltip
        mid_x = p_start + (p_end - p_start) / 2
        mid_y = (y0 + y1) / 2

        if ptype == "transition_overlap":
            hover = (
                f"<b>Transition Overlap</b><br>"
                f"{_fmt_med(period['primary_med'])} → {_fmt_med(period['secondary_med'])}<br>"
                f"{p_start.strftime('%Y-%m-%d')} – {p_end.strftime('%Y-%m-%d')}<br>"
                f"Overlap: {period.get('overlap_days', '?')} days"
                "<extra></extra>"
            )
        else:
            hover = (
                f"<b>Concurrent Therapy</b><br>"
                f"{_fmt_med(period['primary_med'])} + {_fmt_med(period['secondary_med'])}<br>"
                f"{p_start.strftime('%Y-%m-%d')} – {p_end.strftime('%Y-%m-%d')}<br>"
                f"Secondary mentions: {period.get('secondary_mentions', '?')}"
                "<extra></extra>"
            )

        traces.append(go.Scatter(
            x=[mid_x], y=[mid_y],
            mode="markers",
            marker=dict(size=18, opacity=0),
            hovertemplate=hover,
            showlegend=False,
            name="",
        ))

        poly_types_shown.add(ptype)

    # ------------------------------------------------------------------ #
    # Layer 3 — Raw prescription records (scatter dots, frontmost)        #
    # ------------------------------------------------------------------ #
    for i, med in enumerate(med_order):
        med_dat = dat[dat["drug_label"] == med]
        if med_dat.empty:
            continue
        color = PALETTE[i % len(PALETTE)]
        traces.append(go.Scatter(
            x=med_dat["date"],
            y=[med_to_y[med]] * len(med_dat),
            mode="markers",
            marker=dict(
                size=8, color=color, opacity=0.9,
                line=dict(color="white", width=0.8),
            ),
            name=_fmt_med(med),
            legendgroup=med,
            hovertemplate=(
                f"<b>{_fmt_med(med)}</b><br>"
                "Date: %{x|%Y-%m-%d}<extra></extra>"
            ),
        ))

    # Dummy legend entries
    traces.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=13, color="rgba(120,120,120,0.4)", symbol="square"),
        name="Predicted Episode",
    ))
    for ptype, style in POLY_STYLE.items():
        if ptype in poly_types_shown:
            traces.append(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(
                    size=13, symbol="square",
                    color=style["fill"],
                    line=dict(color=style["border"], width=2),
                ),
                name=style["label"],
            ))

    y_labels = [_fmt_med(m) for m in med_order]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f"Patient {patient_id}", font=dict(size=14, color="#333")),
        xaxis=dict(
            title="Date",
            showgrid=True, gridcolor="rgba(0,0,0,0.08)",
            type="date",
        ),
        yaxis=dict(
            title="Drug",
            tickmode="array",
            tickvals=list(range(len(med_order))),
            ticktext=y_labels,
            showgrid=True, gridcolor="rgba(0,0,0,0.08)",
        ),
        shapes=shapes,
        annotations=[],
        height=max(400, 80 + len(med_order) * 55),
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        margin=dict(l=190, r=20, t=80, b=60),
        hovermode="closest",
    )

    panel_html = _build_panel_html(p_pred, polypharmacy)
    return fig, panel_html


def _build_panel_html(p_pred: pd.DataFrame, polypharmacy: list) -> str:
    """Build the episode + polypharmacy summary panel for one patient."""
    sections = []

    # Episodes
    if p_pred.empty:
        sections.append("<span style='color:#888'>No predicted episodes.</span>")
    else:
        rows = []
        for _, r in p_pred.iterrows():
            med = _fmt_med(r["medication"])
            s = pd.Timestamp(r["date"]).strftime("%Y-%m-%d") if pd.notna(r["date"]) else "?"
            e = pd.Timestamp(r["end_date"]).strftime("%Y-%m-%d") if pd.notna(r["end_date"]) else "ongoing"
            rows.append(f"<li><b>{med}</b>: {s} &rarr; {e}</li>")
        sections.append(
            "<b>Predicted episodes:</b>"
            "<ul style='margin:4px 0 0 18px;padding:0'>"
            + "".join(rows) + "</ul>"
        )

    # Polypharmacy
    if polypharmacy:
        rows = []
        for p in polypharmacy:
            ptype = p.get("type", "")
            p_start = pd.Timestamp(p["period_start"]).strftime("%Y-%m-%d")
            p_end   = pd.Timestamp(p["period_end"]).strftime("%Y-%m-%d")
            prim    = _fmt_med(p["primary_med"])
            sec     = _fmt_med(p["secondary_med"])

            if ptype == "transition_overlap":
                detail = f"overlap {p.get('overlap_days', '?')} days"
                label  = (
                    f"<span style='color:#d47800'><b>Transition overlap</b></span>: "
                    f"{prim} &rarr; {sec} &nbsp; "
                    f"<span style='color:#666'>{p_start} &ndash; {p_end} &nbsp;({detail})</span>"
                )
            else:
                detail = f"{p.get('secondary_mentions', '?')} secondary mentions"
                label  = (
                    f"<span style='color:#b41e2d'><b>Concurrent therapy</b></span>: "
                    f"{prim} + {sec} &nbsp; "
                    f"<span style='color:#666'>{p_start} &ndash; {p_end} &nbsp;({detail})</span>"
                )
            rows.append(f"<li>{label}</li>")

        sections.append(
            "<b>Polypharmacy periods:</b>"
            "<ul style='margin:4px 0 0 18px;padding:0'>"
            + "".join(rows) + "</ul>"
        )

    return "<br>".join(sections)


# ---------------------------------------------------------------------------
# HTML wrapper
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Antipsychotic Episode Timeline Viewer</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f4f6f9; }}
    h1   {{ color: #2c3e50; margin-bottom: 4px; }}
    p.sub {{ color: #555; margin-bottom: 16px; font-size: 13px; }}
    .legend-row {{
      display: flex; gap: 22px; align-items: center; flex-wrap: wrap;
      background: white; padding: 12px 18px; border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin-bottom: 14px;
    }}
    .li {{ display: flex; align-items: center; gap: 7px; font-size: 12px; color: #333; }}
    .sw {{ width: 20px; height: 14px; border-radius: 3px; }}
    .controls {{
      display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
      background: white; padding: 12px 18px; border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin-bottom: 14px;
    }}
    .controls label {{ font-weight: bold; color: #444; font-size: 13px; }}
    select {{
      padding: 7px 11px; border: 1px solid #ccc; border-radius: 6px;
      font-size: 13px; min-width: 220px; cursor: pointer;
    }}
    button {{
      padding: 7px 15px; background: #3a7bd5; color: white;
      border: none; border-radius: 6px; cursor: pointer; font-size: 13px;
    }}
    button:hover {{ background: #2d62ae; }}
    .badge {{
      background: #3a7bd5; color: white; border-radius: 12px;
      padding: 2px 10px; font-size: 12px; margin-left: 4px;
    }}
    #plot-container {{
      background: white; border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 10px;
    }}
    #episode-panel {{
      background: white; border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.1);
      padding: 14px 20px; margin-top: 14px;
      font-size: 13px; line-height: 1.9; color: #333;
    }}
    #episode-panel ul {{ margin: 4px 0 8px 18px; padding: 0; }}
  </style>
</head>
<body>
<h1>Antipsychotic Episode Timeline Viewer</h1>
<p class="sub">
  {n_patients} patients loaded
  <span class="badge">Interactive — zoom · pan · hover</span>
</p>

<div class="legend-row">
  <strong style="font-size:13px">Legend:</strong>
  <div class="li">
    <div class="sw" style="background:rgba(100,100,210,0.3);border:2px solid rgba(100,100,210,0.85);"></div>
    Predicted Episode (colour = drug)
  </div>
  <div class="li">
    <div class="sw" style="background:rgba(255,165,0,0.22);border:2px dotted rgba(210,120,0,0.85);"></div>
    Transition Overlap
  </div>
  <div class="li">
    <div class="sw" style="background:rgba(220,53,69,0.18);border:2px dotted rgba(180,30,45,0.80);"></div>
    Concurrent Therapy
  </div>
  <div class="li">
    <div style="width:10px;height:10px;border-radius:50%;background:#636efa;opacity:0.85;"></div>
    Raw Prescription Records
  </div>
</div>

<div class="controls">
  <label for="ptSel">Patient:</label>
  <select id="ptSel" onchange="showPatient(this.value)">
    {options}
  </select>
  <button onclick="navigate(-1)">&#8592; Prev</button>
  <button onclick="navigate(1)">Next &#8594;</button>
  <span id="pt-counter" style="color:#666;font-size:13px;"></span>
</div>

<div id="plot-container">
  <div id="main-plot"></div>
</div>

<div id="episode-panel"></div>

<script>
var figData = {fig_data_json};
var patientIds = {patient_ids_json};
var panelData = {panel_data_json};

function showPatient(id) {{
  id = Number(id);
  var f = figData[id];
  if (!f) return;
  Plotly.newPlot('main-plot', f.data, f.layout, {{responsive: true, displayModeBar: true}});
  document.getElementById('ptSel').value = id;
  var idx = patientIds.indexOf(id);
  document.getElementById('pt-counter').textContent =
    'Patient ' + (idx + 1) + ' of ' + patientIds.length;
  document.getElementById('episode-panel').innerHTML =
    panelData[id] || '<span style="color:#888">No data.</span>';
}}

function navigate(dir) {{
  var sel = document.getElementById('ptSel');
  var idx = patientIds.indexOf(Number(sel.value));
  var newIdx = ((idx + dir) % patientIds.length + patientIds.length) % patientIds.length;
  showPatient(patientIds[newIdx]);
}}

if (patientIds.length > 0) showPatient(patientIds[0]);
</script>
</body>
</html>
"""


def build_html(patient_ids: list, aps: pd.DataFrame, pred: pd.DataFrame,
               polypharmacy: dict) -> str:
    """Render all patient figures to JSON and embed in HTML template."""
    fig_data = {}
    panel_data = {}
    options_html = []

    for i, patient_id in enumerate(patient_ids):
        print(f"  Rendering patient {i+1}/{len(patient_ids)}: {patient_id} ...", end="\r")
        patient_poly = polypharmacy.get(patient_id, [])
        fig, panel_html = make_patient_figure(patient_id, aps, pred, patient_poly)

        fig_data[int(patient_id)] = json.loads(fig.to_json())
        panel_data[int(patient_id)] = panel_html
        options_html.append(f'<option value="{int(patient_id)}">Patient {int(patient_id)}</option>')

    print()
    return HTML_TEMPLATE.format(
        n_patients=len(patient_ids),
        options="\n    ".join(options_html),
        fig_data_json=json.dumps(fig_data),
        patient_ids_json=json.dumps([int(b) for b in patient_ids]),
        panel_data_json=json.dumps(panel_data),
    )


# ---------------------------------------------------------------------------
# Patient selection helpers
# ---------------------------------------------------------------------------

def select_patients(pred: pd.DataFrame, n: int) -> list:
    """Pick up to n patients with the most predicted episodes."""
    top = pred.groupby("patient_id")["order"].count().nlargest(n).index.tolist()
    return sorted(top[:n])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive Plotly antipsychotic episode timeline viewer"
    )
    parser.add_argument(
        "--patient_ids", nargs="+", type=int,
        help="Specific patient IDs to include",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of patients to auto-select (default: 20)",
    )
    parser.add_argument(
        "--out", type=str,
        default=os.path.join(RESULTS_DIR, "viewer.html"),
        help="Output HTML path",
    )
    args = parser.parse_args()

    print("Loading data ...")
    aps = load_aps()
    print(f"  APS records: {len(aps):,}")

    pred = load_predictions()
    print(f"  Predicted episodes: {len(pred):,}")

    polypharmacy = load_polypharmacy()
    n_poly = sum(len(v) for v in polypharmacy.values())
    print(f"  Polypharmacy periods: {n_poly:,} across {len(polypharmacy)} patients")

    if args.patient_ids:
        patient_ids = args.patient_ids
    else:
        patient_ids = select_patients(pred, args.n)

    print(f"Building HTML for {len(patient_ids)} patients ...")
    html = build_html(patient_ids, aps, pred, polypharmacy)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nSaved: {args.out}")
    print("Open the HTML file in any browser — no server required.")


if __name__ == "__main__":
    main()
