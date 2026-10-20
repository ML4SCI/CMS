"""Export interactive Plotly ablation dashboard from metrics.jsonl.

    python -m ablation.export_plotly \\
        --runs part_ablation/runs \\
        --out logs/ablation-plots.html
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation.metrics import JETCLASS_LABELS  # noqa: E402
from ablation.plot_metrics import load_series  # noqa: E402

SIGNAL_CLASSES = [c for c in JETCLASS_LABELS if c != "QCD"]

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HERON / ParT ablation curves</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    --bg: #f6f4ef;
    --paper: #fffcf7;
    --ink: #1c1917;
    --muted: #57534e;
    --line: #e7e5e4;
    --accent: #1d4ed8;
    --chip: #e7e5e4;
    --plot-paper: #fffcf7;
    --plot-bg: #faf8f4;
    --grid: rgba(87, 83, 78, 0.18);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0c0a09;
      --paper: #1c1917;
      --ink: #f5f5f4;
      --muted: #a8a29e;
      --line: #44403c;
      --accent: #93c5fd;
      --chip: #292524;
      --plot-paper: #1c1917;
      --plot-bg: #0c0a09;
      --grid: rgba(168, 162, 158, 0.15);
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font: 15px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink);
    background: var(--bg);
  }
  .toolbar {
    position: sticky;
    top: 0;
    z-index: 10;
    background: var(--bg);
    border-bottom: 1px solid var(--line);
    padding: 12px 18px 14px;
  }
  .toolbar h1 {
    margin: 0 0 10px;
    font-size: 20px;
    font-weight: 650;
    letter-spacing: -0.02em;
  }
  .tabs { display: flex; gap: 6px; margin-bottom: 12px; flex-wrap: wrap; }
  .tab {
    border: 1px solid var(--line);
    background: var(--paper);
    color: var(--ink);
    padding: 6px 14px;
    cursor: pointer;
    font-size: 13px;
  }
  .tab.active { border-color: var(--accent); color: var(--accent); }
  .row { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; margin-bottom: 10px; }
  .row label { font-size: 13px; color: var(--muted); }
  .btn {
    border: 1px solid var(--line);
    background: var(--paper);
    color: var(--ink);
    padding: 4px 10px;
    font-size: 12px;
    cursor: pointer;
  }
  .btn:hover { border-color: var(--accent); color: var(--accent); }
  .arms {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 12px;
    max-height: 120px;
    overflow-y: auto;
    padding: 8px 10px;
    border: 1px solid var(--line);
    background: var(--paper);
    font-size: 12px;
  }
  .arms label { display: inline-flex; align-items: center; gap: 4px; cursor: pointer; white-space: nowrap; }
  select, input[type="number"] {
    border: 1px solid var(--line);
    background: var(--paper);
    color: var(--ink);
    padding: 4px 8px;
    font-size: 13px;
  }
  input[type="range"] { width: 140px; vertical-align: middle; }
  .content { padding: 16px 18px 48px; }
  .panel-grid { display: grid; gap: 18px; }
  .panel-grid.comparison { grid-template-columns: 1fr; }
  .panel-grid.per-arm { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .panel-grid.rejection { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  @media (max-width: 1100px) {
    .panel-grid.per-arm, .panel-grid.rejection { grid-template-columns: 1fr; }
  }
  .chart-wrap {
    background: var(--paper);
    border: 1px solid var(--line);
    padding: 8px 8px 4px;
    min-height: 400px;
  }
  .chart { width: 100%; height: 400px; }
  .caption {
    margin: 0 18px 24px;
    font-size: 12px;
    color: var(--muted);
    line-height: 1.5;
  }
  .hidden { display: none !important; }
  .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 1px;
    margin-right: 2px;
    vertical-align: middle;
  }
</style>
</head>
<body>
<div class="toolbar">
  <h1>HERON / ParT ablation training curves</h1>
  <div class="tabs">
    <button class="tab active" data-view="comparison">Comparison</button>
    <button class="tab" data-view="per-arm">Per arm</button>
    <button class="tab" data-view="rejection">Rejection</button>
  </div>
  <div class="row" id="arm-controls">
    <span><strong>Arms</strong></span>
    <button class="btn" id="btn-all">All</button>
    <button class="btn" id="btn-none">None</button>
    <button class="btn" id="btn-invert">Invert</button>
  </div>
  <div class="arms" id="arm-checkboxes"></div>
  <div class="row">
    <label>Matched step max
      <input type="range" id="step-slider" min="0" max="__STEP_MAX__" step="1000" value="__STEP_MAX__">
      <input type="number" id="step-input" min="0" max="__STEP_MAX__" step="1000" value="__STEP_MAX__" style="width:90px">
    </label>
    <button class="btn" id="btn-step-full">Full range</button>
    <button class="btn" id="btn-step-200k">To 200k</button>
  </div>
  <div class="row hidden" id="per-arm-row">
    <label>Arm
      <select id="arm-select"></select>
    </label>
  </div>
  <div class="row hidden" id="rej-row">
    <label><input type="checkbox" id="log-y" checked> Log y-axis (rejection)</label>
  </div>
</div>
<div class="content">
  <div class="panel-grid comparison" id="grid-comparison"></div>
  <div class="panel-grid per-arm hidden" id="grid-per-arm"></div>
  <div class="panel-grid rejection hidden" id="grid-rejection"></div>
</div>
<p class="caption">
  Source: <code>part_ablation/runs/*/metrics.jsonl</code> (local sync).
  <code>n8_k6</code> is a rotary-kernel bug archive, not a physics result.
  <code>n8_k6_v2</code> is registered not-U.
  Box-zoom any chart; double-click to reset. Toggle traces via legend.
</p>
<script>
const DATA = __DATA_JSON__;
const COLORS = [
  "#1d4ed8","#dc2626","#16a34a","#ca8a04","#9333ea","#0891b2","#ea580c",
  "#4f46e5","#be185d","#059669","#b45309","#7c3aed","#0d9488","#c2410c",
  "#2563eb","#15803d","#a21caf","#57534e","#0f766e","#991b1b"
];
const css = getComputedStyle(document.documentElement);
const PLOT_THEME = () => ({
  paper_bgcolor: css.getPropertyValue("--plot-paper").trim(),
  plot_bgcolor: css.getPropertyValue("--plot-bg").trim(),
  font: { color: css.getPropertyValue("--ink").trim(), size: 12 },
  margin: { t: 36, r: 16, b: 72, l: 56 },
  dragmode: "zoom",
  hovermode: "x unified",
  xaxis: { autorange: true, gridcolor: css.getPropertyValue("--grid").trim(), zeroline: false },
  yaxis: { autorange: true, gridcolor: css.getPropertyValue("--grid").trim(), zeroline: false },
  legend: { orientation: "h", y: -0.22, x: 0, font: { size: 10 } },
  showlegend: true,
});
const PLOT_CONFIG = { responsive: true, displayModeBar: true, scrollZoom: true };

const state = {
  view: "comparison",
  stepMax: DATA.global_step_max,
  logY: true,
};

function armColor(name) {
  const i = DATA.arms.indexOf(name);
  return COLORS[i % COLORS.length];
}

function selectedArms() {
  return DATA.arms.filter((a) => document.getElementById("cb-" + a).checked);
}

function clipSeries(steps, values, maxStep) {
  if (!steps || !steps.length) return [[], []];
  const xs = [], ys = [];
  for (let i = 0; i < steps.length; i++) {
    const x = steps[i];
    if (x == null || x > maxStep) continue;
    const y = values[i];
    xs.push(x);
    ys.push(y == null ? null : y);
  }
  return [xs, ys];
}

function traceLine(name, x, y, color, dash, width) {
  return {
    type: "scatter",
    mode: "lines",
    name,
    x,
    y,
    line: { color, width: width || 1.6, dash: dash || "solid" },
    connectgaps: false,
    hovertemplate: "%{fullData.name}<br>step=%{x}<br>y=%{y}<extra></extra>",
  };
}

function mountChart(el, traces, title, yTitle, logY) {
  const layout = Object.assign({}, PLOT_THEME(), {
    title: { text: title, font: { size: 13 } },
    yaxis: Object.assign({}, PLOT_THEME().yaxis, {
      title: yTitle,
      type: logY ? "log" : "linear",
    }),
  });
  Plotly.newPlot(el, traces, layout, PLOT_CONFIG);
  window.addEventListener("resize", () => Plotly.Plots.resize(el));
}

function renderComparison() {
  const grid = document.getElementById("grid-comparison");
  grid.innerHTML = "";
  const arms = selectedArms();
  const panels = [
    { split: "train", key: "loss", title: "Train loss", y: "loss" },
    { split: "train", key: "accuracy", title: "Train accuracy", y: "accuracy" },
    { split: "eval", key: "accuracy", title: "Validation accuracy", y: "accuracy" },
    { split: "eval", key: "auc", title: "Validation macro AUC", y: "AUC" },
    { split: "train", key: "lr", title: "Learning rate", y: "lr" },
    { split: "train", key: "jets_per_sec", title: "Throughput", y: "jets/s" },
  ];
  const hasCounts = arms.some((a) => (DATA.series[a].train.class_min || []).length);
  if (hasCounts) {
    panels.push({ split: "train", key: "class_counts", title: "Class balance (log window)", y: "jets" });
  }
  panels.forEach((p) => {
    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const el = document.createElement("div");
    el.className = "chart";
    wrap.appendChild(el);
    grid.appendChild(wrap);
    const traces = [];
    arms.forEach((arm) => {
      const s = DATA.series[arm];
      const color = armColor(arm);
      if (p.key === "class_counts") {
        const [xmin, ymin] = clipSeries(s.train.step, s.train.class_min, state.stepMax);
        const [, ymax] = clipSeries(s.train.step, s.train.class_max, state.stepMax);
        if (xmin.length) {
          traces.push(traceLine(arm + " min", xmin, ymin, color, "dot", 1.2));
          traces.push(traceLine(arm + " max", xmin, ymax, color, "dash", 1.2));
        }
        return;
      }
      const block = s[p.split];
      const [x, y] = clipSeries(block.step, block[p.key], state.stepMax);
      if (x.length) traces.push(traceLine(arm, x, y, color));
    });
    mountChart(el, traces, p.title, p.y, false);
  });
}

function renderPerArm() {
  const grid = document.getElementById("grid-per-arm");
  grid.innerHTML = "";
  const arm = document.getElementById("arm-select").value;
  const s = DATA.series[arm];
  const panels = [
    { x: s.train.step, y: s.train.loss, title: "Train loss", ylab: "loss" },
    { x: s.train.step, y: s.train.accuracy, title: "Train accuracy", ylab: "accuracy", extra: [
      { x: s.eval.step, y: s.eval.accuracy, name: "val", dash: "solid", mode: "lines+markers" }
    ]},
    { x: s.eval.step, y: s.eval.auc, title: "Validation macro AUC", ylab: "AUC", mode: "lines+markers" },
    { x: s.train.step, y: s.train.lr, title: "Learning rate", ylab: "lr" },
    { x: s.train.step, y: s.train.jets_per_sec, title: "Throughput", ylab: "jets/s" },
  ];
  if ((s.train.class_min || []).length) {
    panels.push({
      x: s.train.step, y: s.train.class_min, title: "Class balance (log window)", ylab: "jets",
      extra: [{ x: s.train.step, y: s.train.class_max, name: "max", dash: "dash" }],
    });
  }
  panels.forEach((p) => {
    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const el = document.createElement("div");
    el.className = "chart";
    wrap.appendChild(el);
    grid.appendChild(wrap);
    const traces = [];
    const [x, y] = clipSeries(p.x, p.y, state.stepMax);
    traces.push(traceLine(p.title.split(" ")[0].toLowerCase(), x, y, armColor(arm), "solid"));
    (p.extra || []).forEach((e) => {
      const [ex, ey] = clipSeries(e.x, e.y, state.stepMax);
      if (ex.length) {
        traces.push({
          type: "scatter",
          mode: e.mode || "lines",
          name: e.name,
          x: ex, y: ey,
          line: { color: armColor(arm), dash: e.dash || "solid", width: 1.6 },
          marker: { size: 4 },
          connectgaps: false,
        });
      }
    });
    mountChart(el, traces, p.title + " — " + arm, p.ylab, false);
  });
}

function renderRejection() {
  const grid = document.getElementById("grid-rejection");
  grid.innerHTML = "";
  const arms = selectedArms();
  // summary val accuracy
  {
    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    wrap.style.gridColumn = "1 / -1";
    const el = document.createElement("div");
    el.className = "chart";
    wrap.appendChild(el);
    grid.appendChild(wrap);
    const traces = arms.map((arm) => {
      const s = DATA.series[arm];
      const [x, y] = clipSeries(s.eval.step, s.eval.accuracy, state.stepMax);
      return traceLine(arm, x, y, armColor(arm));
    }).filter((t) => t.x.length);
    mountChart(el, traces, "Validation accuracy (summary)", "accuracy", false);
  }
  DATA.signal_classes.forEach((cls) => {
    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const el = document.createElement("div");
    el.className = "chart";
    wrap.appendChild(el);
    grid.appendChild(wrap);
    const traces = [];
    arms.forEach((arm) => {
      const s = DATA.series[arm];
      const rej50 = (s.eval.per_class_rej50 || {})[cls] || [];
      const rej99 = (s.eval.per_class_rej99 || {})[cls] || [];
      const color = armColor(arm);
      const [x50, y50] = clipSeries(s.eval.step, rej50, state.stepMax);
      const [x99, y99] = clipSeries(s.eval.step, rej99, state.stepMax);
      if (x50.length) traces.push(traceLine(arm + " rej@50%", x50, y50, color, "solid"));
      if (x99.length) traces.push(traceLine(arm + " rej@99%", x99, y99, color, "dash"));
    });
    mountChart(el, traces, cls + " rejection vs QCD", "Rej", state.logY);
  });
}

function render() {
  document.getElementById("grid-comparison").classList.toggle("hidden", state.view !== "comparison");
  document.getElementById("grid-per-arm").classList.toggle("hidden", state.view !== "per-arm");
  document.getElementById("grid-rejection").classList.toggle("hidden", state.view !== "rejection");
  document.getElementById("per-arm-row").classList.toggle("hidden", state.view !== "per-arm");
  document.getElementById("rej-row").classList.toggle("hidden", state.view !== "rejection");
  document.getElementById("arm-controls").classList.toggle("hidden", state.view === "per-arm");
  document.getElementById("arm-checkboxes").classList.toggle("hidden", state.view === "per-arm");
  if (state.view === "comparison") renderComparison();
  else if (state.view === "per-arm") renderPerArm();
  else renderRejection();
}

function buildArmCheckboxes() {
  const box = document.getElementById("arm-checkboxes");
  DATA.arms.forEach((arm) => {
    const id = "cb-" + arm;
    const label = document.createElement("label");
    label.innerHTML = '<input type="checkbox" id="' + id + '" checked> <span class="swatch" style="background:' + armColor(arm) + '"></span>' + arm;
    label.querySelector("input").addEventListener("change", render);
    box.appendChild(label);
  });
  const sel = document.getElementById("arm-select");
  DATA.arms.forEach((arm) => {
    const opt = document.createElement("option");
    opt.value = arm;
    opt.textContent = arm;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", render);
}

function setStepMax(v) {
  state.stepMax = Math.max(0, Math.min(v, DATA.global_step_max));
  document.getElementById("step-slider").value = state.stepMax;
  document.getElementById("step-input").value = state.stepMax;
  render();
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.view = btn.dataset.view;
    render();
  });
});

document.getElementById("btn-all").addEventListener("click", () => {
  DATA.arms.forEach((a) => { document.getElementById("cb-" + a).checked = true; });
  render();
});
document.getElementById("btn-none").addEventListener("click", () => {
  DATA.arms.forEach((a) => { document.getElementById("cb-" + a).checked = false; });
  render();
});
document.getElementById("btn-invert").addEventListener("click", () => {
  DATA.arms.forEach((a) => {
    const el = document.getElementById("cb-" + a);
    el.checked = !el.checked;
  });
  render();
});
document.getElementById("step-slider").addEventListener("input", (e) => setStepMax(Number(e.target.value)));
document.getElementById("step-input").addEventListener("change", (e) => setStepMax(Number(e.target.value)));
document.getElementById("btn-step-full").addEventListener("click", () => setStepMax(DATA.global_step_max));
document.getElementById("btn-step-200k").addEventListener("click", () => setStepMax(Math.min(200000, DATA.global_step_max)));
document.getElementById("log-y").addEventListener("change", (e) => {
  state.logY = e.target.checked;
  if (state.view === "rejection") renderRejection();
});

buildArmCheckboxes();
render();
</script>
</body>
</html>
"""


def _tolist(arr: np.ndarray) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for v in np.asarray(arr).flat:
        fv = float(v)
        if not math.isfinite(fv):
            out.append(None)
        else:
            out.append(fv)
    return out


def _serialize_block(block: dict) -> dict:
    out: Dict[str, Any] = {}
    for key, val in block.items():
        if isinstance(val, np.ndarray):
            out[key] = _tolist(val)
        elif isinstance(val, dict):
            out[key] = {k: _tolist(v) for k, v in val.items()}
        else:
            out[key] = val
    return out


def serialize_series(series: dict) -> dict:
    return {
        "run": series["run"],
        "arm": series.get("arm", series["run"]),
        "experiment": series.get("experiment"),
        "git": series.get("git"),
        "train": _serialize_block(series["train"]),
        "eval": _serialize_block(series["eval"]),
    }


def discover_all_series(runs_dir: Path) -> List[dict]:
    ordered: List[dict] = []
    if not runs_dir.is_dir():
        return ordered
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        series = load_series(run_dir)
        if series is not None:
            ordered.append(series)
    return ordered


def global_step_max(all_series: Sequence[dict]) -> int:
    mx = 0
    for series in all_series:
        for split in ("train", "eval"):
            steps = series[split]["step"]
            if steps.size:
                finite = steps[np.isfinite(steps)]
                if finite.size:
                    mx = max(mx, int(finite.max()))
    return mx


def build_payload(all_series: Sequence[dict]) -> dict:
    arms = [s["run"] for s in all_series]
    return {
        "arms": arms,
        "signal_classes": list(SIGNAL_CLASSES),
        "global_step_max": global_step_max(all_series),
        "series": {s["run"]: serialize_series(s) for s in all_series},
    }


def render_html(payload: dict) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    html = html.replace("__STEP_MAX__", str(payload["global_step_max"]))
    return html


def export_html(
    runs_dir: Path,
    out_path: Path,
    *,
    arms: Optional[Sequence[str]] = None,
) -> List[str]:
    if arms:
        all_series = []
        for name in arms:
            series = load_series(runs_dir / name)
            if series is not None:
                all_series.append(series)
    else:
        all_series = discover_all_series(runs_dir)

    if not all_series:
        raise SystemExit(f"No metrics found under {runs_dir}")

    payload = build_payload(all_series)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(payload), encoding="utf-8")
    return payload["arms"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--runs",
        default="part_ablation/runs",
        help="directory containing per-arm run folders",
    )
    parser.add_argument(
        "--out",
        default="logs/ablation-plots.html",
        help="output HTML path",
    )
    parser.add_argument(
        "--arm",
        action="append",
        dest="arms",
        help="include only these run folder names (repeatable)",
    )
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs)
    out_path = Path(args.out)
    included = export_html(runs_dir, out_path, arms=args.arms)
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({size_kb:.0f} KiB, {len(included)} arms)")
    print("arms:", ", ".join(included))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
