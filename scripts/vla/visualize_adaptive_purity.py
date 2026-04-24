#!/usr/bin/env python3
"""Render a static HTML dashboard for adaptive leaf purity distributions."""

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
ACTION_COLORS = {
    "downsample_frequent": "#4C78A8",
    "keep_tail": "#F58518",
    "review_first": "#E45756",
    "balanced_keep": "#54A24B",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a static HTML dashboard for adaptive leaf purity summaries.")
    parser.add_argument("--adaptive-dir", required=True, help="Directory produced by adaptive_cluster_mining.py.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.05,
        help="Purity bin size in (0, 1]. Default: 0.05.")
    parser.add_argument("--topn", type=int, default=30, help="Top leaves shown in tables. Default: 30.")
    parser.add_argument(
        "--plotly-js",
        default=PLOTLY_CDN,
        help="Plotly.js URL or local browser path used by the HTML. Default: Plotly CDN.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    adaptive_dir = Path(args.adaptive_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    leaf_rows = read_csv_dicts(adaptive_dir / "adaptive_leaf_summary.csv")
    summary = read_json_if_exists(adaptive_dir / "adaptive_summary.json")
    data = build_dashboard_data(leaf_rows, summary, args.bin_size, args.topn)
    output.write_text(render_html(data, args.plotly_js), encoding="utf-8")
    print(f"wrote adaptive purity dashboard -> {output}")


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.bin_size <= 1:
        raise ValueError("--bin-size must be in (0, 1].")


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dashboard_data(
        leaf_rows: Sequence[Dict[str, str]],
        summary: Dict[str, Any],
        bin_size: float,
        topn: int) -> Dict[str, Any]:
    bins = build_purity_bins(bin_size)
    bucketed = bucket_leaf_rows(leaf_rows, bins)
    cards = build_cards(leaf_rows, summary)
    return {
        "cards": cards,
        "purity_bins": build_purity_bin_chart(bucketed),
        "scatter": build_scatter(leaf_rows),
        "action_stack": build_action_stack(bucketed),
        "largest_low_purity": build_extreme_table(leaf_rows, topn, mode="largest_low_purity"),
        "highest_purity_large": build_extreme_table(leaf_rows, topn, mode="highest_purity_large"),
    }


def build_purity_bins(bin_size: float) -> List[Tuple[float, float]]:
    bins = []
    start = 0.0
    while start < 1.0:
        end = min(1.0, start + bin_size)
        bins.append((start, end))
        start = end
    return bins


def bucket_leaf_rows(
        leaf_rows: Sequence[Dict[str, str]],
        bins: Sequence[Tuple[float, float]]) -> List[Dict[str, Any]]:
    bucketed = []
    for lower, upper in bins:
        bucket_rows = []
        for row in leaf_rows:
            purity = to_float(row.get("purity"))
            if in_bin(purity, lower, upper):
                bucket_rows.append(row)
        action_counts: Counter = Counter(row.get("suggested_action", "UNKNOWN") for row in bucket_rows)
        bucketed.append({
            "label": format_bin(lower, upper),
            "lower": lower,
            "upper": upper,
            "leaf_count": len(bucket_rows),
            "row_count": sum(to_int(row.get("size")) for row in bucket_rows),
            "target_count": sum(to_int(row.get("suggested_target_count")) for row in bucket_rows),
            "action_counts": dict(action_counts),
        })
    return bucketed


def in_bin(value: float, lower: float, upper: float) -> bool:
    if upper >= 1.0:
        return lower <= value <= upper
    return lower <= value < upper


def format_bin(lower: float, upper: float) -> str:
    right = "]" if upper >= 1.0 else ")"
    return f"[{lower:.2f}, {upper:.2f}{right}"


def build_cards(leaf_rows: Sequence[Dict[str, str]], summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"name": "Leaves", "value": summary.get("num_leaves", len(leaf_rows))},
        {"name": "Rows", "value": summary.get("num_rows", sum(to_int(row.get("size")) for row in leaf_rows))},
        {
            "name": "Suggested Target",
            "value": summary.get("target_total", sum(to_int(row.get("suggested_target_count")) for row in leaf_rows))
        },
        {
            "name": "Avg Purity",
            "value": format_float(safe_div(sum(to_float(row.get("purity")) for row in leaf_rows), len(leaf_rows)))
        },
        {
            "name": "Pure Coverage",
            "value": format_percent(summary.get("pure_coverage_ratio"))
            if "pure_coverage_ratio" in summary else "-"
        },
    ]


def build_purity_bin_chart(bucketed: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "x": [row["label"] for row in bucketed],
        "leaf_count": [row["leaf_count"] for row in bucketed],
        "row_count": [row["row_count"] for row in bucketed],
        "target_count": [row["target_count"] for row in bucketed],
    }


def build_scatter(leaf_rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    rows = []
    for row in leaf_rows:
        action = row.get("suggested_action", "UNKNOWN")
        size = to_int(row.get("size"))
        purity = to_float(row.get("purity"))
        rows.append({
            "x": purity,
            "y": size,
            "color": ACTION_COLORS.get(action, "#72B7B2"),
            "action": action,
            "text": (
                f"leaf={row.get('leaf_id')}<br>"
                f"size={size}<br>"
                f"purity={purity:.4f}<br>"
                f"dominant={html.escape(str(row.get('dominant_label', 'UNKNOWN')))}<br>"
                f"target={to_int(row.get('suggested_target_count'))}<br>"
                f"action={action}<br>"
                f"stop_reason={row.get('stop_reason', 'UNKNOWN')}"
            ),
        })
    return rows


def build_action_stack(bucketed: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    actions = sorted({action for row in bucketed for action in row["action_counts"]})
    return {
        "x": [row["label"] for row in bucketed],
        "actions": actions,
        "y": {
            action: [row["action_counts"].get(action, 0) for row in bucketed]
            for action in actions
        },
    }


def build_extreme_table(leaf_rows: Sequence[Dict[str, str]], topn: int, mode: str) -> List[Dict[str, Any]]:
    rows = [normalize_leaf_row(row) for row in leaf_rows]
    if mode == "largest_low_purity":
        rows.sort(key=lambda row: (row["purity"], -row["size"]))
    elif mode == "highest_purity_large":
        rows.sort(key=lambda row: (-row["purity"], -row["size"]))
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return rows[:topn]


def normalize_leaf_row(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "leaf_id": to_int(row.get("leaf_id")),
        "size": to_int(row.get("size")),
        "purity": round(to_float(row.get("purity")), 6),
        "target": to_int(row.get("suggested_target_count")),
        "dominant_label": row.get("dominant_label", "UNKNOWN"),
        "suggested_action": row.get("suggested_action", "UNKNOWN"),
        "stop_reason": row.get("stop_reason", "UNKNOWN"),
        "primary_scene_top": row.get("primary_scene_top", ""),
        "secondary_scene_top": row.get("secondary_scene_top", ""),
    }


def render_html(data: Dict[str, Any], plotly_js: str) -> str:
    cards_html = "".join(
        f'<div class="card"><div class="card-name">{html.escape(str(card["name"]))}</div>'
        f'<div class="card-value">{html.escape(str(card["value"]))}</div></div>'
        for card in data["cards"])
    low_purity_table = render_table(
        data["largest_low_purity"],
        ["leaf_id", "size", "purity", "target", "dominant_label", "suggested_action", "stop_reason"])
    high_purity_table = render_table(
        data["highest_purity_large"],
        ["leaf_id", "size", "purity", "target", "dominant_label", "suggested_action", "stop_reason"])
    payload = json.dumps(data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Adaptive Purity Dashboard</title>
  <script src="{html.escape(plotly_js, quote=True)}"></script>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7fb;
      color: #16202a;
    }}
    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #d7dde7;
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .card-name {{
      font-size: 12px;
      color: #607080;
      margin-bottom: 6px;
    }}
    .card-value {{
      font-size: 26px;
      font-weight: 600;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .panel {{
      background: #fff;
      border: 1px solid #d7dde7;
      border-radius: 8px;
      padding: 12px;
    }}
    .panel.full {{
      grid-column: 1 / -1;
    }}
    .panel h2 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .plot {{
      width: 100%;
      height: 420px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid #e6ebf2;
      text-align: left;
      padding: 6px 8px;
      vertical-align: top;
    }}
    th {{
      color: #52606d;
      background: #f8fafc;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="cards">{cards_html}</div>
    <div class="grid">
      <section class="panel full">
        <h2>Purity Bin Distribution</h2>
        <div id="purity-bin-chart" class="plot"></div>
      </section>
      <section class="panel">
        <h2>Leaf Count by Action in Purity Bins</h2>
        <div id="action-stack-chart" class="plot"></div>
      </section>
      <section class="panel">
        <h2>Leaf Purity vs Size</h2>
        <div id="scatter-chart" class="plot"></div>
      </section>
      <section class="panel">
        <h2>Largest Low-Purity Leaves</h2>
        {low_purity_table}
      </section>
      <section class="panel">
        <h2>Highest-Purity Large Leaves</h2>
        {high_purity_table}
      </section>
    </div>
  </div>
  <script>
    const data = {payload};

    Plotly.newPlot('purity-bin-chart', [
      {{
        type: 'bar',
        name: 'leaf_count',
        x: data.purity_bins.x,
        y: data.purity_bins.leaf_count,
        marker: {{ color: '#4C78A8' }}
      }},
      {{
        type: 'bar',
        name: 'row_count',
        x: data.purity_bins.x,
        y: data.purity_bins.row_count,
        marker: {{ color: '#F58518' }},
        yaxis: 'y2'
      }},
      {{
        type: 'scatter',
        mode: 'lines+markers',
        name: 'target_count',
        x: data.purity_bins.x,
        y: data.purity_bins.target_count,
        marker: {{ color: '#54A24B' }},
        line: {{ color: '#54A24B' }},
        yaxis: 'y2'
      }}
    ], {{
      barmode: 'group',
      margin: {{ l: 56, r: 56, t: 20, b: 80 }},
      xaxis: {{ title: 'purity bin', tickangle: -35 }},
      yaxis: {{ title: 'leaf count' }},
      yaxis2: {{
        title: 'row / target count',
        overlaying: 'y',
        side: 'right',
        showgrid: false
      }},
      legend: {{ orientation: 'h', y: 1.12 }}
    }}, {{ responsive: true }});

    const actionTraces = data.action_stack.actions.map((action) => ({{
      type: 'bar',
      name: action,
      x: data.action_stack.x,
      y: data.action_stack.y[action],
      marker: {{ color: {json.dumps(ACTION_COLORS)}[action] || '#72B7B2' }}
    }}));
    Plotly.newPlot('action-stack-chart', actionTraces, {{
      barmode: 'stack',
      margin: {{ l: 56, r: 24, t: 20, b: 80 }},
      xaxis: {{ title: 'purity bin', tickangle: -35 }},
      yaxis: {{ title: 'leaf count' }},
      legend: {{ orientation: 'h', y: 1.12 }}
    }}, {{ responsive: true }});

    Plotly.newPlot('scatter-chart', [{{
      type: 'scattergl',
      mode: 'markers',
      x: data.scatter.map((row) => row.x),
      y: data.scatter.map((row) => row.y),
      text: data.scatter.map((row) => row.text),
      hovertemplate: '%{{text}}<extra></extra>',
      marker: {{
        color: data.scatter.map((row) => row.color),
        size: 8,
        opacity: 0.75
      }}
    }}], {{
      margin: {{ l: 56, r: 24, t: 20, b: 56 }},
      xaxis: {{ title: 'purity', range: [0, 1] }},
      yaxis: {{ title: 'leaf size', type: 'log' }}
    }}, {{ responsive: true }});
  </script>
</body>
</html>
"""


def render_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def to_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(round(float(value)))


def to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def format_float(value: float) -> str:
    return f"{value:.4f}"


def format_percent(value: Any) -> str:
    if value in (None, ""):
        return "-"
    return f"{100.0 * float(value):.2f}%"


if __name__ == "__main__":
    main()
