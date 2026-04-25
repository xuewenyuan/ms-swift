#!/usr/bin/env python3
"""Analyze whether channel-loss convergence correlates with adaptive leaf purity."""

import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from analyze_cluster_mining import UNKNOWN, safe_div, write_csv
from split_raw_by_adaptive_leaf import purity_token, sanitize_channel


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join TensorBoard channel loss curves with adaptive leaf purity and report correlations.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--tb-dir",
        help="TensorBoard log dir or event file. Recursively loads events.out.tfevents* when a directory is given.")
    input_group.add_argument(
        "--loss-csv",
        help="CSV with either tag/step/value columns or channel/step/loss columns.")
    parser.add_argument(
        "--split-manifest",
        required=True,
        help="split_manifest.csv from split_raw_by_adaptive_leaf.py. Usually generated with --group-by leaf.")
    parser.add_argument("--output-dir", required=True, help="Output directory for analysis CSV/HTML/Markdown.")
    parser.add_argument(
        "--channel-prefix",
        default="vla_adaptive",
        help="Channel prefix used by split_raw_by_adaptive_leaf.py --channel-prefix. Default: vla_adaptive.")
    parser.add_argument(
        "--tag-prefix",
        default="loss_",
        help="Scalar tag basename prefix for channel loss. Default: loss_.")
    parser.add_argument(
        "--mode-prefix",
        default="train/",
        help="Optional TensorBoard tag prefix to prefer, e.g. train/ or eval/. Empty means no mode filter. Default: train/.")
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="Minimum scalar points required for one channel to be analyzed. Default: 5.")
    parser.add_argument(
        "--window-points",
        type=int,
        default=5,
        help="Number of first/last scalar points used for convergence summary. Default: 5.")
    parser.add_argument(
        "--topn",
        type=int,
        default=30,
        help="Rows shown in Markdown/HTML diagnostic tables. Default: 30.")
    parser.add_argument(
        "--purity-bin-size",
        type=float,
        default=0.05,
        help="Purity bin size for trend charts in (0, 1]. Default: 0.05.")
    parser.add_argument(
        "--plotly-js",
        default=PLOTLY_CDN,
        help="Plotly.js URL or browser-visible local path for generated HTML. Default: Plotly CDN.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_meta = load_channel_meta(Path(args.split_manifest), args.channel_prefix)
    loss_points = load_loss_points(args)
    metrics = build_channel_metrics(loss_points, channel_meta, args.min_points, args.window_points)
    correlations = build_correlations(metrics)
    purity_bins = build_purity_bins(metrics, args.purity_bin_size)
    missing_rows = build_missing_rows(loss_points, channel_meta)

    write_csv(output_dir / "channel_loss_metrics.csv", metrics)
    write_csv(output_dir / "channel_loss_correlations.csv", correlations)
    write_csv(output_dir / "channel_loss_purity_bins.csv", purity_bins)
    write_csv(output_dir / "channel_loss_unmatched.csv", missing_rows)
    summary = build_summary(metrics, correlations, purity_bins, missing_rows, args)
    (output_dir / "channel_loss_purity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "channel_loss_purity_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    (output_dir / "channel_loss_purity_dashboard.html").write_text(render_html(summary, args.plotly_js),
                                                                    encoding="utf-8")
    print(f"wrote channel loss purity analysis -> {output_dir}")


def load_channel_meta(split_manifest: Path, channel_prefix: str) -> Dict[str, Dict[str, Any]]:
    rows = read_csv_dicts(split_manifest)
    by_channel: Dict[str, Dict[str, Any]] = {}
    by_leaf_id: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        leaf_ids = parse_leaf_ids(row.get("leaf_ids", ""))
        dominant_label = str(row.get("dominant_label") or UNKNOWN)
        purity = to_float(row.get("avg_leaf_purity"))
        count = to_int(row.get("count"))
        meta = {
            "group_key": row.get("group_key", ""),
            "file": row.get("file", ""),
            "count": count,
            "suggested_target_count": to_int(row.get("suggested_target_count")),
            "leaf_ids": " ".join(str(v) for v in leaf_ids),
            "leaf_id": leaf_ids[0] if len(leaf_ids) == 1 else None,
            "num_leaves": len(leaf_ids),
            "avg_leaf_purity": purity,
            "dominant_label": dominant_label,
            "dominant_label_ratio": to_float(row.get("dominant_label_ratio")),
            "primary_scene_top": row.get("primary_scene_top", ""),
            "secondary_scene_top": row.get("secondary_scene_top", ""),
            "suggested_action": row.get("suggested_action_top", ""),
            "stop_reason": row.get("stop_reason_top", ""),
        }
        channel = expected_channel(channel_prefix, leaf_ids, purity, dominant_label)
        by_channel[channel] = dict(meta, channel=channel)
        for leaf_id in leaf_ids:
            by_leaf_id[leaf_id] = dict(meta, channel=channel, leaf_id=leaf_id)
    for leaf_id, meta in by_leaf_id.items():
        by_channel[f"leaf:{leaf_id}"] = meta
    return by_channel


def expected_channel(channel_prefix: str, leaf_ids: Sequence[int], purity: float, dominant_label: str) -> str:
    if len(leaf_ids) == 1:
        return sanitize_channel(
            f"{channel_prefix}_leaf_{leaf_ids[0]:06d}_{purity_token(purity)}_{dominant_label}")
    return sanitize_channel(f"{channel_prefix}_leaves_{len(leaf_ids):04d}_{purity_token(purity)}_{dominant_label}")


def parse_leaf_ids(value: str) -> List[int]:
    ids = []
    for item in str(value or "").split():
        try:
            ids.append(int(float(item)))
        except ValueError:
            continue
    return ids


def load_loss_points(args: argparse.Namespace) -> Dict[str, List[Tuple[int, float]]]:
    if args.loss_csv:
        raw_points = load_loss_csv(Path(args.loss_csv))
    else:
        raw_points = load_tensorboard_scalars(Path(args.tb_dir))
    return filter_channel_scalars(raw_points, args.tag_prefix, args.mode_prefix)


def load_loss_csv(path: Path) -> Dict[str, List[Tuple[int, float]]]:
    rows = read_csv_dicts(path)
    points: DefaultDict[str, List[Tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if "channel" in row:
            tag = str(row.get("channel") or "")
            step = to_int(row.get("step"))
            value = to_float(row.get("loss"))
        else:
            tag = str(row.get("tag") or "")
            step = to_int(row.get("step"))
            value = to_float(row.get("value"))
        if tag:
            points[tag].append((step, value))
    return {tag: sorted(values) for tag, values in points.items()}


def load_tensorboard_scalars(path_value: str) -> Dict[str, List[Tuple[int, float]]]:
    event_paths = resolve_event_paths(Path(path_value))
    if not event_paths:
        raise FileNotFoundError(f"no TensorBoard event files found under {path_value}")
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "tensorboard is required to read event files. Install tensorboard or pass --loss-csv.") from e

    merged: DefaultDict[str, Dict[int, float]] = defaultdict(dict)
    for event_path in event_paths:
        accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception as e:
            print(f"warning: failed to read TensorBoard event file {event_path}: {e}")
            continue
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                merged[tag][int(event.step)] = float(event.value)
    return {tag: sorted(values.items()) for tag, values in merged.items()}


def resolve_event_paths(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("events.out.tfevents*") if p.is_file())


def filter_channel_scalars(
        raw_points: Dict[str, List[Tuple[int, float]]],
        tag_prefix: str,
        mode_prefix: str) -> Dict[str, List[Tuple[int, float]]]:
    points: Dict[str, List[Tuple[int, float]]] = {}
    for tag, values in raw_points.items():
        if mode_prefix and not tag.startswith(mode_prefix):
            continue
        basename = tag[len(mode_prefix):] if mode_prefix and tag.startswith(mode_prefix) else tag.rsplit("/", 1)[-1]
        if not basename.startswith(tag_prefix):
            continue
        channel = basename[len(tag_prefix):]
        points[channel] = sorted(values)
    if points or not mode_prefix:
        return points
    return filter_channel_scalars(raw_points, tag_prefix, "")


def build_channel_metrics(
        loss_points: Dict[str, List[Tuple[int, float]]],
        channel_meta: Dict[str, Dict[str, Any]],
        min_points: int,
        window_points: int) -> List[Dict[str, Any]]:
    rows = []
    for channel, points in sorted(loss_points.items()):
        if len(points) < min_points:
            continue
        meta = match_channel_meta(channel, channel_meta)
        if meta is None:
            continue
        clean_points = [(step, value) for step, value in points if math.isfinite(value)]
        if len(clean_points) < min_points:
            continue
        summary = summarize_curve(clean_points, window_points)
        rows.append({
            "channel": channel,
            "leaf_id": meta.get("leaf_id"),
            "leaf_ids": meta.get("leaf_ids", ""),
            "count": meta.get("count", 0),
            "suggested_target_count": meta.get("suggested_target_count", 0),
            "avg_leaf_purity": meta.get("avg_leaf_purity", 0.0),
            "dominant_label": meta.get("dominant_label", UNKNOWN),
            "dominant_label_ratio": meta.get("dominant_label_ratio", 0.0),
            "primary_scene_top": meta.get("primary_scene_top", ""),
            "secondary_scene_top": meta.get("secondary_scene_top", ""),
            "suggested_action": meta.get("suggested_action", ""),
            "stop_reason": meta.get("stop_reason", ""),
            **summary,
        })
    rows.sort(key=lambda row: (float(row["avg_leaf_purity"]), -float(row["final_loss"])))
    return rows


def match_channel_meta(channel: str, channel_meta: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if channel in channel_meta:
        return channel_meta[channel]
    leaf_id = parse_leaf_id_from_channel(channel)
    if leaf_id is not None:
        return channel_meta.get(f"leaf:{leaf_id}")
    return None


def parse_leaf_id_from_channel(channel: str) -> Optional[int]:
    match = re.search(r"(?:^|_)leaf_(\d+)(?:_|$)", channel)
    if not match:
        return None
    return int(match.group(1))


def summarize_curve(points: Sequence[Tuple[int, float]], window_points: int) -> Dict[str, Any]:
    points = sorted(points)
    window = max(1, min(window_points, len(points)))
    steps = [int(step) for step, _ in points]
    values = [float(value) for _, value in points]
    first_loss = mean(values[:window])
    final_loss = mean(values[-window:])
    min_loss = min(values)
    max_loss = max(values)
    loss_delta = final_loss - first_loss
    relative_loss_drop = safe_div(first_loss - final_loss, first_loss)
    slope_per_1k_steps = linear_slope(steps, values) * 1000.0
    tail_std = std(values[-window:])
    auc_mean_loss = mean(values)
    return {
        "num_points": len(points),
        "first_step": steps[0],
        "last_step": steps[-1],
        "first_loss": first_loss,
        "final_loss": final_loss,
        "min_loss": min_loss,
        "max_loss": max_loss,
        "loss_delta": loss_delta,
        "relative_loss_drop": relative_loss_drop,
        "slope_per_1k_steps": slope_per_1k_steps,
        "tail_std": tail_std,
        "auc_mean_loss": auc_mean_loss,
    }


def build_correlations(metrics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    x = [float(row["avg_leaf_purity"]) for row in metrics]
    specs = [
        ("final_loss", "lower_is_better"),
        ("relative_loss_drop", "higher_is_better"),
        ("slope_per_1k_steps", "lower_is_better"),
        ("tail_std", "lower_is_better"),
        ("auc_mean_loss", "lower_is_better"),
    ]
    rows = []
    for metric, direction in specs:
        y = [float(row[metric]) for row in metrics]
        rows.append({
            "x": "avg_leaf_purity",
            "y": metric,
            "better_direction": direction,
            "num_channels": len(metrics),
            "pearson": pearson(x, y),
            "spearman": spearman(x, y),
        })
    return rows


def build_missing_rows(loss_points: Dict[str, List[Tuple[int, float]]], channel_meta: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for channel, points in sorted(loss_points.items()):
        if match_channel_meta(channel, channel_meta) is None:
            rows.append({
                "channel": channel,
                "num_points": len(points),
                "parsed_leaf_id": parse_leaf_id_from_channel(channel),
            })
    return rows


def build_purity_bins(metrics: Sequence[Dict[str, Any]], bin_size: float) -> List[Dict[str, Any]]:
    if not 0 < bin_size <= 1:
        raise ValueError("--purity-bin-size must be in (0, 1].")
    rows = []
    num_bins = max(1, int(math.ceil(1.0 / bin_size)))
    for idx in range(num_bins):
        start = round(idx * bin_size, 10)
        end = round(min(1.0, (idx + 1) * bin_size), 10)
        bucket = [
            row for row in metrics
            if purity_in_bin(float(row["avg_leaf_purity"]), start, end)
        ]
        count_sum = sum(int(row["count"]) for row in bucket)
        rows.append({
            "purity_bin": format_purity_bin(start, end),
            "purity_lower": start,
            "purity_upper": end,
            "num_channels": len(bucket),
            "sample_count": count_sum,
            "mean_purity": mean([float(row["avg_leaf_purity"]) for row in bucket]),
            "mean_final_loss": mean([float(row["final_loss"]) for row in bucket]),
            "mean_relative_loss_drop": mean([float(row["relative_loss_drop"]) for row in bucket]),
            "mean_slope_per_1k_steps": mean([float(row["slope_per_1k_steps"]) for row in bucket]),
            "mean_tail_std": mean([float(row["tail_std"]) for row in bucket]),
            "mean_auc_loss": mean([float(row["auc_mean_loss"]) for row in bucket]),
            "weighted_final_loss": weighted_mean(bucket, "final_loss", "count"),
            "weighted_relative_loss_drop": weighted_mean(bucket, "relative_loss_drop", "count"),
        })
    return rows


def purity_in_bin(value: float, lower: float, upper: float) -> bool:
    if upper >= 1.0:
        return lower <= value <= upper
    return lower <= value < upper


def format_purity_bin(lower: float, upper: float) -> str:
    right = "]" if upper >= 1.0 else ")"
    return f"[{lower:.2f}, {upper:.2f}{right}"


def weighted_mean(rows: Sequence[Dict[str, Any]], value_key: str, weight_key: str) -> float:
    total_weight = sum(float(row[weight_key]) for row in rows)
    return safe_div(sum(float(row[value_key]) * float(row[weight_key]) for row in rows), total_weight)


def build_summary(
        metrics: Sequence[Dict[str, Any]],
        correlations: Sequence[Dict[str, Any]],
        purity_bins: Sequence[Dict[str, Any]],
        missing_rows: Sequence[Dict[str, Any]],
        args: argparse.Namespace) -> Dict[str, Any]:
    sorted_high_purity_hard = sorted(
        metrics,
        key=lambda row: (-float(row["avg_leaf_purity"]), -float(row["final_loss"])))[:args.topn]
    sorted_low_purity_easy = sorted(
        metrics,
        key=lambda row: (float(row["avg_leaf_purity"]), float(row["final_loss"])))[:args.topn]
    return {
        "num_channels_analyzed": len(metrics),
        "num_unmatched_channels": len(missing_rows),
        "channel_prefix": args.channel_prefix,
        "tag_prefix": args.tag_prefix,
        "mode_prefix": args.mode_prefix,
        "correlations": list(correlations),
        "purity_bins": list(purity_bins),
        "high_purity_high_final_loss": slim_rows(sorted_high_purity_hard),
        "low_purity_low_final_loss": slim_rows(sorted_low_purity_easy),
        "metrics": list(metrics),
        "outputs": [
            "channel_loss_metrics.csv",
            "channel_loss_correlations.csv",
            "channel_loss_purity_bins.csv",
            "channel_loss_unmatched.csv",
            "channel_loss_purity_summary.json",
            "channel_loss_purity_summary.md",
            "channel_loss_purity_dashboard.html",
        ],
    }


def slim_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = [
        "channel",
        "leaf_id",
        "count",
        "avg_leaf_purity",
        "dominant_label",
        "first_loss",
        "final_loss",
        "relative_loss_drop",
        "slope_per_1k_steps",
    ]
    return [{key: row.get(key) for key in keys} for row in rows]


def render_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Channel Loss vs Adaptive Purity",
        "",
        f"- Channels analyzed: {summary['num_channels_analyzed']}",
        f"- Unmatched channels: {summary['num_unmatched_channels']}",
        f"- Channel prefix: `{summary['channel_prefix']}`",
        f"- Tag prefix: `{summary['tag_prefix']}`",
        f"- Mode prefix: `{summary['mode_prefix']}`",
        "",
        "## Correlations",
        "",
        "| x | y | direction | pearson | spearman | channels |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["correlations"]:
        lines.append(
            f"| {row['x']} | {row['y']} | {row['better_direction']} | "
            f"{float(row['pearson']):.4f} | {float(row['spearman']):.4f} | {row['num_channels']} |")
    lines.extend(["", "## Purity Bin Trends", ""])
    lines.append("| purity_bin | channels | samples | mean_final_loss | mean_rel_drop | weighted_final_loss | weighted_rel_drop |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["purity_bins"]:
        if int(row["num_channels"]) == 0:
            continue
        lines.append(
            f"| {row['purity_bin']} | {row['num_channels']} | {row['sample_count']} | "
            f"{float(row['mean_final_loss']):.4f} | {float(row['mean_relative_loss_drop']):.4f} | "
            f"{float(row['weighted_final_loss']):.4f} | {float(row['weighted_relative_loss_drop']):.4f} |")
    lines.extend(["", "## High Purity But High Final Loss", ""])
    lines.extend(render_rows(summary["high_purity_high_final_loss"]))
    lines.extend(["", "## Low Purity But Low Final Loss", ""])
    lines.extend(render_rows(summary["low_purity_low_final_loss"]))
    lines.append("")
    return "\n".join(lines)


def render_rows(rows: Sequence[Dict[str, Any]]) -> List[str]:
    if not rows:
        return ["- No rows."]
    lines = [
        "| channel | purity | count | first_loss | final_loss | rel_drop | slope_per_1k |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['channel']}` | {float(row['avg_leaf_purity']):.4f} | {row['count']} | "
            f"{float(row['first_loss']):.4f} | {float(row['final_loss']):.4f} | "
            f"{float(row['relative_loss_drop']):.4f} | {float(row['slope_per_1k_steps']):.4f} |")
    return lines


def render_html(summary: Dict[str, Any], plotly_js: str) -> str:
    payload = json.dumps(summary, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Channel Loss vs Adaptive Purity</title>
  <script src="{html.escape(plotly_js, quote=True)}"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #17212b; }}
    .page {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .card {{ background: white; border: 1px solid #dde3ea; border-radius: 8px; padding: 14px 16px; }}
    .label {{ color: #5d6b7a; font-size: 12px; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 16px; }}
    .panel {{ background: white; border: 1px solid #dde3ea; border-radius: 8px; padding: 14px; }}
    .chart {{ height: 420px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ padding: 7px 8px; border-bottom: 1px solid #e6ebf0; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    h1 {{ font-size: 22px; margin: 0 0 16px; }}
    h2 {{ font-size: 15px; margin: 0 0 10px; }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Channel Loss vs Adaptive Purity</h1>
    <div class="cards">
      <div class="card"><div class="label">Channels</div><div class="value" id="channels"></div></div>
      <div class="card"><div class="label">Unmatched</div><div class="value" id="unmatched"></div></div>
      <div class="card"><div class="label">Final Loss Pearson</div><div class="value" id="pearson-final"></div></div>
      <div class="card"><div class="label">Rel Drop Pearson</div><div class="value" id="pearson-drop"></div></div>
    </div>
    <div class="grid">
      <div class="panel"><h2>Purity Bin Trend</h2><div id="bin-trend" class="chart"></div></div>
      <div class="panel"><h2>Purity Bin Volume</h2><div id="bin-volume" class="chart"></div></div>
      <div class="panel"><h2>Purity vs Final Loss</h2><div id="final-loss" class="chart"></div></div>
      <div class="panel"><h2>Purity vs Relative Loss Drop</h2><div id="rel-drop" class="chart"></div></div>
      <div class="panel"><h2>Purity vs Slope Per 1k Steps</h2><div id="slope" class="chart"></div></div>
      <div class="panel"><h2>High Purity But High Final Loss</h2><div id="hard-table"></div></div>
    </div>
  </div>
  <script>
    const data = {payload};
    const metrics = data.metrics;
    document.getElementById("channels").textContent = data.num_channels_analyzed;
    document.getElementById("unmatched").textContent = data.num_unmatched_channels;
    function corr(name) {{
      const row = data.correlations.find(r => r.y === name);
      return row ? Number(row.pearson).toFixed(3) : "-";
    }}
    document.getElementById("pearson-final").textContent = corr("final_loss");
    document.getElementById("pearson-drop").textContent = corr("relative_loss_drop");
    const bins = data.purity_bins.filter(r => r.num_channels > 0);
    Plotly.newPlot("bin-trend", [
      {{
        type: "scatter",
        mode: "lines+markers",
        name: "mean final loss",
        x: bins.map(r => r.purity_bin),
        y: bins.map(r => r.mean_final_loss),
        yaxis: "y1",
        line: {{ color: "#4C78A8", width: 3 }}
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        name: "mean relative drop",
        x: bins.map(r => r.purity_bin),
        y: bins.map(r => r.mean_relative_loss_drop),
        yaxis: "y2",
        line: {{ color: "#F58518", width: 3 }}
      }}
    ], {{
      margin: {{ l: 58, r: 58, t: 16, b: 88 }},
      xaxis: {{ title: "purity bin", tickangle: -35 }},
      yaxis: {{ title: "mean final loss" }},
      yaxis2: {{ title: "mean relative drop", overlaying: "y", side: "right" }},
      legend: {{ orientation: "h", y: 1.12 }}
    }}, {{ responsive: true }});
    Plotly.newPlot("bin-volume", [
      {{
        type: "bar",
        name: "channels",
        x: bins.map(r => r.purity_bin),
        y: bins.map(r => r.num_channels),
        marker: {{ color: "#54A24B" }}
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        name: "sample count",
        x: bins.map(r => r.purity_bin),
        y: bins.map(r => r.sample_count),
        yaxis: "y2",
        line: {{ color: "#E45756", width: 3 }}
      }}
    ], {{
      margin: {{ l: 58, r: 58, t: 16, b: 88 }},
      xaxis: {{ title: "purity bin", tickangle: -35 }},
      yaxis: {{ title: "channels" }},
      yaxis2: {{ title: "sample count", overlaying: "y", side: "right" }},
      legend: {{ orientation: "h", y: 1.12 }}
    }}, {{ responsive: true }});
    function scatter(div, y, title) {{
      Plotly.newPlot(div, [{{
        type: "scatter",
        mode: "markers",
        x: metrics.map(r => r.avg_leaf_purity),
        y: metrics.map(r => r[y]),
        text: metrics.map(r => `${{r.channel}}<br>label=${{r.dominant_label}}<br>count=${{r.count}}`),
        marker: {{ size: 8, color: metrics.map(r => r.count), colorscale: "Viridis", showscale: true }}
      }}], {{
        margin: {{ l: 58, r: 24, t: 16, b: 48 }},
        xaxis: {{ title: "avg_leaf_purity" }},
        yaxis: {{ title }}
      }}, {{ responsive: true }});
    }}
    scatter("final-loss", "final_loss", "final_loss");
    scatter("rel-drop", "relative_loss_drop", "relative_loss_drop");
    scatter("slope", "slope_per_1k_steps", "slope_per_1k_steps");
    function table(rows) {{
      if (!rows.length) return "<p>No rows.</p>";
      return `<table><thead><tr><th>channel</th><th>purity</th><th>count</th><th>first</th><th>final</th><th>rel_drop</th></tr></thead><tbody>` +
        rows.map(r => `<tr><td>${{r.channel}}</td><td>${{Number(r.avg_leaf_purity).toFixed(4)}}</td><td>${{r.count}}</td><td>${{Number(r.first_loss).toFixed(4)}}</td><td>${{Number(r.final_loss).toFixed(4)}}</td><td>${{Number(r.relative_loss_drop).toFixed(4)}}</td></tr>`).join("") +
        `</tbody></table>`;
    }}
    document.getElementById("hard-table").innerHTML = table(data.high_purity_high_final_loss);
  </script>
</body>
</html>
"""


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(values: Sequence[float]) -> float:
    return safe_div(sum(values), len(values))


def std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return math.sqrt(mean([(value - mu) ** 2 for value in values]))


def linear_slope(xs: Sequence[int], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_mean = mean([float(v) for v in xs])
    y_mean = mean(ys)
    numerator = sum((float(x) - x_mean) * (float(y) - y_mean) for x, y in zip(xs, ys))
    denominator = sum((float(x) - x_mean) ** 2 for x in xs)
    return safe_div(numerator, denominator)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return 0.0
    px, py = zip(*pairs)
    x_mean = mean(px)
    y_mean = mean(py)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in px))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in py))
    return safe_div(numerator, x_den * y_den)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return 0.0
    px, py = zip(*pairs)
    return pearson(ranks(px), ranks(py))


def ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + j + 1) / 2.0
        for k in range(i, j):
            result[indexed[k][0]] = rank
        i = j
    return result


def to_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


if __name__ == "__main__":
    main()
