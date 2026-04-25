#!/usr/bin/env python3
"""Compare two adaptive_cluster_mining report directories."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence

from analyze_cluster_mining import UNKNOWN, iter_jsonl, read_json, safe_div, write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two adaptive_cluster_mining outputs and summarize which one is cleaner.")
    parser.add_argument("--old-report", required=True, help="Baseline adaptive report directory.")
    parser.add_argument("--new-report", required=True, help="New adaptive report directory.")
    parser.add_argument("--output-dir", required=True, help="Output dir for comparison artifacts.")
    parser.add_argument("--old-name", default="old", help="Display name for --old-report. Default: old.")
    parser.add_argument("--new-name", default="new", help="Display name for --new-report. Default: new.")
    parser.add_argument(
        "--purity-threshold",
        type=float,
        default=None,
        help="Purity threshold used for comparison. Default: inferred from adaptive_summary.json.")
    parser.add_argument(
        "--high-purity-threshold",
        type=float,
        default=0.9,
        help="Threshold for counting highly pure rows/leaves. Default: 0.9.")
    parser.add_argument(
        "--low-purity-threshold",
        type=float,
        default=0.5,
        help="Threshold for counting low-purity rows/leaves. Default: 0.5.")
    parser.add_argument("--topn", type=int, default=20, help="Top improved/worsened groups in Markdown. Default: 20.")
    parser.add_argument(
        "--min-group-rows",
        type=int,
        default=200,
        help="Minimum max(old_rows, new_rows) for improved/worsened group ranking. Default: 200.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    old_report = load_report(Path(args.old_report), args.old_name)
    new_report = load_report(Path(args.new_report), args.new_name)
    purity_threshold = resolve_purity_threshold(old_report, new_report, args.purity_threshold)

    old_metrics = build_report_metrics(old_report, purity_threshold, args.high_purity_threshold, args.low_purity_threshold)
    new_metrics = build_report_metrics(new_report, purity_threshold, args.high_purity_threshold, args.low_purity_threshold)

    overview_rows = build_overview_rows(old_metrics, new_metrics, args.old_name, args.new_name)
    label_rows = build_group_comparison_rows(
        old_metrics["label_metrics"], new_metrics["label_metrics"], "label", args.old_name, args.new_name)
    primary_scene_rows = build_group_comparison_rows(
        old_metrics["primary_scene_metrics"], new_metrics["primary_scene_metrics"], "primary_scene", args.old_name,
        args.new_name)

    write_csv(output_dir / "compare_overview.csv", overview_rows)
    write_csv(output_dir / "compare_label_metrics.csv", label_rows)
    write_csv(output_dir / "compare_primary_scene_metrics.csv", primary_scene_rows)

    summary = build_summary(
        old_metrics,
        new_metrics,
        overview_rows,
        label_rows,
        primary_scene_rows,
        args,
        purity_threshold,
    )
    (output_dir / "compare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "compare_summary.md").write_text(render_markdown(summary, args.old_name, args.new_name),
                                                    encoding="utf-8")
    print(f"wrote adaptive comparison artifacts -> {output_dir}")


def load_report(report_dir: Path, report_name: str) -> Dict[str, Any]:
    leaf_summary_path = report_dir / "adaptive_leaf_summary.csv"
    assignments_path = report_dir / "adaptive_leaf_assignments.jsonl"
    summary_path = report_dir / "adaptive_summary.json"
    review_path = report_dir / "adaptive_review_candidates.jsonl"
    if not leaf_summary_path.is_file():
        raise FileNotFoundError(f"adaptive_leaf_summary.csv not found: {leaf_summary_path}")
    if not assignments_path.is_file():
        raise FileNotFoundError(f"adaptive_leaf_assignments.jsonl not found: {assignments_path}")

    return {
        "name": report_name,
        "report_dir": str(report_dir),
        "leaf_rows": read_csv_dicts(leaf_summary_path),
        "assignments": list(iter_jsonl(assignments_path)),
        "review_rows": list(iter_jsonl(review_path)) if review_path.is_file() else [],
        "summary": read_json(summary_path) if summary_path.is_file() else {},
        "review_counts": count_review_candidates(review_path) if review_path.is_file() else {},
    }


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def count_review_candidates(path: Path) -> Dict[str, int]:
    counts: Counter = Counter()
    for row in iter_jsonl(path):
        counts[str(row.get("candidate_type") or UNKNOWN)] += 1
    return dict(counts)


def resolve_purity_threshold(old_report: Dict[str, Any], new_report: Dict[str, Any], cli_value: Optional[float]) -> float:
    if cli_value is not None:
        return float(cli_value)
    old_value = to_optional_float(old_report["summary"].get("purity_threshold"))
    new_value = to_optional_float(new_report["summary"].get("purity_threshold"))
    if old_value is not None and new_value is not None and abs(old_value - new_value) > 1e-9:
        raise ValueError(
            f"report purity thresholds differ: {old_value} vs {new_value}. Please pass --purity-threshold explicitly.")
    if old_value is not None:
        return old_value
    if new_value is not None:
        return new_value
    return 0.7


def build_report_metrics(
        report: Dict[str, Any],
        purity_threshold: float,
        high_purity_threshold: float,
        low_purity_threshold: float) -> Dict[str, Any]:
    leaf_rows = [normalize_leaf_row(row) for row in report["leaf_rows"]]
    assignments = [normalize_assignment(row) for row in report["assignments"]]
    summary = report["summary"]

    num_rows = sum(row["size"] for row in leaf_rows)
    num_leaves = len(leaf_rows)
    weighted_avg_purity = safe_div(sum(row["size"] * row["purity"] for row in leaf_rows), num_rows)
    macro_avg_purity = safe_div(sum(row["purity"] for row in leaf_rows), num_leaves)
    rows_ge_purity = sum(row["size"] for row in leaf_rows if row["purity"] >= purity_threshold)
    rows_ge_high = sum(row["size"] for row in leaf_rows if row["purity"] >= high_purity_threshold)
    rows_lt_low = sum(row["size"] for row in leaf_rows if row["purity"] < low_purity_threshold)
    impure_leaf_count = sum(1 for row in leaf_rows if row["purity"] < purity_threshold)
    impure_row_count = sum(row["size"] for row in leaf_rows if row["purity"] < purity_threshold)
    review_first_leaf_count = sum(1 for row in leaf_rows if row["suggested_action"] == "review_first")
    review_first_row_count = sum(row["size"] for row in leaf_rows if row["suggested_action"] == "review_first")
    suggested_total = sum(row["suggested_target_count"] for row in leaf_rows)
    keep_ratio = safe_div(suggested_total, num_rows)

    review_counts = dict(report["review_counts"])
    if not review_counts and isinstance(summary.get("review_candidate_counts"), dict):
        review_counts = {str(k): int(v) for k, v in summary["review_candidate_counts"].items()}

    label_review_counts = aggregate_review_counts(report["review_rows"], "label")
    primary_scene_review_counts = aggregate_review_counts(report["review_rows"], "primary_scene")

    return {
        "name": report["name"],
        "report_dir": report["report_dir"],
        "purity_threshold": purity_threshold,
        "high_purity_threshold": high_purity_threshold,
        "low_purity_threshold": low_purity_threshold,
        "num_rows": num_rows,
        "num_leaves": num_leaves,
        "macro_avg_purity": macro_avg_purity,
        "weighted_avg_purity": weighted_avg_purity,
        "rows_ge_purity": rows_ge_purity,
        "rows_ge_purity_ratio": safe_div(rows_ge_purity, num_rows),
        "rows_ge_high_purity": rows_ge_high,
        "rows_ge_high_purity_ratio": safe_div(rows_ge_high, num_rows),
        "rows_lt_low_purity": rows_lt_low,
        "rows_lt_low_purity_ratio": safe_div(rows_lt_low, num_rows),
        "impure_leaf_count": impure_leaf_count,
        "impure_row_count": impure_row_count,
        "impure_row_ratio": safe_div(impure_row_count, num_rows),
        "review_first_leaf_count": review_first_leaf_count,
        "review_first_row_count": review_first_row_count,
        "review_first_row_ratio": safe_div(review_first_row_count, num_rows),
        "suggested_total": suggested_total,
        "keep_ratio": keep_ratio,
        "review_candidate_total": sum(review_counts.values()),
        "review_candidate_counts": review_counts,
        "stop_reason_counts": dict(Counter(row["stop_reason"] for row in leaf_rows)),
        "action_counts": dict(Counter(row["suggested_action"] for row in leaf_rows)),
        "label_metrics": build_group_metrics(assignments, "label", purity_threshold, high_purity_threshold,
                                              low_purity_threshold, label_review_counts),
        "primary_scene_metrics": build_group_metrics(assignments, "primary_scene", purity_threshold,
                                                     high_purity_threshold, low_purity_threshold,
                                                     primary_scene_review_counts),
    }


def normalize_leaf_row(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "leaf_id": to_int(row.get("leaf_id")),
        "size": to_int(row.get("size")),
        "purity": to_float(row.get("purity")),
        "dominant_label": str(row.get("dominant_label") or UNKNOWN),
        "suggested_action": str(row.get("suggested_action") or UNKNOWN),
        "stop_reason": str(row.get("stop_reason") or UNKNOWN),
        "suggested_target_count": to_int(row.get("suggested_target_count")),
    }


def normalize_assignment(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": str(row.get("label") or UNKNOWN),
        "primary_scene": str(row.get("primary_scene") or UNKNOWN),
        "leaf_id": to_int(row.get("leaf_id")),
        "leaf_purity": to_float(row.get("leaf_purity")),
        "dominant_label": str(row.get("dominant_label") or UNKNOWN),
        "is_dominant_label": bool(row.get("is_dominant_label")),
    }


def aggregate_review_counts(review_rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    if not review_rows:
        return {}
    counts: Counter = Counter()
    for row in review_rows:
        counts[str(row.get(key) or UNKNOWN)] += 1
    return dict(counts)


def build_group_metrics(
        assignments: Sequence[Dict[str, Any]],
        key: str,
        purity_threshold: float,
        high_purity_threshold: float,
        low_purity_threshold: float,
        review_counts: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
    stats: DefaultDict[str, Dict[str, Any]] = defaultdict(new_group_stats)
    for row in assignments:
        group_value = str(row.get(key) or UNKNOWN)
        group = stats[group_value]
        leaf_id = int(row["leaf_id"])
        purity = float(row["leaf_purity"])
        dominant_label = str(row["dominant_label"])

        group["rows"] += 1
        group["leaf_purity_sum"] += purity
        group["leaf_ids"].add(leaf_id)
        if purity >= purity_threshold:
            group["rows_ge_purity"] += 1
            group["high_enough_leaf_ids"].add(leaf_id)
        if purity >= high_purity_threshold:
            group["rows_ge_high_purity"] += 1
            group["high_purity_leaf_ids"].add(leaf_id)
        if purity < low_purity_threshold:
            group["rows_lt_low_purity"] += 1
            group["low_purity_leaf_ids"].add(leaf_id)
        if key == "label":
            if dominant_label == group_value:
                group["dominant_leaf_ids"].add(leaf_id)
                if purity >= purity_threshold:
                    group["dominant_high_purity_leaf_ids"].add(leaf_id)
            if row["is_dominant_label"]:
                group["dominant_rows"] += 1
            if purity >= purity_threshold and dominant_label != group_value:
                group["minority_rows_in_pure_leaf"] += 1
        if key == "primary_scene":
            group["label_set"].add(str(row["label"]))
            group["dominant_label_set"].add(dominant_label)

    result = {}
    for group_value, group in stats.items():
        rows = int(group["rows"])
        result[group_value] = {
            key: group_value,
            "rows": rows,
            "leaf_count": len(group["leaf_ids"]),
            "weighted_avg_leaf_purity": safe_div(group["leaf_purity_sum"], rows),
            "rows_ge_purity": int(group["rows_ge_purity"]),
            "rows_ge_purity_ratio": safe_div(group["rows_ge_purity"], rows),
            "rows_ge_high_purity": int(group["rows_ge_high_purity"]),
            "rows_ge_high_purity_ratio": safe_div(group["rows_ge_high_purity"], rows),
            "rows_lt_low_purity": int(group["rows_lt_low_purity"]),
            "rows_lt_low_purity_ratio": safe_div(group["rows_lt_low_purity"], rows),
            "high_enough_leaf_count": len(group["high_enough_leaf_ids"]),
            "high_purity_leaf_count": len(group["high_purity_leaf_ids"]),
            "low_purity_leaf_count": len(group["low_purity_leaf_ids"]),
            "review_candidate_count": int(review_counts.get(group_value, 0)),
        }
        if key == "label":
            result[group_value].update({
                "dominant_leaf_count": len(group["dominant_leaf_ids"]),
                "dominant_high_purity_leaf_count": len(group["dominant_high_purity_leaf_ids"]),
                "dominant_row_ratio": safe_div(group["dominant_rows"], rows),
                "minority_rows_in_pure_leaf": int(group["minority_rows_in_pure_leaf"]),
                "minority_rows_in_pure_leaf_ratio": safe_div(group["minority_rows_in_pure_leaf"], rows),
            })
        if key == "primary_scene":
            result[group_value].update({
                "label_count": len(group["label_set"]),
                "dominant_label_count": len(group["dominant_label_set"]),
            })
    return result


def new_group_stats() -> Dict[str, Any]:
    return {
        "rows": 0,
        "leaf_purity_sum": 0.0,
        "rows_ge_purity": 0,
        "rows_ge_high_purity": 0,
        "rows_lt_low_purity": 0,
        "dominant_rows": 0,
        "minority_rows_in_pure_leaf": 0,
        "leaf_ids": set(),
        "high_enough_leaf_ids": set(),
        "high_purity_leaf_ids": set(),
        "low_purity_leaf_ids": set(),
        "dominant_leaf_ids": set(),
        "dominant_high_purity_leaf_ids": set(),
        "label_set": set(),
        "dominant_label_set": set(),
    }


def build_overview_rows(old_metrics: Dict[str, Any], new_metrics: Dict[str, Any], old_name: str,
                        new_name: str) -> List[Dict[str, Any]]:
    specs = [
        ("num_rows", "Rows", "higher"),
        ("num_leaves", "Leaves", "neutral"),
        ("macro_avg_purity", "Macro Avg Purity", "higher"),
        ("weighted_avg_purity", "Weighted Avg Purity", "higher"),
        ("rows_ge_purity", "Rows >= Purity Threshold", "higher"),
        ("rows_ge_purity_ratio", "Rows >= Purity Threshold Ratio", "higher"),
        ("rows_ge_high_purity", "Rows >= High Purity Threshold", "higher"),
        ("rows_ge_high_purity_ratio", "Rows >= High Purity Threshold Ratio", "higher"),
        ("rows_lt_low_purity", "Rows < Low Purity Threshold", "lower"),
        ("rows_lt_low_purity_ratio", "Rows < Low Purity Threshold Ratio", "lower"),
        ("impure_leaf_count", "Impure Leaf Count", "lower"),
        ("impure_row_count", "Impure Row Count", "lower"),
        ("impure_row_ratio", "Impure Row Ratio", "lower"),
        ("review_first_leaf_count", "Review-First Leaf Count", "lower"),
        ("review_first_row_count", "Review-First Row Count", "lower"),
        ("review_first_row_ratio", "Review-First Row Ratio", "lower"),
        ("suggested_total", "Suggested Total", "neutral"),
        ("keep_ratio", "Keep Ratio", "neutral"),
        ("review_candidate_total", "Review Candidate Total", "lower"),
    ]
    rows = []
    for metric_key, metric_name, better in specs:
        old_value = old_metrics[metric_key]
        new_value = new_metrics[metric_key]
        rows.append({
            "metric": metric_key,
            "metric_name": metric_name,
            "better_direction": better,
            f"{old_name}_value": old_value,
            f"{new_name}_value": new_value,
            "delta": new_value - old_value,
            "winner": pick_winner(old_value, new_value, better, old_name, new_name),
        })
    for candidate_type in sorted(set(old_metrics["review_candidate_counts"]) | set(new_metrics["review_candidate_counts"])):
        old_value = int(old_metrics["review_candidate_counts"].get(candidate_type, 0))
        new_value = int(new_metrics["review_candidate_counts"].get(candidate_type, 0))
        rows.append({
            "metric": f"review_candidate_{candidate_type}",
            "metric_name": f"Review Candidate: {candidate_type}",
            "better_direction": "lower",
            f"{old_name}_value": old_value,
            f"{new_name}_value": new_value,
            "delta": new_value - old_value,
            "winner": pick_winner(old_value, new_value, "lower", old_name, new_name),
        })
    return rows


def build_group_comparison_rows(old_metrics: Dict[str, Dict[str, Any]], new_metrics: Dict[str, Dict[str, Any]], key: str,
                                old_name: str, new_name: str) -> List[Dict[str, Any]]:
    metric_keys = [
        "rows",
        "leaf_count",
        "weighted_avg_leaf_purity",
        "rows_ge_purity",
        "rows_ge_purity_ratio",
        "rows_ge_high_purity",
        "rows_ge_high_purity_ratio",
        "rows_lt_low_purity",
        "rows_lt_low_purity_ratio",
        "high_enough_leaf_count",
        "high_purity_leaf_count",
        "low_purity_leaf_count",
        "review_candidate_count",
    ]
    optional_metric_keys = [
        "dominant_leaf_count",
        "dominant_high_purity_leaf_count",
        "dominant_row_ratio",
        "minority_rows_in_pure_leaf",
        "minority_rows_in_pure_leaf_ratio",
        "label_count",
        "dominant_label_count",
    ]
    included_optional_metric_keys = [
        metric_key for metric_key in optional_metric_keys
        if any(metric_key in row for row in old_metrics.values()) or any(metric_key in row for row in new_metrics.values())
    ]
    keys = sorted(set(old_metrics) | set(new_metrics))
    rows = []
    for group_value in keys:
        old_row = old_metrics.get(group_value, {})
        new_row = new_metrics.get(group_value, {})
        row = {key: group_value}
        for metric_key in metric_keys:
            old_value = old_row.get(metric_key, 0)
            new_value = new_row.get(metric_key, 0)
            row[f"old_{metric_key}"] = old_value
            row[f"new_{metric_key}"] = new_value
            row[f"delta_{metric_key}"] = new_value - old_value
        for metric_key in included_optional_metric_keys:
            old_value = old_row.get(metric_key, 0)
            new_value = new_row.get(metric_key, 0)
            row[f"old_{metric_key}"] = old_value
            row[f"new_{metric_key}"] = new_value
            row[f"delta_{metric_key}"] = new_value - old_value
        row["purity_delta_rank_key"] = row["delta_weighted_avg_leaf_purity"]
        row["max_rows_for_ranking"] = max(row["old_rows"], row["new_rows"])
        row["old_name"] = old_name
        row["new_name"] = new_name
        rows.append(row)
    rows.sort(key=lambda item: (-item["max_rows_for_ranking"], str(item[key])))
    return rows


def build_summary(
        old_metrics: Dict[str, Any],
        new_metrics: Dict[str, Any],
        overview_rows: Sequence[Dict[str, Any]],
        label_rows: Sequence[Dict[str, Any]],
        primary_scene_rows: Sequence[Dict[str, Any]],
        args: argparse.Namespace,
        purity_threshold: float) -> Dict[str, Any]:
    return {
        "old_name": args.old_name,
        "new_name": args.new_name,
        "old_report_dir": old_metrics["report_dir"],
        "new_report_dir": new_metrics["report_dir"],
        "purity_threshold": purity_threshold,
        "high_purity_threshold": args.high_purity_threshold,
        "low_purity_threshold": args.low_purity_threshold,
        "overview": list(overview_rows),
        "high_level_takeaways": build_takeaways(old_metrics, new_metrics, args.old_name, args.new_name),
        "top_improved_labels": top_group_rows(label_rows, "label", args.topn, args.min_group_rows, reverse=True),
        "top_worsened_labels": top_group_rows(label_rows, "label", args.topn, args.min_group_rows, reverse=False),
        "top_improved_primary_scenes": top_group_rows(
            primary_scene_rows, "primary_scene", args.topn, args.min_group_rows, reverse=True),
        "top_worsened_primary_scenes": top_group_rows(
            primary_scene_rows, "primary_scene", args.topn, args.min_group_rows, reverse=False),
        "outputs": [
            "compare_overview.csv",
            "compare_label_metrics.csv",
            "compare_primary_scene_metrics.csv",
            "compare_summary.json",
            "compare_summary.md",
        ],
    }


def build_takeaways(old_metrics: Dict[str, Any], new_metrics: Dict[str, Any], old_name: str,
                    new_name: str) -> List[str]:
    lines = []
    purity_delta = new_metrics["weighted_avg_purity"] - old_metrics["weighted_avg_purity"]
    if purity_delta > 0:
        lines.append(
            f"{new_name} 的 weighted_avg_purity 更高，提升 {purity_delta:.4f}。")
    elif purity_delta < 0:
        lines.append(
            f"{new_name} 的 weighted_avg_purity 更低，下降 {abs(purity_delta):.4f}。")
    low_delta = new_metrics["rows_lt_low_purity_ratio"] - old_metrics["rows_lt_low_purity_ratio"]
    if low_delta < 0:
        lines.append(
            f"{new_name} 的低纯度样本占比更低，减少 {abs(low_delta):.4f}。")
    elif low_delta > 0:
        lines.append(
            f"{new_name} 的低纯度样本占比更高，增加 {low_delta:.4f}。")
    review_delta = new_metrics["review_candidate_total"] - old_metrics["review_candidate_total"]
    if review_delta < 0:
        lines.append(f"{new_name} 的 review candidate 更少，减少 {abs(review_delta)} 个。")
    elif review_delta > 0:
        lines.append(f"{new_name} 的 review candidate 更多，增加 {review_delta} 个。")
    if not lines:
        lines.append(f"{old_name} 和 {new_name} 在核心离线指标上基本持平。")
    return lines


def top_group_rows(rows: Sequence[Dict[str, Any]], key: str, topn: int, min_group_rows: int,
                   reverse: bool) -> List[Dict[str, Any]]:
    filtered = [row for row in rows if int(row["max_rows_for_ranking"]) >= min_group_rows]
    filtered.sort(key=lambda item: (item["purity_delta_rank_key"], -item["max_rows_for_ranking"], str(item[key])),
                  reverse=reverse)
    output = []
    for row in filtered[:topn]:
        output.append({
            key: row[key],
            "old_rows": row["old_rows"],
            "new_rows": row["new_rows"],
            "delta_rows": row["delta_rows"],
            "old_weighted_avg_leaf_purity": row["old_weighted_avg_leaf_purity"],
            "new_weighted_avg_leaf_purity": row["new_weighted_avg_leaf_purity"],
            "delta_weighted_avg_leaf_purity": row["delta_weighted_avg_leaf_purity"],
            "old_rows_lt_low_purity_ratio": row["old_rows_lt_low_purity_ratio"],
            "new_rows_lt_low_purity_ratio": row["new_rows_lt_low_purity_ratio"],
            "delta_rows_lt_low_purity_ratio": row["delta_rows_lt_low_purity_ratio"],
        })
    return output


def render_markdown(summary: Dict[str, Any], old_name: str, new_name: str) -> str:
    lines = [
        "# Adaptive Report Comparison",
        "",
        "## Overview",
        "",
        f"- Old: `{old_name}` -> `{summary['old_report_dir']}`",
        f"- New: `{new_name}` -> `{summary['new_report_dir']}`",
        f"- Purity threshold: {summary['purity_threshold']}",
        f"- High purity threshold: {summary['high_purity_threshold']}",
        f"- Low purity threshold: {summary['low_purity_threshold']}",
        "",
        "## Takeaways",
        "",
    ]
    for line in summary["high_level_takeaways"]:
        lines.append(f"- {line}")
    lines.extend(["", "## Overview Metrics", ""])
    lines.append(f"| metric | {old_name} | {new_name} | delta | winner |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for row in summary["overview"]:
        lines.append(
            f"| {row['metric_name']} | {format_value(row[f'{old_name}_value'])} | "
            f"{format_value(row[f'{new_name}_value'])} | {format_value(row['delta'])} | {row['winner']} |")
    lines.extend(["", "## Top Improved Labels", ""])
    lines.extend(render_top_group_table(summary["top_improved_labels"], "label"))
    lines.extend(["", "## Top Worsened Labels", ""])
    lines.extend(render_top_group_table(summary["top_worsened_labels"], "label"))
    lines.extend(["", "## Top Improved Primary Scenes", ""])
    lines.extend(render_top_group_table(summary["top_improved_primary_scenes"], "primary_scene"))
    lines.extend(["", "## Top Worsened Primary Scenes", ""])
    lines.extend(render_top_group_table(summary["top_worsened_primary_scenes"], "primary_scene"))
    lines.append("")
    return "\n".join(lines)


def render_top_group_table(rows: Sequence[Dict[str, Any]], key: str) -> List[str]:
    if not rows:
        return ["- No groups matched the ranking filter."]
    lines = [
        f"| {key} | old_rows | new_rows | delta_rows | old_purity | new_purity | delta_purity | old_low_ratio | new_low_ratio | delta_low_ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row[key]}` | {row['old_rows']} | {row['new_rows']} | {row['delta_rows']} | "
            f"{row['old_weighted_avg_leaf_purity']:.4f} | {row['new_weighted_avg_leaf_purity']:.4f} | "
            f"{row['delta_weighted_avg_leaf_purity']:.4f} | {row['old_rows_lt_low_purity_ratio']:.4f} | "
            f"{row['new_rows_lt_low_purity_ratio']:.4f} | {row['delta_rows_lt_low_purity_ratio']:.4f} |")
    return lines


def pick_winner(old_value: float, new_value: float, better_direction: str, old_name: str, new_name: str) -> str:
    if better_direction == "higher":
        if new_value > old_value:
            return new_name
        if new_value < old_value:
            return old_name
        return "tie"
    if better_direction == "lower":
        if new_value < old_value:
            return new_name
        if new_value > old_value:
            return old_name
        return "tie"
    return "tie"


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def to_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def to_optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


if __name__ == "__main__":
    main()
