#!/usr/bin/env python3
"""Build an ms-swift custom dataset_info.json from adaptive split manifest."""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert split_manifest.csv from split_raw_by_adaptive_leaf.py to ms-swift dataset_info.json.")
    parser.add_argument("--split-manifest", required=True, help="split_manifest.csv from split_raw_by_adaptive_leaf.py.")
    parser.add_argument("--output", required=True, help="Output ms-swift custom dataset_info JSON path.")
    parser.add_argument(
        "--dataset-name-prefix",
        default="vla_adaptive",
        help="Prefix for generated dataset_name values. Default: vla_adaptive.")
    parser.add_argument(
        "--dataset-args-output",
        default=None,
        help="Optional text file with recommended --dataset entries, one per line.")
    parser.add_argument(
        "--plan-output",
        default=None,
        help="Optional JSON with rich per-leaf metadata and recommended dataset args.")
    parser.add_argument(
        "--config-fragment-output",
        default=None,
        help="Optional YAML config fragment with custom_dataset_info and dataset entries.")
    parser.add_argument(
        "--enable-channel-loss-in-config",
        action="store_true",
        help="Add enable_channel_loss: true to the generated YAML fragment.")
    parser.add_argument(
        "--path-mode",
        choices=["absolute", "relative"],
        default="absolute",
        help="Write dataset_path as absolute paths or paths relative to the dataset_info JSON. Default: absolute.")
    parser.add_argument(
        "--target-column",
        default="suggested_target_count",
        help="Column used for recommended downsample count. Default: suggested_target_count.")
    parser.add_argument(
        "--purity-ratio-threshold",
        type=float,
        default=None,
        help="If avg_leaf_purity is greater than this value, override target with count * --purity-sample-ratio.")
    parser.add_argument(
        "--purity-sample-ratio",
        type=float,
        default=None,
        help="Sampling ratio for rows whose avg_leaf_purity is greater than --purity-ratio-threshold.")
    parser.add_argument(
        "--non-purity-target-mode",
        choices=["suggested_target_count", "full"],
        default="suggested_target_count",
        help="Target policy for rows not matching the purity-ratio rule. Default: suggested_target_count.")
    parser.add_argument(
        "--min-purity",
        type=float,
        default=None,
        help="Only include groups with avg_leaf_purity >= this value.")
    parser.add_argument(
        "--max-purity",
        type=float,
        default=None,
        help="Only include groups with avg_leaf_purity <= this value.")
    parser.add_argument(
        "--actions",
        default=None,
        help="Comma-separated suggested_action values to include, e.g. balanced_keep,downsample_frequent.")
    parser.add_argument(
        "--downsample-high-purity",
        action="store_true",
        help="Write dataset args as dataset_name#target for included rows whose target is smaller than count.")
    parser.add_argument(
        "--min-target-count",
        type=int,
        default=1,
        help="Drop rows whose target count is smaller than this value. Default: 1.")
    parser.add_argument(
        "--no-columns",
        action="store_true",
        help="Do not emit columns={'messages': 'messages'} in dataset_info entries.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    manifest_path = Path(args.split_manifest)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    action_filter = parse_csv_set(args.actions)
    rows = list(read_manifest(manifest_path))
    selected_rows = [
        row for row in rows
        if row_matches(row, args.target_column, args.min_purity, args.max_purity, action_filter, args.min_target_count,
                       args.purity_ratio_threshold, args.purity_sample_ratio, args.non_purity_target_mode)
    ]
    dataset_entries, plan_rows = build_dataset_entries(
        selected_rows,
        manifest_path.parent,
        output_path,
        args.dataset_name_prefix,
        args.target_column,
        args.path_mode,
        args.purity_ratio_threshold,
        args.purity_sample_ratio,
        args.non_purity_target_mode,
        emit_columns=not args.no_columns,
        downsample_high_purity=args.downsample_high_purity,
    )
    output_path.write_text(json.dumps(dataset_entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dataset_args_output = Path(args.dataset_args_output) if args.dataset_args_output else output_path.with_suffix(
        output_path.suffix + ".dataset_args.txt")
    dataset_args_output.parent.mkdir(parents=True, exist_ok=True)
    dataset_args_output.write_text("\n".join(row["recommended_dataset_arg"] for row in plan_rows) + "\n",
                                   encoding="utf-8")

    plan_output = Path(args.plan_output) if args.plan_output else output_path.with_suffix(output_path.suffix +
                                                                                          ".plan.json")
    plan_output.parent.mkdir(parents=True, exist_ok=True)
    plan = {
        "source_manifest": str(manifest_path),
        "dataset_info": str(output_path),
        "dataset_args": str(dataset_args_output),
        "total_manifest_rows": len(rows),
        "selected_rows": len(plan_rows),
        "raw_total_count": sum(int(row["raw_count"]) for row in plan_rows),
        "recommended_total_count": sum(int(row["recommended_sample_count"]) for row in plan_rows),
        "rows": plan_rows,
    }
    plan_output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_fragment_output = Path(args.config_fragment_output) if args.config_fragment_output else output_path.with_suffix(
        output_path.suffix + ".dataset_config.yaml")
    config_fragment_output.parent.mkdir(parents=True, exist_ok=True)
    config_fragment_output.write_text(
        render_config_fragment(output_path, plan_rows, args.enable_channel_loss_in_config), encoding="utf-8")
    print(f"wrote {len(dataset_entries)} ms-swift dataset entries -> {output_path}")
    print(f"recommended dataset args -> {dataset_args_output}")
    print(f"sampling plan -> {plan_output}")
    print(f"dataset config fragment -> {config_fragment_output}")


def read_manifest(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "group_key",
            "file",
            "count",
            "suggested_target_count",
            "leaf_ids",
            "dominant_label",
            "dominant_lateral",
            "dominant_longitudinal",
            "primary_scene_top",
            "secondary_scene_top",
            "avg_leaf_purity",
            "suggested_action_top",
            "stop_reason_top",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns in {path}: {sorted(missing)}")
        yield from reader


def validate_args(args: argparse.Namespace) -> None:
    if (args.purity_ratio_threshold is None) != (args.purity_sample_ratio is None):
        raise ValueError("--purity-ratio-threshold and --purity-sample-ratio must be set together")
    if args.purity_sample_ratio is not None and not 0 <= args.purity_sample_ratio <= 1:
        raise ValueError("--purity-sample-ratio must be in [0, 1]")


def build_dataset_entries(
        rows: Sequence[Dict[str, str]],
        manifest_dir: Path,
        output_path: Path,
        dataset_name_prefix: str,
        target_column: str,
        path_mode: str,
        purity_ratio_threshold: Optional[float],
        purity_sample_ratio: Optional[float],
        non_purity_target_mode: str,
        *,
        emit_columns: bool,
        downsample_high_purity: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    entries = []
    plan_rows = []
    used_names = set()
    for row in rows:
        dataset_name = unique_name(dataset_name_prefix, row, used_names)
        dataset_path = format_dataset_path(resolve_manifest_file(row["file"], manifest_dir), output_path, path_mode)
        count = to_int(row["count"])
        target, target_source = target_count(row, target_column, purity_ratio_threshold, purity_sample_ratio,
                                             non_purity_target_mode)
        recommended_arg = dataset_name
        if (downsample_high_purity or target_source != "full") and target < count:
            recommended_arg = f"{dataset_name}#{target}"
        help_text = build_help(row, target, target_source)
        tags = build_tags(row)
        entry: Dict[str, Any] = {
            "dataset_path": dataset_path,
            "dataset_name": dataset_name,
            "split": ["train"],
            "tags": tags,
            "help": help_text,
        }
        if emit_columns:
            entry["columns"] = {"messages": "messages"}
        entries.append(entry)
        plan_rows.append({
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "recommended_dataset_arg": recommended_arg,
            "recommended_sample_count": target if "#" in recommended_arg else count,
            "raw_count": count,
            "target_count": target,
            "target_source": target_source,
            "sample_ratio": safe_div(target, count),
            "leaf_ids": row["leaf_ids"],
            "group_key": row["group_key"],
            "dominant_label": row["dominant_label"],
            "dominant_lateral": row["dominant_lateral"],
            "dominant_longitudinal": row["dominant_longitudinal"],
            "primary_scene_top": row["primary_scene_top"],
            "secondary_scene_top": row["secondary_scene_top"],
            "avg_leaf_purity": to_float(row["avg_leaf_purity"]),
            "suggested_action": row["suggested_action_top"],
            "stop_reason": row["stop_reason_top"],
        })
    return entries, plan_rows


def row_matches(
        row: Dict[str, str],
        target_column: str,
        min_purity: Optional[float],
        max_purity: Optional[float],
        action_filter: Optional[set],
        min_target_count: int,
        purity_ratio_threshold: Optional[float],
        purity_sample_ratio: Optional[float],
        non_purity_target_mode: str) -> bool:
    purity = to_float(row["avg_leaf_purity"])
    if min_purity is not None and purity < min_purity:
        return False
    if max_purity is not None and purity > max_purity:
        return False
    if action_filter is not None and row["suggested_action_top"] not in action_filter:
        return False
    target, _ = target_count(row, target_column, purity_ratio_threshold, purity_sample_ratio, non_purity_target_mode)
    return target >= min_target_count


def target_count(
        row: Dict[str, str],
        target_column: str,
        purity_ratio_threshold: Optional[float],
        purity_sample_ratio: Optional[float],
        non_purity_target_mode: str) -> Tuple[int, str]:
    count = to_int(row["count"])
    purity = to_float(row["avg_leaf_purity"])
    if purity_ratio_threshold is not None and purity > purity_ratio_threshold:
        target = int(round(count * float(purity_sample_ratio)))
        return min(count, max(0, target)), "purity_ratio"
    if non_purity_target_mode == "full":
        return count, "full"
    return min(count, max(0, to_int(row[target_column]))), target_column


def build_help(row: Dict[str, str], target: int, target_source: str) -> str:
    parts = [
        "adaptive leaf split",
        f"group_key={row['group_key']}",
        f"leaf_ids={row['leaf_ids']}",
        f"count={row['count']}",
        f"suggested_target_count={target}",
        f"target_source={target_source}",
        f"avg_leaf_purity={format_float(to_float(row['avg_leaf_purity']))}",
        f"dominant_label={row['dominant_label']}",
        f"primary_scene_top={row['primary_scene_top']}",
        f"secondary_scene_top={row['secondary_scene_top']}",
        f"suggested_action={row['suggested_action_top']}",
        f"stop_reason={row['stop_reason_top']}",
    ]
    return "; ".join(parts)


def build_tags(row: Dict[str, str]) -> List[str]:
    purity = to_float(row["avg_leaf_purity"])
    purity_bucket = "purity_high" if purity >= 0.9 else "purity_mid" if purity >= 0.7 else "purity_low"
    return [
        "vla",
        "adaptive_leaf",
        purity_bucket,
        f"action:{sanitize_tag(row['suggested_action_top'])}",
        f"dominant:{sanitize_tag(row['dominant_label'])}",
        f"primary:{sanitize_tag(row['primary_scene_top'])}",
    ]


def render_config_fragment(dataset_info_path: Path, plan_rows: Sequence[Dict[str, Any]],
                           enable_channel_loss: bool) -> str:
    lines = [
        "# Paste this block into a swift --config YAML file.",
        "custom_dataset_info:",
        f"  - {yaml_quote(str(dataset_info_path))}",
        "dataset:",
    ]
    for row in plan_rows:
        lines.append(f"  - {yaml_quote(str(row['recommended_dataset_arg']))}")
    if enable_channel_loss:
        lines.append("enable_channel_loss: true")
    return "\n".join(lines) + "\n"


def unique_name(prefix: str, row: Dict[str, str], used_names: set) -> str:
    leaf_ids = [part for part in row["leaf_ids"].split() if part]
    purity = purity_token(to_float(row["avg_leaf_purity"]))
    dominant = sanitize_name(row["dominant_label"], max_len=64)
    if len(leaf_ids) == 1:
        base = f"{prefix}_leaf_{int(leaf_ids[0]):06d}_{purity}_{dominant}"
    else:
        base = f"{prefix}_{sanitize_name(row['group_key'], max_len=80)}_{purity}_{dominant}"
    name = sanitize_name(base, max_len=120)
    if name not in used_names:
        used_names.add(name)
        return name
    suffix = 1
    while f"{name}_{suffix}" in used_names:
        suffix += 1
    unique = f"{name}_{suffix}"
    used_names.add(unique)
    return unique


def format_dataset_path(path: Path, output_path: Path, path_mode: str) -> str:
    path = path.expanduser()
    if path_mode == "absolute":
        return str(path.resolve())
    return os.path.relpath(path.resolve(), output_path.parent.resolve())


def resolve_manifest_file(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return manifest_dir / path


def parse_csv_set(value: Optional[str]) -> Optional[set]:
    if value in (None, ""):
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def sanitize_name(value: str, max_len: int = 120) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return (value or "dataset")[:max_len]


def purity_token(value: float) -> str:
    value = max(0.0, min(1.0, value))
    return f"p{int(round(value * 1000)):04d}"


def sanitize_tag(value: str, max_len: int = 80) -> str:
    value = re.sub(r"[\s,;]+", "_", value)
    return (value or "UNKNOWN")[:max_len]


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


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
    return f"{value:.6g}"


if __name__ == "__main__":
    main()
