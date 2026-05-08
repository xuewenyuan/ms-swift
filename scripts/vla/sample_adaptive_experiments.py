#!/usr/bin/env python3
"""Build adaptive-leaf sampling experiment datasets for VLA training."""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from analyze_cluster_mining import UNKNOWN, safe_div, split_label, write_csv
from dump_messages_features import parse_vla_output, split_input_output


STRATEGY_A = "A_high_purity_dedup"
STRATEGY_B = "B_remove_roundabout_dirty"
STRATEGY_C = "C_balanced_decision"
STRATEGY_D = "D_aggressive_purity_label_sampling"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample adaptive leaf JSONL files into A/B/C/D training datasets.")
    parser.add_argument(
        "--split-manifest",
        required=True,
        help="split_manifest.csv from split_raw_by_adaptive_leaf.py.")
    parser.add_argument("--output-dir", required=True, help="Output directory for experiment datasets.")
    parser.add_argument(
        "--strategies",
        default="A,B,C",
        help="Comma-separated strategies to build: A,B,C,D. Default: A,B,C.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    parser.add_argument("--num-shards", type=int, default=8, help="Number of output shards per strategy. Default: 8.")
    parser.add_argument("--output-prefix", default="sampled", help="Output shard filename prefix. Default: sampled.")
    parser.add_argument(
        "--upsample-mode",
        choices=["repeat", "with_replacement"],
        default="repeat",
        help="How to sample when target > available. Default: repeat.")
    parser.add_argument(
        "--shuffle-output",
        action="store_true",
        help="Shuffle selected lines before writing each group. Useful when num-shards is 1.")

    parser.add_argument("--purity-high-threshold", type=float, default=0.9, help="High purity threshold. Default: 0.9.")
    parser.add_argument("--purity-mid-threshold", type=float, default=0.7, help="Mid purity threshold. Default: 0.7.")
    parser.add_argument(
        "--high-purity-ratio",
        type=float,
        default=0.3,
        help="A/C default keep ratio for purity >= high threshold. Default: 0.3.")
    parser.add_argument(
        "--mid-purity-ratio",
        type=float,
        default=0.5,
        help="A/C default keep ratio for mid <= purity < high. Default: 0.5.")
    parser.add_argument(
        "--low-purity-ratio",
        type=float,
        default=1.0,
        help="C keep ratio for non-dirty low-purity leaves. Set 1.2 for light upsampling. Default: 1.0.")

    parser.add_argument(
        "--roundabout-regex",
        default="环岛|绕岛|进入环岛|驶出环岛|roundabout",
        help="Regex matched against user prompts to detect roundabout samples.")
    parser.add_argument(
        "--roundabout-ratio-threshold",
        type=float,
        default=0.5,
        help="Roundabout ratio threshold for dirty leaf removal. Default: 0.5.")
    parser.add_argument(
        "--roundabout-low-purity-threshold",
        type=float,
        default=0.5,
        help="Leaf is considered low-purity enough for roundabout removal below this threshold. Default: 0.5.")
    parser.add_argument(
        "--dirty-secondary-scene-regex",
        default="环岛|roundabout",
        help="Regex matched against secondary_scene_top for review_first dirty removal.")
    parser.add_argument(
        "--write-holdout",
        action="store_true",
        help="Write removed dirty roundabout leaves to holdout JSONL shards for B/C/D.")

    parser.add_argument(
        "--high-label-count",
        type=int,
        default=100000,
        help="Decision labels with count >= this are high-frequency. Default: 100000.")
    parser.add_argument(
        "--low-label-count",
        type=int,
        default=10000,
        help="Decision labels with count <= this are low-frequency. Default: 10000.")
    parser.add_argument(
        "--high-purity-high-frequency-ratio",
        type=float,
        default=0.3,
        help="C ratio for purity>=0.9 and high-frequency labels. Default: 0.3.")
    parser.add_argument(
        "--mid-purity-mid-frequency-ratio",
        type=float,
        default=0.7,
        help="C ratio for 0.7<=purity<0.9 and mid-frequency labels. Default: 0.7.")
    parser.add_argument(
        "--low-label-upsample-ratio",
        type=float,
        default=2.0,
        help="C upsample ratio for low-frequency labels when not dirty. Default: 2.0.")
    parser.add_argument(
        "--max-low-label-upsample-ratio",
        type=float,
        default=3.0,
        help="C cap for low-frequency upsampling ratio. Default: 3.0.")
    parser.add_argument(
        "--d-high-purity-threshold",
        type=float,
        default=0.7,
        help="D treats leaves with purity >= this as high-purity. Default: 0.7.")
    parser.add_argument(
        "--d-mid-purity-low",
        type=float,
        default=0.4,
        help="D treats leaves with this <= purity < high threshold as mid-purity. Default: 0.4.")
    parser.add_argument(
        "--d-dominant-label-ratio",
        type=float,
        default=0.1,
        help="D keep ratio for the dominant label in high-purity leaves. Default: 0.1.")
    parser.add_argument(
        "--d-mid-label-target-share",
        type=float,
        default=0.1,
        help="D target share for labels whose ratio in a mid-purity leaf is at least this value. Default: 0.1.")
    parser.add_argument(
        "--d-min-label-target-count",
        type=int,
        default=100,
        help="D up-samples labels below this count in mid-purity leaves. Default: 100.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    rng = random.Random(args.seed)
    manifest_path = Path(args.split_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    roundabout_re = re.compile(args.roundabout_regex, re.IGNORECASE)
    leaf_stats = [scan_leaf_row(row, roundabout_re) for row in rows]
    label_counts = build_label_counts(leaf_stats)
    strategies = parse_strategies(args.strategies)

    for strategy in strategies:
        strategy_dir = output_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        if strategy == STRATEGY_D:
            plan_rows, label_plan_rows, total_target = build_strategy_d_plans(leaf_stats, label_counts, args)
            write_csv(strategy_dir / "sampling_plan.csv", plan_rows)
            write_csv(strategy_dir / "label_sampling_plan.csv", label_plan_rows)
            write_strategy_d_dataset(strategy, strategy_dir, label_plan_rows, args, rng)
            summary = build_summary(strategy, plan_rows, total_target, label_counts, args, label_plan_rows)
        else:
            plan_rows, total_target = build_strategy_plan(strategy, leaf_stats, label_counts, args)
            write_csv(strategy_dir / "sampling_plan.csv", plan_rows)
            write_strategy_dataset(strategy, strategy_dir, plan_rows, args, rng)
            summary = build_summary(strategy, plan_rows, total_target, label_counts, args)
        (strategy_dir / "sampling_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{strategy}: target={total_target}, groups={len(plan_rows)} -> {strategy_dir}")


def validate_args(args: argparse.Namespace) -> None:
    for name in [
            "high_purity_ratio",
            "mid_purity_ratio",
            "low_purity_ratio",
            "roundabout_ratio_threshold",
            "high_purity_high_frequency_ratio",
            "mid_purity_mid_frequency_ratio",
            "low_label_upsample_ratio",
            "max_low_label_upsample_ratio",
            "d_high_purity_threshold",
            "d_mid_purity_low",
            "d_dominant_label_ratio",
            "d_mid_label_target_share",
    ]:
        value = float(getattr(args, name))
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.d_mid_purity_low > args.d_high_purity_threshold:
        raise ValueError("--d-mid-purity-low should be <= --d-high-purity-threshold")
    if args.d_mid_label_target_share <= 0:
        raise ValueError("--d-mid-label-target-share must be positive")
    if args.d_min_label_target_count < 0:
        raise ValueError("--d-min-label-target-count must be non-negative")
    if args.low_label_count > args.high_label_count:
        raise ValueError("--low-label-count should be <= --high-label-count")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")


def parse_strategies(value: str) -> List[str]:
    mapping = {"A": STRATEGY_A, "B": STRATEGY_B, "C": STRATEGY_C, "D": STRATEGY_D}
    strategies = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in mapping:
            raise ValueError(f"unknown strategy {item}; choose from A,B,C,D")
        strategies.append(mapping[item])
    if not strategies:
        raise ValueError("no strategies selected")
    return strategies


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"file", "count", "dominant_label", "avg_leaf_purity", "suggested_action_top", "secondary_scene_top"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns in {path}: {sorted(missing)}")
        rows = []
        for row in reader:
            item = dict(row)
            file_path = Path(item["file"]).expanduser()
            if not file_path.is_absolute():
                file_path = path.parent / file_path
            item["file"] = str(file_path)
            rows.append(item)
        return rows


def scan_leaf_row(row: Dict[str, str], roundabout_re: re.Pattern) -> Dict[str, Any]:
    path = Path(row["file"]).expanduser()
    label_counts: Counter = Counter()
    roundabout_count = 0
    available = 0
    for line in iter_jsonl_lines(path):
        available += 1
        obj = json.loads(line)
        if is_roundabout_sample(obj, roundabout_re):
            roundabout_count += 1
        label = sample_decision_label(obj)
        if label:
            label_counts[label] += 1

    manifest_count = to_int(row.get("count"))
    if available != manifest_count:
        print(f"warning: manifest count {manifest_count} != scanned count {available} for {path}")
    if not label_counts:
        label_counts[str(row.get("dominant_label") or UNKNOWN)] = available
    dominant_label, dominant_label_count = label_counts.most_common(1)[0]
    purity = to_float(row.get("avg_leaf_purity"))
    return {
        **row,
        "file": str(path),
        "available": available,
        "scanned_dominant_label": dominant_label,
        "scanned_dominant_label_count": dominant_label_count,
        "scanned_dominant_label_ratio": safe_div(dominant_label_count, available),
        "label_counts": label_counts,
        "roundabout_count": roundabout_count,
        "roundabout_ratio": safe_div(roundabout_count, available),
        "purity": purity,
    }


def build_label_counts(leaf_stats: Sequence[Dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for row in leaf_stats:
        counts.update(row["label_counts"])
    return counts


def build_strategy_plan(
        strategy: str,
        leaf_stats: Sequence[Dict[str, Any]],
        label_counts: Counter,
        args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], int]:
    plan_rows = []
    total_target = 0
    for row in leaf_stats:
        ratio, action, reason = target_ratio(strategy, row, label_counts, args)
        available = int(row["available"])
        target = int(round(available * ratio))
        if ratio > 0 and available > 0:
            target = max(1, target)
        target = max(0, target)
        total_target += target if action != "holdout_dirty_roundabout" else 0
        label = row["scanned_dominant_label"]
        freq_bucket = label_frequency_bucket(label_counts[label], args)
        plan_rows.append({
            "strategy": strategy,
            "group_key": row.get("group_key", ""),
            "file": row["file"],
            "available": available,
            "target": target,
            "sample_ratio": safe_div(target, available),
            "action": action,
            "reason": reason,
            "purity": row["purity"],
            "roundabout_count": row["roundabout_count"],
            "roundabout_ratio": row["roundabout_ratio"],
            "dominant_label": label,
            "dominant_label_count_in_leaf": row["scanned_dominant_label_count"],
            "dominant_label_ratio_in_leaf": row["scanned_dominant_label_ratio"],
            "label_count_global": label_counts[label],
            "label_frequency_bucket": freq_bucket,
            "secondary_scene_top": row.get("secondary_scene_top", ""),
            "suggested_action_top": row.get("suggested_action_top", ""),
            "stop_reason_top": row.get("stop_reason_top", ""),
            "leaf_ids": row.get("leaf_ids", ""),
        })
    return plan_rows, total_target


def target_ratio(
        strategy: str,
        row: Dict[str, Any],
        label_counts: Counter,
        args: argparse.Namespace) -> Tuple[float, str, str]:
    purity = float(row["purity"])
    if strategy == STRATEGY_A:
        return high_purity_dedup_ratio(purity, args)

    dirty, dirty_reason = is_dirty_roundabout_leaf(row, args)
    if strategy == STRATEGY_B:
        if dirty:
            return 0.0, "holdout_dirty_roundabout", dirty_reason
        ratio, action, reason = high_purity_dedup_ratio(purity, args)
        return ratio, action, reason

    if strategy == STRATEGY_C:
        if dirty:
            return 0.0, "holdout_dirty_roundabout", dirty_reason
        label = row["scanned_dominant_label"]
        label_count = label_counts[label]
        freq_bucket = label_frequency_bucket(label_count, args)
        if freq_bucket == "low":
            ratio = min(args.low_label_upsample_ratio, args.max_low_label_upsample_ratio)
            return ratio, "upsample_low_frequency_label", f"label_count={label_count} <= low_label_count"
        if purity >= args.purity_high_threshold and freq_bucket == "high":
            return args.high_purity_high_frequency_ratio, "downsample_high_purity_high_frequency", (
                f"purity>={args.purity_high_threshold} and label_count={label_count}>=high_label_count")
        if args.purity_mid_threshold <= purity < args.purity_high_threshold and freq_bucket == "mid":
            return args.mid_purity_mid_frequency_ratio, "keep_mid_purity_mid_frequency", (
                f"{args.purity_mid_threshold}<=purity<{args.purity_high_threshold} and mid-frequency label")
        if purity < args.purity_mid_threshold:
            return args.low_purity_ratio, "keep_or_light_upsample_low_purity_non_roundabout", (
                f"purity<{args.purity_mid_threshold}, non-dirty")
        return high_purity_dedup_ratio(purity, args)

    raise ValueError(f"unsupported strategy: {strategy}")


def build_strategy_d_plans(
        leaf_stats: Sequence[Dict[str, Any]],
        label_counts: Counter,
        args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    label_plan_rows = []
    leaf_plan_rows = []
    total_target = 0
    for row in leaf_stats:
        leaf_label_rows = build_strategy_d_leaf_label_rows(row, label_counts, args)
        label_plan_rows.extend(leaf_label_rows)
        leaf_target = sum(int(item["target"]) for item in leaf_label_rows if item["action"] != "holdout_dirty_roundabout")
        if leaf_label_rows and leaf_label_rows[0]["action"] == "holdout_dirty_roundabout":
            leaf_action = "holdout_dirty_roundabout"
            leaf_reason = leaf_label_rows[0]["reason"]
            total_target += 0
        else:
            leaf_action = strategy_d_leaf_action(float(row["purity"]), args)
            leaf_reason = strategy_d_leaf_reason(float(row["purity"]), args)
            total_target += leaf_target
        available = int(row["available"])
        dominant_label = row["scanned_dominant_label"]
        leaf_plan_rows.append({
            "strategy": STRATEGY_D,
            "group_key": row.get("group_key", ""),
            "file": row["file"],
            "available": available,
            "target": leaf_target,
            "sample_ratio": safe_div(leaf_target, available),
            "action": leaf_action,
            "reason": leaf_reason,
            "purity": row["purity"],
            "roundabout_count": row["roundabout_count"],
            "roundabout_ratio": row["roundabout_ratio"],
            "dominant_label": dominant_label,
            "dominant_label_count_in_leaf": row["scanned_dominant_label_count"],
            "dominant_label_ratio_in_leaf": row["scanned_dominant_label_ratio"],
            "label_count_global": label_counts[dominant_label],
            "label_frequency_bucket": label_frequency_bucket(label_counts[dominant_label], args),
            "secondary_scene_top": row.get("secondary_scene_top", ""),
            "suggested_action_top": row.get("suggested_action_top", ""),
            "stop_reason_top": row.get("stop_reason_top", ""),
            "leaf_ids": row.get("leaf_ids", ""),
            "num_label_groups": len([item for item in leaf_label_rows if item["action"] != "holdout_dirty_roundabout"]),
        })
    return leaf_plan_rows, label_plan_rows, total_target


def build_strategy_d_leaf_label_rows(
        row: Dict[str, Any],
        label_counts: Counter,
        args: argparse.Namespace) -> List[Dict[str, Any]]:
    dirty, dirty_reason = is_dirty_roundabout_leaf(row, args)
    available = int(row["available"])
    purity = float(row["purity"])
    dominant_label = str(row["scanned_dominant_label"])
    if dirty:
        return [
            strategy_d_label_row(
                row,
                label_counts,
                label="__ALL__",
                label_count=available,
                target=0,
                action="holdout_dirty_roundabout",
                reason=dirty_reason,
                args=args,
            )
        ]

    label_rows = []
    for label, label_count in sorted(row["label_counts"].items(), key=lambda item: (-item[1], str(item[0]))):
        target, action, reason = strategy_d_label_target(
            purity=purity,
            available=available,
            label=str(label),
            label_count=int(label_count),
            dominant_label=dominant_label,
            args=args,
        )
        label_rows.append(strategy_d_label_row(row, label_counts, str(label), int(label_count), target, action, reason,
                                              args))
    return label_rows


def strategy_d_label_target(
        *,
        purity: float,
        available: int,
        label: str,
        label_count: int,
        dominant_label: str,
        args: argparse.Namespace) -> Tuple[int, str, str]:
    label_ratio = safe_div(label_count, available)
    if purity >= args.d_high_purity_threshold:
        if label == dominant_label:
            target = proportional_target(label_count, args.d_dominant_label_ratio)
            return target, "downsample_high_purity_dominant_label", (
                f"purity>={args.d_high_purity_threshold}, dominant_label, "
                f"keep_ratio={args.d_dominant_label_ratio}")
        return label_count, "keep_high_purity_non_dominant_label", (
            f"purity>={args.d_high_purity_threshold}, non-dominant label")

    if args.d_mid_purity_low <= purity < args.d_high_purity_threshold:
        if label_count < args.d_min_label_target_count:
            return args.d_min_label_target_count, "upsample_mid_purity_small_label", (
                f"{args.d_mid_purity_low}<=purity<{args.d_high_purity_threshold}, "
                f"label_count={label_count}<min_label_target_count={args.d_min_label_target_count}")
        if label_ratio >= args.d_mid_label_target_share:
            keep_ratio = args.d_mid_label_target_share / label_ratio
            target = proportional_target(label_count, keep_ratio)
            return target, "downsample_mid_purity_frequent_label", (
                f"{args.d_mid_purity_low}<=purity<{args.d_high_purity_threshold}, "
                f"label_ratio={label_ratio:.6f}>={args.d_mid_label_target_share}, keep_ratio={keep_ratio:.6f}")
        return label_count, "keep_mid_purity_tail_label", (
            f"{args.d_mid_purity_low}<=purity<{args.d_high_purity_threshold}, "
            f"label_ratio={label_ratio:.6f}<{args.d_mid_label_target_share}")

    return label_count, "keep_low_purity_label", f"purity<{args.d_mid_purity_low}"


def strategy_d_label_row(
        row: Dict[str, Any],
        label_counts: Counter,
        label: str,
        label_count: int,
        target: int,
        action: str,
        reason: str,
        args: argparse.Namespace) -> Dict[str, Any]:
    available = int(row["available"])
    label_ratio = safe_div(label_count, available)
    dominant_label = str(row["scanned_dominant_label"])
    return {
        "strategy": STRATEGY_D,
        "group_key": row.get("group_key", ""),
        "file": row["file"],
        "label": label,
        "label_count_in_leaf": label_count,
        "label_ratio_in_leaf": label_ratio,
        "target": target,
        "sample_ratio": safe_div(target, label_count),
        "action": action,
        "reason": reason,
        "purity": row["purity"],
        "available": available,
        "is_dominant_label": label == dominant_label,
        "dominant_label": dominant_label,
        "dominant_label_count_in_leaf": row["scanned_dominant_label_count"],
        "dominant_label_ratio_in_leaf": row["scanned_dominant_label_ratio"],
        "label_count_global": label_counts[label] if label != "__ALL__" else "",
        "label_frequency_bucket": label_frequency_bucket(label_counts[label], args) if label != "__ALL__" else "",
        "roundabout_count": row["roundabout_count"],
        "roundabout_ratio": row["roundabout_ratio"],
        "secondary_scene_top": row.get("secondary_scene_top", ""),
        "suggested_action_top": row.get("suggested_action_top", ""),
        "stop_reason_top": row.get("stop_reason_top", ""),
        "leaf_ids": row.get("leaf_ids", ""),
    }


def strategy_d_leaf_action(purity: float, args: argparse.Namespace) -> str:
    if purity >= args.d_high_purity_threshold:
        return "label_balance_high_purity"
    if purity >= args.d_mid_purity_low:
        return "label_balance_mid_purity"
    return "keep_low_purity"


def strategy_d_leaf_reason(purity: float, args: argparse.Namespace) -> str:
    if purity >= args.d_high_purity_threshold:
        return f"purity>={args.d_high_purity_threshold}; downsample dominant label only"
    if purity >= args.d_mid_purity_low:
        return (
            f"{args.d_mid_purity_low}<=purity<{args.d_high_purity_threshold}; "
            "downsample labels with p>=target_share and upsample small labels")
    return f"purity<{args.d_mid_purity_low}; keep unchanged"


def proportional_target(count: int, ratio: float) -> int:
    if count <= 0 or ratio <= 0:
        return 0
    return max(1, int(round(count * ratio)))


def high_purity_dedup_ratio(purity: float, args: argparse.Namespace) -> Tuple[float, str, str]:
    if purity >= args.purity_high_threshold:
        return args.high_purity_ratio, "downsample_high_purity", f"purity>={args.purity_high_threshold}"
    if purity >= args.purity_mid_threshold:
        return args.mid_purity_ratio, "downsample_mid_purity", (
            f"{args.purity_mid_threshold}<=purity<{args.purity_high_threshold}")
    return 1.0, "keep_low_purity", f"purity<{args.purity_mid_threshold}"


def is_dirty_roundabout_leaf(row: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, str]:
    purity = float(row["purity"])
    roundabout_ratio = float(row["roundabout_ratio"])
    secondary_scene = str(row.get("secondary_scene_top") or "")
    suggested_action = str(row.get("suggested_action_top") or "")
    scene_match = bool(re.search(args.dirty_secondary_scene_regex, secondary_scene, re.IGNORECASE))
    if roundabout_ratio >= args.roundabout_ratio_threshold and purity <= args.roundabout_low_purity_threshold:
        return True, (
            f"roundabout_ratio={roundabout_ratio:.4f}>={args.roundabout_ratio_threshold} "
            f"and purity={purity:.4f}<={args.roundabout_low_purity_threshold}")
    if scene_match and suggested_action == "review_first":
        return True, "secondary_scene_top matches dirty regex and suggested_action_top=review_first"
    return False, ""


def label_frequency_bucket(label_count: int, args: argparse.Namespace) -> str:
    if label_count >= args.high_label_count:
        return "high"
    if label_count <= args.low_label_count:
        return "low"
    return "mid"


def write_strategy_dataset(
        strategy: str,
        strategy_dir: Path,
        plan_rows: Sequence[Dict[str, Any]],
        args: argparse.Namespace,
        rng: random.Random) -> None:
    shard_paths = [strategy_dir / f"{args.output_prefix}_{idx:05d}.jsonl" for idx in range(args.num_shards)]
    holdout_paths = [strategy_dir / f"holdout_dirty_roundabout_{idx:05d}.jsonl" for idx in range(args.num_shards)]
    total_written = 0
    total_holdout = 0
    with ExitStackFiles(shard_paths) as shard_files:
        holdout_context = ExitStackFiles(holdout_paths) if args.write_holdout else NullFiles()
        with holdout_context as holdout_files:
            next_shard = 0
            next_holdout = 0
            for row in plan_rows:
                lines = list(iter_jsonl_lines(Path(row["file"])))
                if row["action"] == "holdout_dirty_roundabout":
                    if args.write_holdout:
                        for line in lines:
                            holdout_files[next_holdout].write(line)
                            total_holdout += 1
                            next_holdout = (next_holdout + 1) % len(holdout_files)
                    continue
                selected = sample_lines(lines, int(row["target"]), args.upsample_mode, args.shuffle_output, rng)
                for line in selected:
                    shard_files[next_shard].write(line)
                    total_written += 1
                    next_shard = (next_shard + 1) % len(shard_files)
    print(f"{strategy}: wrote {total_written} sampled rows")
    if args.write_holdout:
        print(f"{strategy}: wrote {total_holdout} dirty roundabout holdout rows")


def write_strategy_d_dataset(
        strategy: str,
        strategy_dir: Path,
        label_plan_rows: Sequence[Dict[str, Any]],
        args: argparse.Namespace,
        rng: random.Random) -> None:
    shard_paths = [strategy_dir / f"{args.output_prefix}_{idx:05d}.jsonl" for idx in range(args.num_shards)]
    holdout_paths = [strategy_dir / f"holdout_dirty_roundabout_{idx:05d}.jsonl" for idx in range(args.num_shards)]
    rows_by_file: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in label_plan_rows:
        rows_by_file[str(row["file"])].append(row)

    total_written = 0
    total_holdout = 0
    with ExitStackFiles(shard_paths) as shard_files:
        holdout_context = ExitStackFiles(holdout_paths) if args.write_holdout else NullFiles()
        with holdout_context as holdout_files:
            next_shard = 0
            next_holdout = 0
            for file_path, file_plan_rows in rows_by_file.items():
                all_lines = list(iter_jsonl_lines(Path(file_path)))
                if any(row["action"] == "holdout_dirty_roundabout" for row in file_plan_rows):
                    if args.write_holdout:
                        for line in all_lines:
                            holdout_files[next_holdout].write(line)
                            total_holdout += 1
                            next_holdout = (next_holdout + 1) % len(holdout_files)
                    continue
                lines_by_label = group_lines_by_decision_label(all_lines)
                for row in file_plan_rows:
                    label = str(row["label"])
                    selected = sample_lines(
                        lines_by_label.get(label, []),
                        int(row["target"]),
                        args.upsample_mode,
                        args.shuffle_output,
                        rng,
                    )
                    for line in selected:
                        shard_files[next_shard].write(line)
                        total_written += 1
                        next_shard = (next_shard + 1) % len(shard_files)
    print(f"{strategy}: wrote {total_written} sampled rows")
    if args.write_holdout:
        print(f"{strategy}: wrote {total_holdout} dirty roundabout holdout rows")


def group_lines_by_decision_label(lines: Sequence[str]) -> Dict[str, List[str]]:
    lines_by_label: DefaultDict[str, List[str]] = defaultdict(list)
    for line in lines:
        label = UNKNOWN
        try:
            obj = json.loads(line)
            label = sample_decision_label(obj) or UNKNOWN
        except json.JSONDecodeError:
            pass
        lines_by_label[label].append(line)
    return dict(lines_by_label)


def sample_lines(
        lines: Sequence[str],
        target: int,
        upsample_mode: str,
        shuffle_output: bool,
        rng: random.Random) -> List[str]:
    available = len(lines)
    if target <= 0 or available == 0:
        return []
    if target <= available:
        selected = rng.sample(list(lines), target) if target < available else list(lines)
    elif upsample_mode == "with_replacement":
        selected = [rng.choice(lines) for _ in range(target)]
    else:
        selected = []
        full_repeats = target // available
        remainder = target % available
        for _ in range(full_repeats):
            selected.extend(lines)
        if remainder:
            selected.extend(rng.sample(list(lines), remainder))
    if shuffle_output:
        rng.shuffle(selected)
    return selected


class ExitStackFiles:
    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = list(paths)
        self.files = []

    def __enter__(self) -> List[Any]:
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.files.append(path.open("w", encoding="utf-8"))
        return self.files

    def __exit__(self, exc_type, exc, tb) -> None:
        for file_obj in self.files:
            file_obj.close()


class NullFiles:
    def __enter__(self) -> List[Any]:
        return []

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def build_summary(
        strategy: str,
        plan_rows: Sequence[Dict[str, Any]],
        total_target: int,
        label_counts: Counter,
        args: argparse.Namespace,
        label_plan_rows: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    action_counts: Counter = Counter(row["action"] for row in plan_rows)
    action_targets: Counter = Counter()
    label_targets: Counter = Counter()
    for row in plan_rows:
        if row["action"] == "holdout_dirty_roundabout":
            continue
        action_targets[row["action"]] += int(row["target"])
        label_targets[row["dominant_label"]] += int(row["target"])
    label_action_counts: Counter = Counter()
    label_action_targets: Counter = Counter()
    if label_plan_rows is not None:
        label_targets = Counter()
        for row in label_plan_rows:
            label_action_counts[row["action"]] += 1
            if row["action"] == "holdout_dirty_roundabout":
                continue
            label_action_targets[row["action"]] += int(row["target"])
            label_targets[row["label"]] += int(row["target"])
    summary = {
        "strategy": strategy,
        "total_target": total_target,
        "num_groups": len(plan_rows),
        "action_counts": dict(action_counts),
        "action_targets": dict(action_targets),
        "label_targets": dict(label_targets.most_common()),
        "label_counts": dict(label_counts.most_common()),
        "args": {
            "purity_high_threshold": args.purity_high_threshold,
            "purity_mid_threshold": args.purity_mid_threshold,
            "roundabout_regex": args.roundabout_regex,
            "roundabout_ratio_threshold": args.roundabout_ratio_threshold,
            "roundabout_low_purity_threshold": args.roundabout_low_purity_threshold,
            "high_label_count": args.high_label_count,
            "low_label_count": args.low_label_count,
            "d_high_purity_threshold": args.d_high_purity_threshold,
            "d_mid_purity_low": args.d_mid_purity_low,
            "d_dominant_label_ratio": args.d_dominant_label_ratio,
            "d_mid_label_target_share": args.d_mid_label_target_share,
            "d_min_label_target_count": args.d_min_label_target_count,
        },
    }
    if label_plan_rows is not None:
        summary["num_label_groups"] = len(label_plan_rows)
        summary["label_action_counts"] = dict(label_action_counts)
        summary["label_action_targets"] = dict(label_action_targets)
        summary["files"] = {
            "leaf_sampling_plan": "sampling_plan.csv",
            "label_sampling_plan": "label_sampling_plan.csv",
        }
    return summary


def is_roundabout_sample(row: Dict[str, Any], pattern: re.Pattern) -> bool:
    return bool(pattern.search(user_prompt_text(row)))


def user_prompt_text(row: Dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return messages
    if not isinstance(messages, list):
        return ""
    return "\n".join(str(message.get("content", "")) for message in messages if message.get("role") == "user")


def sample_decision_label(row: Dict[str, Any]) -> Optional[str]:
    messages = row.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return None
    if not isinstance(messages, list):
        return None
    _, output_messages = split_input_output(messages)
    output_text = "\n".join(str(message.get("content", "")) for message in output_messages)
    parsed = parse_vla_output(output_text)
    lateral = parsed.get("lateral")
    longitudinal = parsed.get("longitudinal")
    if not lateral or not longitudinal:
        return None
    return f"{lateral}|{longitudinal}"


def iter_jsonl_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if not line.endswith("\n"):
                line += "\n"
            yield line


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
