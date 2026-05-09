#!/usr/bin/env python3
"""Diagnose adaptive buckets from TensorBoard channel loss and eval prediction JSONL files."""

import argparse
import csv
import glob
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from analyze_channel_loss_purity import (
    filter_channel_scalars,
    load_loss_csv,
    load_tensorboard_scalars,
    parse_leaf_id_from_channel,
)
from analyze_cluster_mining import safe_div, write_csv


# =========================
# Config area
# =========================
OUTPUT_DIR = ""

# Fill one or more experiments here, or pass --experiments-json.
# eval_json_glob can be a string glob or a list of JSONL paths/globs, one prediction file per checkpoint.
# Every JSONL row is expected to contain response, labels, and id fields.
EXPERIMENTS = [
    {
        "name": "",
        "tb_dir": "",
        "loss_csv": "",
        "eval_json_glob": "",
        "eval_leaf_assignments": "",
        "split_manifest": "",
        "sampling_plan": "",
        "label_sampling_plan": "",
    },
]

TAG_PREFIX = "loss_"
MODE_PREFIX = "train/"
MIN_LOSS_POINTS = 5
WINDOW_POINTS = 5
TAIL_FRACTION = 0.2

PRIMARY_EVAL_METRIC = ""  # empty means auto-select
EVAL_METRIC_HIGHER_IS_BETTER = True
MIN_EVAL_COUNT = 20
PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"

EASY_LOSS_THRESHOLD = 0.25
HIGH_LOSS_THRESHOLD = 0.8
NOISY_LOSS_THRESHOLD = 1.0
LOW_EVAL_THRESHOLD = 0.6
HIGH_EVAL_THRESHOLD = 0.85
TAIL_SLOPE_EPS_PER_1K = 0.02
HIGH_VOLATILITY_THRESHOLD = 0.12
THRESHOLD_MODE = "manual"  # manual or auto
# =========================


UNKNOWN = "UNKNOWN"
DEFAULT_STEP_RE = re.compile(r"(?:step|ckpt|checkpoint|global_step)[-_]?(\d+)", re.IGNORECASE)
SPECIAL_TOKEN_RE = re.compile(r"<([^<>]+)>")
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
PREFERRED_METRICS = [
    "decision_acc",
    "lateral_acc",
    "longitudinal_acc",
    "both_present_ratio",
    "parse_success_ratio",
]
SAMPLE_ID_KEYS = ("sample_id", "eval_sample_id", "id", "uid", "sample_token", "eval_sample_token")

LATERAL_DECISION_NAME_TO_TOKEN = {
    "左换道": "LAT_LANE_CHANGE_LEFT",
    "右换道": "LAT_LANE_CHANGE_RIGHT",
    "左避让": "LAT_NUDGE_LEFT",
    "右避让": "LAT_NUDGE_RIGHT",
    "车道居中": "LAT_LANE_KEEP",
    "车道内居中": "LAT_LANE_KEEP",
    "路口顺行": "LAT_INTERSECTION_FOLLOW",
    "路口左转": "LAT_TURN_LEFT",
    "路口右转": "LAT_TURN_RIGHT",
    "路口掉头": "LAT_U_TURN",
}
LATERAL_DECISION_TOKENS = [
    "LAT_LANE_CHANGE_LEFT",
    "LAT_LANE_CHANGE_RIGHT",
    "LAT_NUDGE_LEFT",
    "LAT_NUDGE_RIGHT",
    "LAT_LANE_KEEP",
    "LAT_INTERSECTION_FOLLOW",
    "LAT_TURN_LEFT",
    "LAT_TURN_RIGHT",
    "LAT_U_TURN",
]
LONGITUDINAL_DECISION_NAME_TO_TOKEN = {
    "保持": "LON_MAINTAIN",
    "加速": "LON_ACCELERATE",
    "减速": "LON_DECELERATE",
    "停车": "LON_STOP",
}
LONGITUDINAL_DECISION_TOKENS = [
    "LON_MAINTAIN",
    "LON_ACCELERATE",
    "LON_DECELERATE",
    "LON_STOP",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join bucket channel-loss curves with per-checkpoint eval prediction JSONL files and produce "
            "bucket-level diagnosis recommendations. Defaults come from the config area at the top."))
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--experiments-json",
        default=None,
        help=(
            "Optional JSON list of experiments. Each item supports name, optional tb_dir/loss_csv, "
            "eval_json_glob as a string or list, eval_leaf_assignments, split_manifest, sampling_plan, "
            "label_sampling_plan."))
    parser.add_argument("--tag-prefix", default=TAG_PREFIX, help="TensorBoard channel scalar prefix. Default: loss_.")
    parser.add_argument("--mode-prefix", default=MODE_PREFIX, help="TensorBoard mode prefix. Default: train/.")
    parser.add_argument("--min-loss-points", type=int, default=MIN_LOSS_POINTS)
    parser.add_argument("--window-points", type=int, default=WINDOW_POINTS)
    parser.add_argument("--tail-fraction", type=float, default=TAIL_FRACTION)
    parser.add_argument("--primary-eval-metric", default=PRIMARY_EVAL_METRIC, help="Metric to diagnose. Empty=auto.")
    parser.add_argument(
        "--eval-lower-is-better",
        action="store_true",
        help="Set when primary eval metric is an error/loss metric.")
    parser.add_argument("--min-eval-count", type=int, default=MIN_EVAL_COUNT)
    parser.add_argument("--easy-loss-threshold", type=float, default=EASY_LOSS_THRESHOLD)
    parser.add_argument("--high-loss-threshold", type=float, default=HIGH_LOSS_THRESHOLD)
    parser.add_argument("--noisy-loss-threshold", type=float, default=NOISY_LOSS_THRESHOLD)
    parser.add_argument("--low-eval-threshold", type=float, default=LOW_EVAL_THRESHOLD)
    parser.add_argument("--high-eval-threshold", type=float, default=HIGH_EVAL_THRESHOLD)
    parser.add_argument("--tail-slope-eps-per-1k", type=float, default=TAIL_SLOPE_EPS_PER_1K)
    parser.add_argument("--high-volatility-threshold", type=float, default=HIGH_VOLATILITY_THRESHOLD)
    parser.add_argument(
        "--threshold-mode",
        choices=("manual", "auto"),
        default=THRESHOLD_MODE,
        help="manual uses the explicit thresholds; auto derives thresholds from bucket distributions.")
    parser.add_argument("--plotly-js", default=PLOTLY_JS, help="Plotly.js URL/path for the dashboard.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = load_experiments(args)
    all_loss_rows = []
    all_loss_point_rows = []
    all_eval_rows = []
    all_diag_rows = []
    all_unmatched_eval_rows = []
    summaries = []

    for exp in experiments:
        name = str(exp["name"])
        leaf_meta = load_leaf_meta(exp)
        eval_assignments = load_eval_assignments(Path(exp["eval_leaf_assignments"]).expanduser())
        loss_rows, loss_point_rows = load_experiment_loss(exp, leaf_meta, args)
        eval_rows, unmatched_eval_rows = load_experiment_eval(exp, eval_assignments)
        primary_metric = choose_primary_metric(eval_rows, args.primary_eval_metric)
        diag_rows, thresholds = diagnose_experiment(name, leaf_meta, loss_rows, eval_rows, primary_metric, args)

        all_loss_rows.extend(loss_rows)
        all_loss_point_rows.extend(loss_point_rows)
        all_eval_rows.extend(sorted(eval_rows, key=lambda row: (str(row.get("experiment", "")), int(row.get("step", 0)),
                                                               int(row.get("leaf_id") or -1))))
        all_diag_rows.extend(diag_rows)
        all_unmatched_eval_rows.extend(unmatched_eval_rows)
        summaries.append(build_experiment_summary(name, diag_rows, eval_rows, loss_rows, primary_metric, thresholds))

    write_csv(output_dir / "bucket_loss_metrics.csv", all_loss_rows)
    write_csv(output_dir / "bucket_loss_timeseries.csv", all_loss_point_rows)
    write_csv(output_dir / "bucket_eval_timeseries.csv", all_eval_rows)
    write_csv(output_dir / "bucket_diagnosis.csv", all_diag_rows)
    write_csv(output_dir / "bucket_eval_unmatched.csv", all_unmatched_eval_rows)
    write_csv(output_dir / "bucket_diagnosis_action_summary.csv", build_action_summary(all_diag_rows))
    dashboard_data_dir = output_dir / "bucket_diagnosis_dashboard_data"
    dashboard_data_index = write_dashboard_data(dashboard_data_dir, all_eval_rows, all_loss_point_rows)
    summary = {
        "num_experiments": len(experiments),
        "experiments": summaries,
        "outputs": {
            "bucket_loss_metrics": "bucket_loss_metrics.csv",
            "bucket_loss_timeseries": "bucket_loss_timeseries.csv",
            "bucket_eval_timeseries": "bucket_eval_timeseries.csv",
            "bucket_diagnosis": "bucket_diagnosis.csv",
            "bucket_eval_unmatched": "bucket_eval_unmatched.csv",
            "bucket_diagnosis_action_summary": "bucket_diagnosis_action_summary.csv",
            "bucket_diagnosis_dashboard": "bucket_diagnosis_dashboard.html",
            "bucket_diagnosis_dashboard_data": "bucket_diagnosis_dashboard_data/",
        },
    }
    write_json(output_dir / "bucket_diagnosis_summary.json", summary)
    (output_dir / "bucket_diagnosis_summary.md").write_text(render_markdown(summary, all_diag_rows), encoding="utf-8")
    (output_dir / "bucket_diagnosis_dashboard.html").write_text(
        render_dashboard(summary, all_diag_rows, all_loss_rows, dashboard_data_index, args.plotly_js),
        encoding="utf-8")
    print(f"wrote bucket diagnosis for {len(experiments)} experiments -> {output_dir}")


def validate_args(args: argparse.Namespace) -> None:
    if not args.output_dir:
        raise ValueError("Please set OUTPUT_DIR in the config area or pass --output-dir.")
    if args.min_loss_points <= 0:
        raise ValueError("--min-loss-points must be positive")
    if args.window_points <= 0:
        raise ValueError("--window-points must be positive")
    if not 0 < args.tail_fraction <= 1:
        raise ValueError("--tail-fraction must be in (0, 1]")
    if args.min_eval_count < 0:
        raise ValueError("--min-eval-count must be non-negative")


def load_experiments(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.experiments_json:
        payload = read_json(Path(args.experiments_json).expanduser())
    else:
        payload = EXPERIMENTS
    if not isinstance(payload, list):
        raise ValueError("experiments config must be a JSON list")
    rows = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"experiment #{idx} is not an object")
        exp = dict(item)
        exp["name"] = str(exp.get("name") or f"exp_{idx:02d}")
        if not exp.get("eval_json_glob"):
            raise ValueError(f"experiment {exp['name']} missing eval_json_glob")
        if not exp.get("eval_leaf_assignments"):
            raise ValueError(f"experiment {exp['name']} missing eval_leaf_assignments")
        rows.append(exp)
    return rows


def load_leaf_meta(exp: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    meta: Dict[int, Dict[str, Any]] = {}
    if exp.get("split_manifest"):
        merge_leaf_meta(meta, load_split_manifest_meta(Path(exp["split_manifest"]).expanduser()))
    if exp.get("sampling_plan"):
        merge_leaf_meta(meta, load_sampling_plan_meta(Path(exp["sampling_plan"]).expanduser()))
    if exp.get("label_sampling_plan"):
        merge_leaf_meta(meta, load_label_sampling_plan_meta(Path(exp["label_sampling_plan"]).expanduser()))
    return meta


def merge_leaf_meta(dst: Dict[int, Dict[str, Any]], src: Dict[int, Dict[str, Any]]) -> None:
    for leaf_id, row in src.items():
        if leaf_id not in dst:
            dst[leaf_id] = {}
        dst[leaf_id].update({key: value for key, value in row.items() if value not in (None, "")})


def load_split_manifest_meta(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_csv_dicts(path)
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        for leaf_id in parse_leaf_ids(row.get("leaf_ids", "")):
            out[leaf_id] = {
                "leaf_id": leaf_id,
                "raw_count": to_int(row.get("count")),
                "sampled_target": to_int(row.get("suggested_target_count")),
                "sample_ratio": safe_div(to_int(row.get("suggested_target_count")), to_int(row.get("count"))),
                "purity": to_float(row.get("avg_leaf_purity")),
                "dominant_label": row.get("dominant_label", ""),
                "dominant_label_ratio": to_float(row.get("dominant_label_ratio")),
                "primary_scene_top": row.get("primary_scene_top", ""),
                "secondary_scene_top": row.get("secondary_scene_top", ""),
                "suggested_action": row.get("suggested_action_top", ""),
                "stop_reason": row.get("stop_reason_top", ""),
            }
    return out


def load_sampling_plan_meta(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for row in read_csv_dicts(path):
        for leaf_id in parse_leaf_ids(row.get("leaf_ids", "")):
            out[leaf_id] = {
                "leaf_id": leaf_id,
                "raw_count": to_int(row.get("available")),
                "sampled_target": to_int(row.get("target")),
                "sample_ratio": to_float(row.get("sample_ratio")),
                "purity": to_float(row.get("purity")),
                "dominant_label": row.get("dominant_label", ""),
                "dominant_label_ratio": to_float(row.get("dominant_label_ratio_in_leaf")),
                "sampling_action": row.get("action", ""),
                "sampling_reason": row.get("reason", ""),
                "primary_scene_top": row.get("primary_scene_top", ""),
                "secondary_scene_top": row.get("secondary_scene_top", ""),
                "suggested_action": row.get("suggested_action_top", ""),
                "stop_reason": row.get("stop_reason_top", ""),
            }
    return out


def load_label_sampling_plan_meta(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    per_leaf: DefaultDict[int, Counter] = defaultdict(Counter)
    for row in read_csv_dicts(path):
        for leaf_id in parse_leaf_ids(row.get("leaf_ids", "")):
            per_leaf[leaf_id]["label_groups"] += 1
            per_leaf[leaf_id][f"label_action::{row.get('action', '')}"] += 1
            per_leaf[leaf_id]["label_target_total"] += to_int(row.get("target"))
    out = {}
    for leaf_id, counter in per_leaf.items():
        action_counts = Counter()
        for key, count in counter.items():
            if str(key).startswith("label_action::"):
                action_counts[str(key)[len("label_action::"):]] = count
        out[leaf_id] = {
            "leaf_id": leaf_id,
            "label_groups": int(counter["label_groups"]),
            "label_target_total": int(counter["label_target_total"]),
            "label_sampling_actions": format_counter(action_counts, 8),
        }
    return out


def load_eval_assignments(path: Path) -> Dict[str, Dict[str, Any]]:
    assignments: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        scene_id = row.get("eval_scene_id")
        sample_token = row.get("eval_sample_token")
        keys = [
            row.get("eval_sample_id"),
            row.get("eval_sample_token"),
            row.get("eval_row"),
            row.get("eval_row_idx"),
        ]
        if scene_id not in (None, "") and sample_token not in (None, ""):
            keys.append(f"{scene_id}_{sample_token}")
        for key in keys:
            if key not in (None, ""):
                assignments[str(key)] = row
    return assignments


def load_experiment_loss(
        exp: Dict[str, Any],
        leaf_meta: Dict[int, Dict[str, Any]],
        args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not (exp.get("loss_csv") or exp.get("tb_dir")):
        return [], []
    raw_points = load_loss_csv(Path(exp["loss_csv"]).expanduser()) if exp.get("loss_csv") else load_tensorboard_scalars(
        exp["tb_dir"])
    channel_points = filter_channel_scalars(raw_points, args.tag_prefix, args.mode_prefix)
    rows = []
    point_rows = []
    for channel, points in sorted(channel_points.items()):
        leaf_id = parse_leaf_id_from_channel(channel)
        if leaf_id is None:
            continue
        clean_points = [(int(step), float(value)) for step, value in points if math.isfinite(float(value))]
        for step, value in clean_points:
            point_rows.append({
                "experiment": exp["name"],
                "leaf_id": leaf_id,
                "channel": channel,
                "step": step,
                "loss": value,
            })
        if len(clean_points) < args.min_loss_points:
            continue
        metrics = summarize_loss_curve(clean_points, args.window_points, args.tail_fraction)
        meta = leaf_meta.get(leaf_id, {})
        rows.append({
            "experiment": exp["name"],
            "leaf_id": leaf_id,
            "channel": channel,
            **leaf_meta_fields(meta),
            **metrics,
        })
    return rows, point_rows


def summarize_loss_curve(points: Sequence[Tuple[int, float]], window_points: int, tail_fraction: float) -> Dict[str, Any]:
    points = sorted(points)
    window = max(1, min(window_points, len(points)))
    tail_n = max(window, int(math.ceil(len(points) * tail_fraction)))
    steps = [int(step) for step, _ in points]
    values = [float(value) for _, value in points]
    tail_steps = steps[-tail_n:]
    tail_values = values[-tail_n:]
    first_loss = mean(values[:window])
    final_loss = mean(values[-window:])
    tail_first = mean(tail_values[:window])
    tail_final = mean(tail_values[-window:])
    return {
        "loss_num_points": len(points),
        "loss_first_step": steps[0],
        "loss_last_step": steps[-1],
        "loss_first": first_loss,
        "loss_final": final_loss,
        "loss_min": min(values),
        "loss_max": max(values),
        "loss_delta": final_loss - first_loss,
        "loss_relative_drop": safe_div(first_loss - final_loss, first_loss),
        "loss_slope_per_1k": linear_slope(steps, values) * 1000.0,
        "loss_tail_slope_per_1k": linear_slope(tail_steps, tail_values) * 1000.0,
        "loss_tail_delta": tail_final - tail_first,
        "loss_tail_std": std(tail_values),
        "loss_tail_mean": mean(tail_values),
        "loss_gap_to_min": final_loss - min(values),
        "loss_auc_mean": mean(values),
    }


def load_experiment_eval(
        exp: Dict[str, Any],
        eval_assignments: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eval_paths = resolve_eval_paths(exp["eval_json_glob"])
    rows = []
    unmatched = []
    for path in eval_paths:
        step = infer_step_from_eval(path)
        records = read_eval_records(path)
        if not records:
            rows.append({
                "experiment": exp["name"],
                "step": step,
                "leaf_id": "",
                "eval_count": 0,
                "eval_file": str(path),
            })
            continue
        leaf_metric_values: DefaultDict[int, DefaultDict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))
        leaf_counts: Counter = Counter()
        for rec in records:
            key = record_assignment_key(rec)
            assignment = eval_assignments.get(key)
            if assignment is None:
                unmatched.append({
                    "experiment": exp["name"],
                    "step": step,
                    "eval_file": str(path),
                    "record_key": key,
                })
                continue
            leaf_id = int(assignment["leaf_id"])
            metrics = eval_prediction_metrics(rec)
            for metric, value in metrics.items():
                leaf_metric_values[leaf_id][metric].append(value)
            leaf_counts[leaf_id] += 1
        for leaf_id, metric_values in sorted(leaf_metric_values.items()):
            item = {
                "experiment": exp["name"],
                "step": step,
                "leaf_id": leaf_id,
                "eval_count": int(leaf_counts[leaf_id]),
                "eval_file": str(path),
            }
            for metric, values in sorted(metric_values.items()):
                if metric.endswith("_counter"):
                    item[metric] = format_counter(Counter(str(value) for value in values), 10)
                else:
                    item[metric] = mean([to_float(value) for value in values])
            rows.append(item)
    return rows, unmatched


def resolve_eval_paths(value: Any) -> List[Path]:
    if isinstance(value, (list, tuple)):
        patterns = value
    else:
        patterns = [value]
    paths = []
    seen = set()
    for pattern in patterns:
        if pattern in (None, ""):
            continue
        expanded = sorted(glob.glob(str(Path(str(pattern)).expanduser())))
        if not expanded:
            expanded = [str(Path(str(pattern)).expanduser())]
        for item in expanded:
            path = Path(item)
            key = str(path)
            if key not in seen:
                paths.append(path)
                seen.add(key)
    return sorted(paths, key=lambda path: (infer_step_from_eval(path), str(path)))


def read_eval_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return list(iter_jsonl(path))
    data = read_json(path)
    records = extract_record_list(data)
    if records is None and isinstance(data, list):
        records = [item for item in data if isinstance(item, dict)]
    if records is None and isinstance(data, dict):
        records = dict_values_as_records(data)
    return records or []


def extract_record_list(data: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(data, dict):
        return None
    for key in ("records", "samples", "details", "predictions", "results", "data", "items"):
        value = data.get(key)
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    for value in data.values():
        if isinstance(value, dict):
            nested = extract_record_list(value)
            if nested:
                return nested
    return None


def dict_values_as_records(data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    rows = []
    for key, value in data.items():
        if isinstance(value, dict):
            item = dict(value)
            if not any(sample_key in item for sample_key in SAMPLE_ID_KEYS):
                item["sample_id"] = key
            rows.append(item)
    return rows or None


def record_assignment_key(record: Dict[str, Any]) -> str:
    record_id = get_by_dotted_key(record, "id")
    if record_id not in (None, ""):
        return str(record_id)
    scene_id = get_by_dotted_key(record, "scene_id")
    sample_token = get_by_dotted_key(record, "sample_token")
    if scene_id not in (None, "") and sample_token not in (None, ""):
        return f"{scene_id}_{sample_token}"
    for key in SAMPLE_ID_KEYS:
        value = get_by_dotted_key(record, key)
        if value not in (None, ""):
            return str(value)
    for key in ("row", "eval_row", "idx", "index"):
        value = get_by_dotted_key(record, key)
        if value not in (None, ""):
            return str(value)
    return ""


def eval_prediction_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    response = str(record.get("response") or "")
    labels = str(record.get("labels") or "")
    pred_lateral, pred_longitudinal = extract_decisions_from_response(response)
    gt_lateral_set, gt_longitudinal_set = extract_decision_gt_from_labels(labels)
    lateral_has_gt = bool(gt_lateral_set)
    longitudinal_has_gt = bool(gt_longitudinal_set)
    lateral_correct = lateral_has_gt and pred_lateral in gt_lateral_set
    longitudinal_correct = longitudinal_has_gt and pred_longitudinal in gt_longitudinal_set
    decision_has_gt = lateral_has_gt and longitudinal_has_gt
    decision_correct = decision_has_gt and lateral_correct and longitudinal_correct
    return {
        "lateral_acc": 1.0 if lateral_correct else 0.0 if lateral_has_gt else None,
        "longitudinal_acc": 1.0 if longitudinal_correct else 0.0 if longitudinal_has_gt else None,
        "decision_acc": 1.0 if decision_correct else 0.0 if decision_has_gt else None,
        "lateral_has_gt_ratio": 1.0 if lateral_has_gt else 0.0,
        "longitudinal_has_gt_ratio": 1.0 if longitudinal_has_gt else 0.0,
        "both_present_ratio": 1.0 if pred_lateral and pred_longitudinal else 0.0,
        "parse_success_ratio": 1.0 if pred_lateral or pred_longitudinal else 0.0,
        "pred_lateral_counter": pred_lateral or "NONE",
        "pred_longitudinal_counter": pred_longitudinal or "NONE",
        "gt_lateral_counter": "|".join(sorted(gt_lateral_set)) if gt_lateral_set else "NONE",
        "gt_longitudinal_counter": "|".join(sorted(gt_longitudinal_set)) if gt_longitudinal_set else "NONE",
    }


def extract_decisions_from_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
    lateral_decision = None
    longitudinal_decision = None
    for token in extract_special_tokens(response):
        if lateral_decision is None:
            lateral_decision = normalize_lateral_decision(token)
        if longitudinal_decision is None:
            longitudinal_decision = normalize_longitudinal_decision(token)
        if lateral_decision is not None and longitudinal_decision is not None:
            break
    return lateral_decision, longitudinal_decision


def extract_decision_gt_from_labels(labels: str) -> Tuple[set, set]:
    answer_match = ANSWER_RE.search(labels or "")
    answer_content = answer_match.group(1) if answer_match else labels
    lateral_gt_set = set()
    longitudinal_gt_set = set()
    for result in str(answer_content or "").split(";"):
        tokens = extract_special_tokens(result)
        if not tokens:
            continue
        lateral = normalize_lateral_decision(tokens[0])
        if lateral is not None:
            lateral_gt_set.add(lateral)
        longitudinal = normalize_longitudinal_decision(tokens[1] if len(tokens) >= 2 else None)
        if longitudinal is not None:
            longitudinal_gt_set.add(longitudinal)
    return lateral_gt_set, longitudinal_gt_set


def extract_special_tokens(text: str) -> List[str]:
    return [token.strip() for token in SPECIAL_TOKEN_RE.findall(text or "")]


def normalize_lateral_decision(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip().strip("<>").strip()
    if value in LATERAL_DECISION_TOKENS:
        return value
    return LATERAL_DECISION_NAME_TO_TOKEN.get(value)


def normalize_longitudinal_decision(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip().strip("<>").strip()
    if value in LONGITUDINAL_DECISION_TOKENS:
        return value
    return LONGITUDINAL_DECISION_NAME_TO_TOKEN.get(value)


def flatten_numeric_metrics(data: Any, prefix: str = "", exclude_lists: bool = False) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_numeric_metrics(value, child_key, exclude_lists))
    elif isinstance(data, list):
        if exclude_lists:
            return {}
        for idx, value in enumerate(data):
            child_key = f"{prefix}.{idx}" if prefix else str(idx)
            out.update(flatten_numeric_metrics(value, child_key, exclude_lists))
    elif isinstance(data, bool):
        if prefix:
            out[prefix] = 1.0 if data else 0.0
    elif isinstance(data, (int, float)) and math.isfinite(float(data)):
        if prefix:
            out[prefix] = float(data)
    return out


def infer_step_from_eval(path: Path) -> int:
    data_step = None
    if path.suffix == ".json":
        try:
            payload = read_json(path)
            if isinstance(payload, dict):
                for key in ("step", "global_step", "ckpt_step", "checkpoint_step"):
                    if key in payload:
                        data_step = to_int(payload[key])
                        break
        except Exception:
            data_step = None
    if data_step is not None:
        return data_step
    match = DEFAULT_STEP_RE.search(str(path))
    if match:
        return int(match.group(1))
    return 0


def choose_primary_metric(eval_rows: Sequence[Dict[str, Any]], configured: str) -> str:
    if configured:
        return configured
    available = Counter()
    for row in eval_rows:
        for key, value in row.items():
            if key in {"experiment", "step", "leaf_id", "eval_count", "eval_file"}:
                continue
            if isinstance(value, (int, float)):
                available[key] += 1
    for metric in PREFERRED_METRICS:
        if metric in available:
            return metric
    return available.most_common(1)[0][0] if available else ""


def diagnose_experiment(
        name: str,
        leaf_meta: Dict[int, Dict[str, Any]],
        loss_rows: Sequence[Dict[str, Any]],
        eval_rows: Sequence[Dict[str, Any]],
        primary_metric: str,
        args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    loss_by_leaf = {int(row["leaf_id"]): row for row in loss_rows if row.get("leaf_id") not in (None, "")}
    eval_summary = summarize_eval_by_leaf(eval_rows, primary_metric, not args.eval_lower_is_better)
    thresholds = build_thresholds(loss_rows, eval_summary, args)
    leaf_ids = sorted(set(loss_by_leaf) | set(eval_summary) | set(leaf_meta))
    rows = []
    for leaf_id in leaf_ids:
        loss = loss_by_leaf.get(leaf_id, {})
        eval_item = eval_summary.get(leaf_id, {})
        meta = leaf_meta.get(leaf_id, {})
        recommendation, reason = recommend_bucket(loss, eval_item, meta, primary_metric, thresholds, args)
        rows.append({
            "experiment": name,
            "leaf_id": leaf_id,
            **leaf_meta_fields(meta),
            **prefixed(loss, include_prefix=False, skip={"experiment", "leaf_id"}),
            "primary_eval_metric": primary_metric,
            "threshold_mode": thresholds["threshold_mode"],
            **eval_item,
            "recommendation": recommendation,
            "diagnosis_reason": reason,
        })
    rows.sort(key=lambda row: (row["recommendation"], -float_or(row.get("loss_final"), -1.0), int(row["leaf_id"])))
    return rows, thresholds


def build_thresholds(
        loss_rows: Sequence[Dict[str, Any]],
        eval_summary: Dict[int, Dict[str, Any]],
        args: argparse.Namespace) -> Dict[str, Any]:
    thresholds = {
        "threshold_mode": args.threshold_mode,
        "easy_loss_threshold": args.easy_loss_threshold,
        "high_loss_threshold": args.high_loss_threshold,
        "noisy_loss_threshold": args.noisy_loss_threshold,
        "low_eval_threshold": args.low_eval_threshold,
        "high_eval_threshold": args.high_eval_threshold,
        "tail_slope_eps_per_1k": args.tail_slope_eps_per_1k,
        "high_volatility_threshold": args.high_volatility_threshold,
    }
    if args.threshold_mode != "auto":
        return thresholds

    final_losses = clean_floats(row.get("loss_final") for row in loss_rows)
    tail_slopes = clean_floats(row.get("loss_tail_slope_per_1k") for row in loss_rows)
    tail_stds = clean_floats(row.get("loss_tail_std") for row in loss_rows)
    eval_finals = clean_floats(row.get("eval_metric_final") for row in eval_summary.values())

    if final_losses:
        thresholds["easy_loss_threshold"] = percentile(final_losses, 0.25)
        thresholds["high_loss_threshold"] = percentile(final_losses, 0.75)
        thresholds["noisy_loss_threshold"] = max(thresholds["high_loss_threshold"], percentile(final_losses, 0.90))
    if eval_finals:
        low_eval = percentile(eval_finals, 0.25)
        high_eval = percentile(eval_finals, 0.75)
        thresholds["low_eval_threshold"] = min(low_eval, high_eval)
        thresholds["high_eval_threshold"] = max(low_eval, high_eval)
    if tail_slopes:
        abs_slopes = [abs(value) for value in tail_slopes]
        thresholds["tail_slope_eps_per_1k"] = max(1e-8, percentile(abs_slopes, 0.30))
    if tail_stds:
        thresholds["high_volatility_threshold"] = percentile(tail_stds, 0.75)
    return thresholds


def summarize_eval_by_leaf(
        eval_rows: Sequence[Dict[str, Any]],
        primary_metric: str,
        higher_is_better: bool) -> Dict[int, Dict[str, Any]]:
    by_leaf: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        if row.get("leaf_id") in (None, ""):
            continue
        by_leaf[int(row["leaf_id"])].append(row)
    out = {}
    for leaf_id, rows in by_leaf.items():
        rows = sorted(rows, key=lambda row: int(row.get("step", 0)))
        metric_values = [to_float(row.get(primary_metric)) for row in rows if row.get(primary_metric) not in (None, "")]
        final_row = rows[-1]
        final_metric = to_float(final_row.get(primary_metric))
        best_metric = best_value(metric_values, higher_is_better)
        first_metric = metric_values[0] if metric_values else None
        item = {
            "eval_num_steps": len(rows),
            "eval_first_step": int(rows[0].get("step", 0)),
            "eval_last_step": int(final_row.get("step", 0)),
            "eval_count_final": to_int(final_row.get("eval_count")),
            "eval_metric_first": first_metric,
            "eval_metric_final": final_metric,
            "eval_metric_best": best_metric,
            "eval_metric_delta": none_sub(final_metric, first_metric),
            "eval_metric_slope_per_1k": linear_slope(
                [int(row.get("step", 0)) for row in rows if row.get(primary_metric) not in (None, "")],
                [to_float(row.get(primary_metric)) for row in rows if row.get(primary_metric) not in (None, "")],
            ) * 1000.0 if len(metric_values) >= 2 else None,
        }
        for metric in numeric_eval_metric_names(rows):
            values = [to_float(row.get(metric)) for row in rows if row.get(metric) not in (None, "")]
            if not values:
                continue
            metric_higher_is_better = higher_is_better
            final_value = to_float(final_row.get(metric))
            first_value = values[0]
            item[f"eval_first_{metric}"] = first_value
            item[f"eval_final_{metric}"] = final_value
            item[f"eval_best_{metric}"] = best_value(values, metric_higher_is_better)
            item[f"eval_delta_{metric}"] = none_sub(final_value, first_value)
        out[leaf_id] = item
    return out


def numeric_eval_metric_names(rows: Sequence[Dict[str, Any]]) -> List[str]:
    names = set()
    skip = {"experiment", "step", "leaf_id", "eval_count", "eval_file"}
    for row in rows:
        for key, value in row.items():
            if key in skip or str(key).endswith("_counter"):
                continue
            if isinstance(value, (int, float)):
                names.add(str(key))
    return sorted(names)


def recommend_bucket(
        loss: Dict[str, Any],
        eval_item: Dict[str, Any],
        meta: Dict[str, Any],
        primary_metric: str,
        thresholds: Dict[str, Any],
        args: argparse.Namespace) -> Tuple[str, str]:
    eval_count = to_int(eval_item.get("eval_count_final"))
    has_eval = primary_metric and eval_item.get("eval_metric_final") is not None and eval_count >= args.min_eval_count
    eval_final = to_float(eval_item.get("eval_metric_final"))
    eval_delta = to_float(eval_item.get("eval_metric_delta"))
    if not loss:
        if not has_eval:
            return "eval_only_insufficient_samples", (
                f"no channel-loss curve; eval_count={eval_count}, min_eval_count={args.min_eval_count}")
        if args.eval_lower_is_better:
            if eval_final <= thresholds["low_eval_threshold"]:
                return "eval_only_good", f"no channel-loss curve; eval={eval_final}"
            if eval_final >= thresholds["high_eval_threshold"]:
                return "eval_only_review", f"no channel-loss curve; low eval quality={eval_final}"
        else:
            if eval_final >= thresholds["high_eval_threshold"]:
                return "eval_only_good", f"no channel-loss curve; eval={eval_final}"
            if eval_final <= thresholds["low_eval_threshold"]:
                return "eval_only_review", f"no channel-loss curve; low eval={eval_final}"
        return "eval_only_mixed", f"no channel-loss curve; eval={eval_final}"
    if primary_metric and not has_eval:
        return "insufficient_eval_samples", f"eval_count={eval_count}, min_eval_count={args.min_eval_count}"

    final_loss = to_float(loss.get("loss_final"))
    tail_slope = to_float(loss.get("loss_tail_slope_per_1k"))
    tail_std = to_float(loss.get("loss_tail_std"))
    rel_drop = to_float(loss.get("loss_relative_drop"))
    if args.eval_lower_is_better:
        high_eval = not has_eval or eval_final <= thresholds["low_eval_threshold"]
        low_eval = has_eval and eval_final >= thresholds["high_eval_threshold"]
    else:
        high_eval = not has_eval or eval_final >= thresholds["high_eval_threshold"]
        low_eval = has_eval and eval_final <= thresholds["low_eval_threshold"]
    still_descending = tail_slope <= -thresholds["tail_slope_eps_per_1k"]
    plateau = abs(tail_slope) <= thresholds["tail_slope_eps_per_1k"]
    volatile = tail_std >= thresholds["high_volatility_threshold"]

    if final_loss <= thresholds["easy_loss_threshold"] and high_eval:
        return "downsample_more", f"easy bucket: final_loss={final_loss:.4f}, eval={eval_final}"
    if final_loss >= thresholds["noisy_loss_threshold"] and plateau and volatile and low_eval:
        return "suspected_noisy_review", (
            f"high plateau/volatile loss and low eval: loss={final_loss:.4f}, tail_std={tail_std:.4f}, eval={eval_final}")
    if final_loss >= thresholds["high_loss_threshold"] and still_descending:
        return "undertrained_upsample_or_replay", (
            f"loss still descending: final_loss={final_loss:.4f}, tail_slope_per_1k={tail_slope:.4f}")
    if final_loss >= thresholds["high_loss_threshold"] and plateau and low_eval:
        return "hard_or_noisy_review", f"high plateau loss with low eval: loss={final_loss:.4f}, eval={eval_final}"
    if final_loss >= thresholds["high_loss_threshold"] and (eval_delta is None or eval_delta >= 0):
        return "hard_but_valid_curriculum", f"high loss but eval not degrading: loss={final_loss:.4f}, eval_delta={eval_delta}"
    if final_loss <= thresholds["easy_loss_threshold"] and not high_eval:
        return "eval_mismatch_check", f"low train loss but eval not high: loss={final_loss:.4f}, eval={eval_final}"
    if rel_drop < 0.05 and final_loss > thresholds["easy_loss_threshold"]:
        return "low_learning_signal_review", f"small relative loss drop={rel_drop:.4f}"
    return "healthy_keep", f"loss={final_loss:.4f}, eval={eval_final}"


def build_experiment_summary(
        name: str,
        diag_rows: Sequence[Dict[str, Any]],
        eval_rows: Sequence[Dict[str, Any]],
        loss_rows: Sequence[Dict[str, Any]],
        primary_metric: str,
        thresholds: Dict[str, Any]) -> Dict[str, Any]:
    rec_counts = Counter(row["recommendation"] for row in diag_rows)
    return {
        "experiment": name,
        "primary_eval_metric": primary_metric,
        "thresholds": thresholds,
        "num_diagnosed_buckets": len(diag_rows),
        "num_loss_buckets": len(loss_rows),
        "num_eval_bucket_steps": len(eval_rows),
        "recommendation_counts": dict(rec_counts),
    }


def build_action_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: DefaultDict[Tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        key = (str(row["experiment"]), str(row["recommendation"]))
        counts[key]["num_buckets"] += 1
        counts[key]["raw_count"] += to_int(row.get("raw_count"))
        counts[key]["sampled_target"] += to_int(row.get("sampled_target"))
        counts[key]["eval_count_final"] += to_int(row.get("eval_count_final"))
    out = []
    for (experiment, recommendation), counter in sorted(counts.items()):
        out.append({
            "experiment": experiment,
            "recommendation": recommendation,
            "num_buckets": int(counter["num_buckets"]),
            "raw_count": int(counter["raw_count"]),
            "sampled_target": int(counter["sampled_target"]),
            "eval_count_final": int(counter["eval_count_final"]),
        })
    return out


def leaf_meta_fields(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "raw_count": meta.get("raw_count"),
        "sampled_target": meta.get("sampled_target"),
        "sample_ratio": meta.get("sample_ratio"),
        "purity": meta.get("purity"),
        "dominant_label": meta.get("dominant_label"),
        "dominant_label_ratio": meta.get("dominant_label_ratio"),
        "sampling_action": meta.get("sampling_action", ""),
        "sampling_reason": meta.get("sampling_reason", ""),
        "label_groups": meta.get("label_groups", ""),
        "label_target_total": meta.get("label_target_total", ""),
        "label_sampling_actions": meta.get("label_sampling_actions", ""),
        "primary_scene_top": meta.get("primary_scene_top", ""),
        "secondary_scene_top": meta.get("secondary_scene_top", ""),
        "suggested_action": meta.get("suggested_action", ""),
        "stop_reason": meta.get("stop_reason", ""),
    }


def prefixed(row: Dict[str, Any], *, include_prefix: bool, skip: Sequence[str]) -> Dict[str, Any]:
    skip_set = set(skip)
    return {key: value for key, value in row.items() if key not in skip_set}


def parse_leaf_ids(value: Any) -> List[int]:
    ids = []
    for item in str(value or "").split():
        try:
            ids.append(int(float(item)))
        except ValueError:
            continue
    return ids


def get_by_dotted_key(data: Dict[str, Any], key: str) -> Any:
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def best_value(values: Sequence[Optional[float]], higher_is_better: bool) -> Optional[float]:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return max(clean) if higher_is_better else min(clean)


def none_sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def linear_slope(x: Sequence[Optional[float]], y: Sequence[Optional[float]]) -> float:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 2:
        return 0.0
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((v - x_mean) ** 2 for v in xs)
    if denom == 0:
        return 0.0
    return sum((a - x_mean) * (b - y_mean) for a, b in pairs) / denom


def mean(values: Sequence[Optional[float]]) -> float:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else 0.0


def clean_floats(values: Iterable[Any]) -> List[float]:
    out = []
    for value in values:
        parsed = to_float(value)
        if parsed is not None and math.isfinite(parsed):
            out.append(parsed)
    return out


def percentile(values: Sequence[float], q: float) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return 0.0
    if q <= 0:
        return clean[0]
    if q >= 1:
        return clean[-1]
    pos = (len(clean) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return clean[low]
    weight = pos - low
    return clean[low] * (1.0 - weight) + clean[high] * weight


def std(values: Sequence[Optional[float]]) -> float:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return 0.0
    m = mean(clean)
    return math.sqrt(sum((value - m) ** 2 for value in clean) / len(clean))


def format_counter(counter: Counter, topn: int) -> str:
    total = sum(counter.values())
    return ";".join(f"{key}:{count}:{safe_div(count, total):.4f}" for key, count in counter.most_common(topn))


def to_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def float_or(value: Any, default: float) -> float:
    parsed = to_float(value)
    return default if parsed is None else parsed


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e


def render_markdown(summary: Dict[str, Any], diag_rows: Sequence[Dict[str, Any]]) -> str:
    lines = ["# Bucket Diagnosis", ""]
    for exp in summary["experiments"]:
        lines.extend([
            f"## {exp['experiment']}",
            "",
            f"- Primary eval metric: `{exp['primary_eval_metric']}`",
            f"- Diagnosed buckets: {exp['num_diagnosed_buckets']}",
            f"- Loss buckets: {exp['num_loss_buckets']}",
            f"- Eval bucket steps: {exp['num_eval_bucket_steps']}",
            "",
            "| recommendation | buckets |",
            "| --- | ---: |",
        ])
        for rec, count in sorted(exp["recommendation_counts"].items()):
            lines.append(f"| {rec} | {count} |")
        lines.append("")
    lines.extend(["## Top Buckets To Review", ""])
    review_rows = [
        row for row in diag_rows
        if row["recommendation"] in {"suspected_noisy_review", "hard_or_noisy_review", "low_learning_signal_review"}
    ][:50]
    if not review_rows:
        lines.append("- No review buckets.")
    else:
        lines.append("| experiment | leaf | recommendation | loss_final | eval_final | dominant_label | reason |")
        lines.append("| --- | ---: | --- | ---: | ---: | --- | --- |")
        for row in review_rows:
            lines.append(
                f"| {row['experiment']} | {row['leaf_id']} | {row['recommendation']} | "
                f"{fmt(row.get('loss_final'))} | {fmt(row.get('eval_metric_final'))} | "
                f"`{row.get('dominant_label')}` | {row.get('diagnosis_reason')} |")
    lines.append("")
    return "\n".join(lines)


def write_dashboard_data(
        data_dir: Path,
        eval_rows: Sequence[Dict[str, Any]],
        loss_point_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    leaf_keys = sorted(
        {leaf_data_key(row) for row in eval_rows if row.get("leaf_id") not in (None, "")}
        | {leaf_data_key(row) for row in loss_point_rows if row.get("leaf_id") not in (None, "")})
    leaf_files = {}
    metric_names = set()
    eval_by_key: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    loss_by_key: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        if row.get("leaf_id") not in (None, ""):
            eval_by_key[leaf_data_key(row)].append(dict(row))
        for key, value in row.items():
            if key not in {"experiment", "step", "leaf_id", "eval_count", "eval_file"} and isinstance(value, (int, float)):
                metric_names.add(str(key))
    for row in loss_point_rows:
        if row.get("leaf_id") not in (None, ""):
            loss_by_key[leaf_data_key(row)].append(dict(row))
    for key in leaf_keys:
        filename = f"{safe_filename(key)}.json"
        payload = {
            "eval": sorted(eval_by_key.get(key, []), key=lambda row: int(row.get("step", 0))),
            "loss_points": sorted(loss_by_key.get(key, []), key=lambda row: int(row.get("step", 0))),
        }
        write_json(data_dir / filename, payload)
        leaf_files[key] = filename
    index = {
        "version": 1,
        "data_dir": "bucket_diagnosis_dashboard_data",
        "leaf_files": leaf_files,
        "metric_names": sorted(metric_names),
        "num_leaf_files": len(leaf_files),
    }
    write_json(data_dir / "index.json", index)
    return index


def leaf_data_key(row: Dict[str, Any]) -> str:
    return f"{row.get('experiment', '')}::{row.get('leaf_id', '')}"


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "leaf"


def render_dashboard(
        summary: Dict[str, Any],
        diag_rows: Sequence[Dict[str, Any]],
        loss_rows: Sequence[Dict[str, Any]],
        dashboard_data_index: Dict[str, Any],
        plotly_js: str) -> str:
    payload = json.dumps(
        {
            "summary": summary,
            "diagnosis": list(diag_rows),
            "loss": list(loss_rows),
            "dashboard_data": dashboard_data_index,
        },
        ensure_ascii=False,
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Bucket Diagnosis Dashboard</title>
  <script src="{html.escape(plotly_js, quote=True)}"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #1a2433; }}
    .page {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 16px; font-size: 24px; }}
    .controls {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    label {{ display: block; font-size: 12px; color: #637083; margin-bottom: 5px; }}
    select, input {{ width: 100%; box-sizing: border-box; border: 1px solid #cfd7e3; border-radius: 6px; padding: 8px 10px; background: white; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .card {{ background: white; border: 1px solid #dde3ea; border-radius: 8px; padding: 14px 16px; }}
    .label {{ color: #647184; font-size: 12px; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 16px; }}
    .panel {{ background: white; border: 1px solid #dde3ea; border-radius: 8px; padding: 14px; overflow: auto; }}
    .chart {{ height: 420px; }}
    .hint {{ color: #647184; font-size: 12px; margin: 4px 0 10px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ padding: 7px 8px; border-bottom: 1px solid #e6ebf0; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .reason {{ max-width: 520px; white-space: normal; text-align: left; }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Bucket Diagnosis Dashboard</h1>
    <div class="controls">
      <div><label>Experiment</label><select id="exp"></select></div>
      <div><label>Recommendation</label><select id="rec"></select></div>
      <div><label>Leaf ID</label><select id="leaf"></select></div>
      <div><label>Metric</label><select id="metric"></select></div>
      <div><label>Metric Min</label><input id="metric-min" type="number" step="0.01" placeholder="e.g. 0.0" /></div>
      <div><label>Metric Max</label><input id="metric-max" type="number" step="0.01" placeholder="e.g. 0.8" /></div>
      <div><label>Eval Count Min</label><input id="eval-count-min" type="number" step="1" placeholder="e.g. 20" /></div>
    </div>
    <div class="cards">
      <div class="card"><div class="label">Buckets</div><div class="value" id="bucket-count"></div></div>
      <div class="card"><div class="label">Eval Samples Final</div><div class="value" id="eval-count"></div></div>
      <div class="card"><div class="label">Mean Final Loss</div><div class="value" id="mean-loss"></div></div>
      <div class="card"><div class="label">Mean Eval Metric</div><div class="value" id="mean-eval"></div></div>
    </div>
    <div class="grid">
      <div class="panel"><h2>Loss vs Eval</h2><div id="scatter" class="chart"></div></div>
      <div class="panel"><h2>Leaf Loss and Eval by Step</h2><p id="leaf-trend-hint" class="hint"></p><div id="eval-curve" class="chart"></div></div>
      <div class="panel"><h2>Recommendation Counts</h2><div id="rec-bar" class="chart"></div></div>
      <div class="panel"><h2>Bucket Table</h2><div id="table"></div></div>
    </div>
  </div>
  <script>
    const DATA = {payload};
    const diag = DATA.diagnosis;
    const dashboardData = DATA.dashboard_data || {{}};
    const leafFiles = dashboardData.leaf_files || {{}};
    const leafCache = new Map();
    const experiments = [...new Set(diag.map(r => r.experiment))].sort();
    const recs = [...new Set(diag.map(r => r.recommendation))].sort();
    const metricNames = [...new Set([...(dashboardData.metric_names || []), 'eval_metric_final', 'eval_metric_best', 'eval_metric_delta'])].sort();
    const expSelect = document.getElementById('exp');
    const recSelect = document.getElementById('rec');
    const leafSelect = document.getElementById('leaf');
    const metricSelect = document.getElementById('metric');
    const metricMinInput = document.getElementById('metric-min');
    const metricMaxInput = document.getElementById('metric-max');
    const evalCountMinInput = document.getElementById('eval-count-min');
    expSelect.innerHTML = '<option value="">All</option>' + experiments.map(v => `<option>${{v}}</option>`).join('');
    recSelect.innerHTML = '<option value="">All</option>' + recs.map(v => `<option>${{v}}</option>`).join('');
    metricSelect.innerHTML = metricNames.map(v => `<option>${{v}}</option>`).join('');
    const preferred = metricNames.includes('decision_acc') ? 'decision_acc' : metricNames[0];
    if (preferred) metricSelect.value = preferred;
    updateLeafOptions();
    expSelect.addEventListener('change', () => {{ updateLeafOptions(); render(); }});
    recSelect.addEventListener('change', () => {{ updateLeafOptions(); render(); }});
    metricSelect.addEventListener('change', () => {{ updateLeafOptions(); render(); }});
    leafSelect.addEventListener('input', render);
    leafSelect.addEventListener('change', render);
    for (const el of [metricMinInput, metricMaxInput, evalCountMinInput]) {{
      el.addEventListener('input', () => {{ updateLeafOptions(); render(); }});
      el.addEventListener('change', () => {{ updateLeafOptions(); render(); }});
    }}
    function asNum(v) {{ return v === null || v === undefined || v === '' ? NaN : Number(v); }}
    function filters() {{
      return {{
        exp: expSelect.value,
        rec: recSelect.value,
        leaf: leafSelect.value,
        metric: metricSelect.value,
        metricMin: asNum(metricMinInput.value),
        metricMax: asNum(metricMaxInput.value),
        evalCountMin: asNum(evalCountMinInput.value)
      }};
    }}
    function passesMetricFilters(r, f) {{
      const metricValue = metricValueFromDiag(r, f.metric);
      const evalCount = asNum(r.eval_count_final);
      if (Number.isFinite(f.metricMin) && (!Number.isFinite(metricValue) || metricValue < f.metricMin)) return false;
      if (Number.isFinite(f.metricMax) && (!Number.isFinite(metricValue) || metricValue > f.metricMax)) return false;
      if (Number.isFinite(f.evalCountMin) && (!Number.isFinite(evalCount) || evalCount < f.evalCountMin)) return false;
      return true;
    }}
    function updateLeafOptions() {{
      const current = leafSelect.value;
      const f = filters();
      const items = diag
        .filter(r => (!f.exp || r.experiment === f.exp) && (!f.rec || r.recommendation === f.rec) && passesMetricFilters(r, f))
        .map(r => ({{ key: `${{r.experiment}}::${{r.leaf_id}}`, label: `${{r.experiment}} / leaf ${{r.leaf_id}}`, leaf: Number(r.leaf_id) }}));
      const unique = [...new Map(items.map(item => [item.key, item])).values()]
        .sort((a,b) => String(a.label).localeCompare(String(b.label), undefined, {{ numeric:true }}));
      leafSelect.innerHTML = '<option value="">All</option>' + unique.map(item => `<option value="${{item.key}}">${{item.label}}</option>`).join('');
      if (unique.some(item => item.key === current)) leafSelect.value = current;
    }}
    function filteredDiag() {{
      const f = filters();
      return diag.filter(r => (!f.exp || r.experiment === f.exp) && (!f.rec || r.recommendation === f.rec) && (!f.leaf || `${{r.experiment}}::${{r.leaf_id}}` === f.leaf) && passesMetricFilters(r, f));
    }}
    function render() {{
      const f = filters();
      const rows = filteredDiag();
      const leaves = new Set(rows.map(r => `${{r.experiment}}::${{r.leaf_id}}`));
      const finalEvalByLeaf = new Map();
      for (const r of rows) {{
        finalEvalByLeaf.set(`${{r.experiment}}::${{r.leaf_id}}`, {{
          step: r.eval_last_step,
          eval_count: r.eval_count_final,
          eval_metric_final: r.eval_metric_final,
          eval_metric_best: r.eval_metric_best,
          eval_metric_delta: r.eval_metric_delta,
          decision_acc: r.primary_eval_metric === 'decision_acc' ? r.eval_metric_final : undefined
        }});
      }}
      document.getElementById('bucket-count').textContent = rows.length;
      document.getElementById('eval-count').textContent = [...finalEvalByLeaf.values()].reduce((s, r) => s + Number(r.eval_count || 0), 0);
      const meanLoss = mean(rows.map(r => asNum(r.loss_final)).filter(Number.isFinite));
      const meanEval = mean(rows.map(r => metricValueFromDiag(r, f.metric)).filter(Number.isFinite));
      document.getElementById('mean-loss').textContent = fmt(meanLoss);
      document.getElementById('mean-eval').textContent = fmt(meanEval);
      renderScatter(rows, finalEvalByLeaf, f.metric);
      renderLeafTrend(rows, f.metric);
      renderRecBar(rows);
      renderTable(rows, finalEvalByLeaf, f.metric);
    }}
    function mean(xs) {{ return xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : NaN; }}
    function fmt(v) {{ return Number.isFinite(v) ? v.toFixed(4) : '-'; }}
    function metricValueFromDiag(r, metric) {{
      if (metric === 'eval_metric_final') return asNum(r.eval_metric_final);
      if (metric === 'eval_metric_best') return asNum(r.eval_metric_best);
      if (metric === 'eval_metric_delta') return asNum(r.eval_metric_delta);
      if (r[`eval_final_${{metric}}`] !== undefined) return asNum(r[`eval_final_${{metric}}`]);
      if (metric === r.primary_eval_metric) return asNum(r.eval_metric_final);
      return asNum(r[metric]);
    }}
    function renderScatter(rows, finalEvalByLeaf, metric) {{
      const x = [], y = [], text = [], color = [];
      for (const r of rows) {{
        const key = `${{r.experiment}}::${{r.leaf_id}}`;
        const xv = asNum(r.loss_final), yv = metricValueFromDiag(r, metric);
        if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
        x.push(xv); y.push(yv); color.push(asNum(r.purity));
        text.push(`${{r.experiment}} leaf=${{r.leaf_id}}<br>${{r.recommendation}}<br>${{r.dominant_label}}<br>${{r.diagnosis_reason}}`);
      }}
      Plotly.newPlot('scatter', [{{ type:'scatter', mode:'markers', x, y, text, marker:{{ size:9, color, colorscale:'Viridis', showscale:true }} }}], {{
        margin: {{ l: 56, r: 24, t: 12, b: 48 }},
        xaxis: {{ title: 'final loss' }},
        yaxis: {{ title: metric }}
      }}, {{ responsive:true }});
    }}
    async function renderLeafTrend(diagRows, metric) {{
      const f = filters();
      const selectedKey = f.leaf || (diagRows.length === 1 ? `${{diagRows[0].experiment}}::${{diagRows[0].leaf_id}}` : '');
      if (!selectedKey) {{
        document.getElementById('leaf-trend-hint').textContent = '';
        Plotly.newPlot('eval-curve', [], {{
          margin: {{ l: 56, r: 24, t: 24, b: 48 }},
          annotations: [{{ text:'Select one leaf to compare eval metric and train loss.', x:0.5, y:0.5, xref:'paper', yref:'paper', showarrow:false }}]
        }}, {{ responsive:true }});
        return;
      }}
      const allowed = new Set(diagRows.filter(r => `${{r.experiment}}::${{r.leaf_id}}` === selectedKey).map(r => `${{r.experiment}}::${{r.leaf_id}}`));
      let leafPayloads;
      try {{
        leafPayloads = await Promise.all([...allowed].map(loadLeafData));
      }} catch (err) {{
        document.getElementById('leaf-trend-hint').textContent = 'Leaf detail JSON could not be loaded. Serve this directory over HTTP, for example: python3 -m http.server';
        Plotly.newPlot('eval-curve', [], {{
          margin: {{ l: 56, r: 24, t: 24, b: 48 }},
          annotations: [{{ text:String(err), x:0.5, y:0.5, xref:'paper', yref:'paper', showarrow:false }}]
        }}, {{ responsive:true }});
        return;
      }}
      document.getElementById('leaf-trend-hint').innerHTML = leafPayloads.length
        ? `${{leafSummaryHtml(selectedKey, diagRows)}}<br>Loaded detail JSON from ${{esc(detailPathForKey(selectedKey))}}`
        : 'No detail data for this leaf.';
      const evalRowsForLeaf = leafPayloads.flatMap(item => item.eval || []);
      const lossRowsForLeaf = leafPayloads.flatMap(item => item.loss_points || []);
      const trendMetric = metric.startsWith('eval_metric_')
        ? ((diagRows.find(r => `${{r.experiment}}::${{r.leaf_id}}` === selectedKey) || {{}}).primary_eval_metric || 'decision_acc')
        : metric;
      const groups = new Map();
      for (const r of evalRowsForLeaf) {{
        const key = `${{r.experiment}} leaf ${{r.leaf_id}}`;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(r);
      }}
      const traces = [...groups.entries()].slice(0, 80).map(([key, vals]) => {{
        vals.sort((a,b)=>Number(a.step)-Number(b.step));
        return {{ type:'scatter', mode:'lines+markers', name:`eval ${{key}}`, x:vals.map(r=>r.step), y:vals.map(r=>r[trendMetric]), yaxis:'y' }};
      }});
      const lossGroups = new Map();
      for (const r of lossRowsForLeaf) {{
        const key = `${{r.experiment}} leaf ${{r.leaf_id}}`;
        if (!lossGroups.has(key)) lossGroups.set(key, []);
        lossGroups.get(key).push(r);
      }}
      for (const [key, vals] of lossGroups.entries()) {{
        vals.sort((a,b)=>Number(a.step)-Number(b.step));
        traces.push({{ type:'scatter', mode:'lines', name:`loss ${{key}}`, x:vals.map(r=>r.step), y:vals.map(r=>r.loss), yaxis:'y2', line:{{ dash:'dot' }} }});
      }}
      Plotly.newPlot('eval-curve', traces, {{
        margin: {{ l: 56, r: 24, t: 12, b: 48 }},
        xaxis: {{ title:'step' }},
        yaxis: {{ title: trendMetric }},
        yaxis2: {{ title:'train loss', overlaying:'y', side:'right', showgrid:false }},
        showlegend: traces.length <= 20
      }}, {{ responsive:true }});
    }}
    async function loadLeafData(key) {{
      if (leafCache.has(key)) return leafCache.get(key);
      const filename = leafFiles[key];
      if (!filename) return {{ eval: [], loss_points: [] }};
      const path = detailPathForKey(key);
      const response = await fetch(path);
      if (!response.ok) throw new Error(`failed to load ${{path}}: ${{response.status}}`);
      const data = await response.json();
      leafCache.set(key, data);
      return data;
    }}
    function detailPathForKey(key) {{
      const filename = leafFiles[key] || '';
      return `${{dashboardData.data_dir || 'bucket_diagnosis_dashboard_data'}}/${{filename}}`;
    }}
    function leafSummaryHtml(key, rows) {{
      const row = rows.find(r => `${{r.experiment}}::${{r.leaf_id}}` === key) || {{}};
      const trainCount = row.sampled_target ?? row.label_target_total ?? row.raw_count ?? '';
      const parts = [
        `experiment=${{esc(row.experiment || '')}}`,
        `leaf=${{esc(row.leaf_id ?? '')}}`,
        `eval_n=${{esc(row.eval_count_final ?? '')}}`,
        `raw=${{esc(row.raw_count ?? '')}}`,
        `train=${{esc(trainCount)}}`,
        `sample_ratio=${{fmt(asNum(row.sample_ratio))}}`,
        `purity=${{fmt(asNum(row.purity))}}`,
        `label=${{esc(row.dominant_label || '')}}`
      ];
      return parts.join(' · ');
    }}
    function renderRecBar(rows) {{
      const counts = {{}};
      for (const r of rows) counts[r.recommendation] = (counts[r.recommendation] || 0) + 1;
      const keys = Object.keys(counts).sort();
      Plotly.newPlot('rec-bar', [{{ type:'bar', x:keys, y:keys.map(k=>counts[k]), marker:{{ color:'#4C78A8' }} }}], {{
        margin: {{ l: 56, r: 24, t: 12, b: 100 }},
        xaxis: {{ tickangle:-30 }},
        yaxis: {{ title:'buckets' }}
      }}, {{ responsive:true }});
    }}
    function renderTable(rows, finalEvalByLeaf, metric) {{
      const sorted = [...rows].sort((a,b)=>String(a.recommendation).localeCompare(String(b.recommendation)) || asNum(b.loss_final)-asNum(a.loss_final)).slice(0, 300);
      let html = '<table><thead><tr><th>experiment</th><th>leaf</th><th>rec</th><th>purity</th><th>loss</th><th>'+metric+'</th><th>eval_n</th><th>raw</th><th>train</th><th>ratio</th><th>label</th><th class="reason">reason</th></tr></thead><tbody>';
      for (const r of sorted) {{
        const e = finalEvalByLeaf.get(`${{r.experiment}}::${{r.leaf_id}}`) || {{}};
        const trainCount = r.sampled_target ?? r.label_target_total ?? r.raw_count ?? '';
        html += `<tr><td>${{esc(r.experiment)}}</td><td>${{r.leaf_id}}</td><td>${{esc(r.recommendation)}}</td><td>${{fmt(asNum(r.purity))}}</td><td>${{fmt(asNum(r.loss_final))}}</td><td>${{fmt(metricValueFromDiag(r, metric))}}</td><td>${{e.eval_count || ''}}</td><td>${{esc(r.raw_count ?? '')}}</td><td>${{esc(trainCount)}}</td><td>${{fmt(asNum(r.sample_ratio))}}</td><td>${{esc(r.dominant_label || '')}}</td><td class="reason">${{esc(r.diagnosis_reason || '')}}</td></tr>`;
      }}
      html += '</tbody></table>';
      document.getElementById('table').innerHTML = html;
    }}
    function esc(v) {{ return String(v ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch])); }}
    render();
  </script>
</body>
</html>
"""


def fmt(value: Any) -> str:
    value = to_float(value)
    return "" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
