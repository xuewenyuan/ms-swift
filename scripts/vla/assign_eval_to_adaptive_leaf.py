#!/usr/bin/env python3
"""Assign eval feature rows to the nearest adaptive leaf from train features."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from analyze_cluster_mining import safe_div, write_csv
from cluster_features_faiss import (
    configure_cuda_visible_devices,
    decision_pair,
    import_faiss,
    load_dump,
    l2_normalize_matrix,
    parse_gpu_devices,
)


# =========================
# Config area
# =========================
TRAIN_FEATURE_DIR = ""
EVAL_FEATURE_DIR = ""
ADAPTIVE_DIR = ""
OUTPUT = ""
MATRIX = "input_text_hash"
METRIC = "cosine"  # choices: cosine, l2
BATCH_SIZE = 8192
USE_GPU_INDEX = False
GPU_DEVICES = ""  # e.g. "0" or "0,1"; empty means default CPU index
# =========================


UNKNOWN = "UNKNOWN"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assign eval feature rows to train adaptive leaves by nearest-neighbor search. "
            "Defaults come from the config area at the top of this file."))
    parser.add_argument("--train-feature-dir", default=TRAIN_FEATURE_DIR, help="Train feature dump dir.")
    parser.add_argument("--eval-feature-dir", default=EVAL_FEATURE_DIR, help="Eval feature dump dir.")
    parser.add_argument("--adaptive-dir", default=ADAPTIVE_DIR, help="Adaptive clustering report dir.")
    parser.add_argument("--output", default=OUTPUT, help="Output eval assignment JSONL path.")
    parser.add_argument("--matrix", default=MATRIX, help="Matrix name/path used for both train and eval dumps.")
    parser.add_argument("--metric", choices=["cosine", "l2"], default=METRIC, help="Nearest-neighbor metric.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Eval search batch size.")
    parser.add_argument("--use-gpu-index", action="store_true", default=USE_GPU_INDEX, help="Move FAISS index to GPU.")
    parser.add_argument("--gpu-devices", default=GPU_DEVICES, help="Comma-separated GPU ids for FAISS index.")
    parser.add_argument(
        "--assignments",
        default=None,
        help="Optional adaptive_leaf_assignments.jsonl path. Defaults to ADAPTIVE_DIR/adaptive_leaf_assignments.jsonl.")
    parser.add_argument(
        "--leaf-summary",
        default=None,
        help="Optional adaptive_leaf_summary.csv path. Defaults to ADAPTIVE_DIR/adaptive_leaf_summary.csv.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    train_feature_dir = Path(args.train_feature_dir).expanduser()
    eval_feature_dir = Path(args.eval_feature_dir).expanduser()
    adaptive_dir = Path(args.adaptive_dir).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assignments_path = Path(args.assignments).expanduser() if args.assignments else adaptive_dir / "adaptive_leaf_assignments.jsonl"
    leaf_summary_path = Path(args.leaf_summary).expanduser() if args.leaf_summary else adaptive_dir / "adaptive_leaf_summary.csv"

    train_dump = load_dump(train_feature_dir, args.matrix)
    eval_dump = load_dump(eval_feature_dir, args.matrix)
    train_x = np.ascontiguousarray(train_dump["matrix"].astype(np.float32, copy=False))
    eval_x = np.ascontiguousarray(eval_dump["matrix"].astype(np.float32, copy=False))
    if train_x.shape[1] != eval_x.shape[1]:
        raise ValueError(f"matrix dim mismatch: train={train_x.shape}, eval={eval_x.shape}")
    if args.metric == "cosine":
        train_x = l2_normalize_matrix(train_x)
        eval_x = l2_normalize_matrix(eval_x)

    train_assignments = load_train_assignments(assignments_path, train_x.shape[0])
    leaf_summary = load_leaf_summary(leaf_summary_path)
    gpu_devices = parse_gpu_devices(args.gpu_devices)
    configure_cuda_visible_devices(gpu_devices if args.use_gpu_index else None)
    faiss = import_faiss()
    index = build_index(faiss, train_x, args.metric, args.use_gpu_index, gpu_devices)

    eval_meta_rows = eval_dump["meta_rows"]
    if len(eval_meta_rows) != eval_x.shape[0]:
        raise ValueError(f"eval metadata rows ({len(eval_meta_rows)}) != matrix rows ({eval_x.shape[0]})")

    aggregate: DefaultDict[int, Counter] = defaultdict(Counter)
    score_sums: Counter = Counter()
    total = 0
    with output_path.open("w", encoding="utf-8") as f:
        for start in range(0, eval_x.shape[0], args.batch_size):
            end = min(start + args.batch_size, eval_x.shape[0])
            scores, ids = index.search(np.ascontiguousarray(eval_x[start:end]), 1)
            for offset, (score_row, id_row) in enumerate(zip(scores, ids)):
                eval_row = start + offset
                train_row = int(id_row[0])
                raw_score = float(score_row[0])
                train_assignment = train_assignments[train_row]
                eval_meta = eval_meta_rows[eval_row]
                eval_lateral, eval_longitudinal = decision_pair(eval_meta)
                eval_label = f"{eval_lateral}|{eval_longitudinal}"
                leaf_id = int(train_assignment["leaf_id"])
                closeness = raw_score if args.metric == "cosine" else -raw_score
                output = build_output_row(
                    eval_row,
                    eval_meta,
                    eval_label,
                    train_row,
                    train_assignment,
                    leaf_summary.get(leaf_id, {}),
                    raw_score,
                    closeness,
                    args.metric,
                )
                f.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
                aggregate[leaf_id]["eval_count"] += 1
                aggregate[leaf_id][f"label::{eval_label}"] += 1
                aggregate[leaf_id][f"dominant_match::{output['is_eval_label_dominant']}"] += 1
                score_sums[leaf_id] += closeness
                total += 1

    summary_rows = build_eval_leaf_summary(aggregate, score_sums, leaf_summary)
    write_csv(output_path.with_suffix(".leaf_summary.csv"), summary_rows)
    write_json(
        output_path.with_suffix(".summary.json"),
        {
            "train_feature_dir": str(train_feature_dir),
            "eval_feature_dir": str(eval_feature_dir),
            "adaptive_dir": str(adaptive_dir),
            "assignments": str(assignments_path),
            "matrix": args.matrix,
            "metric": args.metric,
            "train_rows": int(train_x.shape[0]),
            "eval_rows": int(eval_x.shape[0]),
            "assigned_rows": total,
            "num_eval_leaves": len(aggregate),
            "outputs": {
                "assignments": str(output_path),
                "leaf_summary_csv": str(output_path.with_suffix(".leaf_summary.csv")),
            },
        },
    )
    print(f"assigned {total} eval rows to {len(aggregate)} train adaptive leaves -> {output_path}")


def validate_args(args: argparse.Namespace) -> None:
    required = {
        "--train-feature-dir": args.train_feature_dir,
        "--eval-feature-dir": args.eval_feature_dir,
        "--adaptive-dir": args.adaptive_dir,
        "--output": args.output,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"missing required config/args: {', '.join(missing)}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")


def build_index(
        faiss: Any,
        train_x: np.ndarray,
        metric: str,
        use_gpu_index: bool,
        gpu_devices: Optional[Sequence[int]]) -> Any:
    if metric == "cosine":
        index = faiss.IndexFlatIP(train_x.shape[1])
    else:
        index = faiss.IndexFlatL2(train_x.shape[1])
    if use_gpu_index:
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError("The installed faiss package does not expose GPU resources. Please install faiss-gpu.")
        device = 0
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, device, index)
    index.add(train_x)
    return index


def load_train_assignments(path: Path, expected_rows: int) -> List[Dict[str, Any]]:
    rows: List[Optional[Dict[str, Any]]] = [None] * expected_rows
    for item in iter_jsonl(path):
        row = int(item["row"])
        if row < 0 or row >= expected_rows:
            raise ValueError(f"assignment row out of range in {path}: {row}, expected < {expected_rows}")
        rows[row] = item
    missing = [idx for idx, item in enumerate(rows) if item is None]
    if missing:
        preview = ",".join(str(idx) for idx in missing[:10])
        raise ValueError(f"{path} missing {len(missing)} train rows; first missing rows: {preview}")
    return [item for item in rows if item is not None]


def load_leaf_summary(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    import csv

    rows: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("leaf_id") in (None, ""):
                continue
            rows[int(row["leaf_id"])] = row
    return rows


def build_output_row(
        eval_row: int,
        eval_meta: Dict[str, Any],
        eval_label: str,
        train_row: int,
        train_assignment: Dict[str, Any],
        leaf_summary: Dict[str, Any],
        nearest_raw_score: float,
        nearest_closeness: float,
        metric: str) -> Dict[str, Any]:
    eval_lateral, eval_longitudinal = split_label(eval_label)
    dominant_label = str(train_assignment.get("dominant_label") or leaf_summary.get("dominant_label") or UNKNOWN)
    train_label = str(train_assignment.get("label") or UNKNOWN)
    train_lateral, train_longitudinal = split_label(train_label)
    return {
        "eval_row": eval_row,
        "eval_row_idx": eval_meta.get("row_idx"),
        "eval_sample_id": eval_meta.get("sample_id"),
        "eval_scene_id": eval_meta.get("scene_id"),
        "eval_sample_token": eval_meta.get("sample_token"),
        "eval_timestamp": eval_meta.get("timestamp"),
        "eval_lateral": eval_lateral,
        "eval_longitudinal": eval_longitudinal,
        "eval_label": eval_label,
        "nearest_train_row": train_row,
        "nearest_train_row_idx": train_assignment.get("row_idx"),
        "nearest_train_sample_id": train_assignment.get("sample_id"),
        "nearest_train_scene_id": train_assignment.get("scene_id"),
        "nearest_train_lateral": train_lateral,
        "nearest_train_longitudinal": train_longitudinal,
        "nearest_train_label": train_label,
        "nearest_raw_score": nearest_raw_score,
        "nearest_closeness": nearest_closeness,
        "metric": metric,
        "leaf_id": int(train_assignment["leaf_id"]),
        "node_id": train_assignment.get("node_id"),
        "depth": train_assignment.get("depth"),
        "path": train_assignment.get("path"),
        "dominant_label": dominant_label,
        "dominant_lateral": split_label(dominant_label)[0],
        "dominant_longitudinal": split_label(dominant_label)[1],
        "leaf_purity": to_float(train_assignment.get("leaf_purity") or leaf_summary.get("purity")),
        "leaf_size": to_int(leaf_summary.get("size")),
        "leaf_stop_reason": leaf_summary.get("stop_reason"),
        "leaf_suggested_action": leaf_summary.get("suggested_action"),
        "is_eval_label_dominant": eval_label == dominant_label,
        "is_eval_label_same_as_nearest_train": eval_label == train_label,
    }


def build_eval_leaf_summary(
        aggregate: Dict[int, Counter],
        score_sums: Counter,
        leaf_summary: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for leaf_id, counter in sorted(aggregate.items()):
        eval_count = int(counter["eval_count"])
        label_counter = Counter()
        dominant_match_count = int(counter.get("dominant_match::True", 0))
        for key, count in counter.items():
            if key.startswith("label::"):
                label_counter[key[len("label::"):]] = count
        eval_dominant_label, eval_dominant_count = (label_counter.most_common(1)[0] if label_counter else (UNKNOWN, 0))
        train_leaf = leaf_summary.get(leaf_id, {})
        rows.append({
            "leaf_id": leaf_id,
            "eval_count": eval_count,
            "eval_dominant_label": eval_dominant_label,
            "eval_dominant_count": int(eval_dominant_count),
            "eval_dominant_ratio": safe_div(eval_dominant_count, eval_count),
            "eval_label_entropy": entropy(label_counter),
            "eval_label_top": format_counter(label_counter, 8),
            "eval_label_matches_train_dominant_ratio": safe_div(dominant_match_count, eval_count),
            "avg_nearest_closeness": safe_div(score_sums[leaf_id], eval_count),
            "train_leaf_size": train_leaf.get("size", ""),
            "train_leaf_purity": train_leaf.get("purity", ""),
            "train_dominant_label": train_leaf.get("dominant_label", ""),
            "train_stop_reason": train_leaf.get("stop_reason", ""),
            "train_suggested_action": train_leaf.get("suggested_action", ""),
            "train_primary_scene_top": train_leaf.get("primary_scene_top", ""),
            "train_secondary_scene_top": train_leaf.get("secondary_scene_top", ""),
        })
    return rows


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


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_label(label: str) -> Tuple[str, str]:
    parts = str(label).split("|", 1)
    if len(parts) == 1:
        return parts[0], UNKNOWN
    return parts[0], parts[1]


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            value -= float(p * np.log(p))
    return value


def format_counter(counter: Counter, topn: int) -> str:
    total = sum(counter.values())
    return ";".join(f"{key}:{count}:{safe_div(count, total):.4f}" for key, count in counter.most_common(topn))


def to_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
