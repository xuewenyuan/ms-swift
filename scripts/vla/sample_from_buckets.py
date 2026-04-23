#!/usr/bin/env python3
"""Sample raw JSONL data according to analyze_cluster_mining sampling buckets."""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from analyze_cluster_mining import (
    UNKNOWN,
    decision_label,
    load_feature_meta,
    load_raw_meta_lookup,
    read_json,
    resolve_assignments_path,
    resolve_input_files,
    safe_div,
    split_label,
    write_csv,
)


Bucket = Tuple[str, str, int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample raw JSONL rows using sampling_buckets.csv target counts.")
    parser.add_argument("--raw-input", required=True, help="Raw JSONL file or directory to sample from.")
    parser.add_argument("--feature-dir", required=True, help="Feature dump dir containing meta.jsonl.")
    parser.add_argument(
        "--cluster-dir",
        required=True,
        help="Cluster report dir containing cluster_assignments.jsonl, or the assignments JSONL path.")
    parser.add_argument("--scene-map", required=True, help="scene_id -> scene labels JSON used by mining.")
    parser.add_argument("--sampling-buckets", required=True, help="sampling_buckets.csv from analyze_cluster_mining.py.")
    parser.add_argument("--output", required=True, help="Output sampled JSONL path.")
    parser.add_argument("--report", default=None, help="Optional sampling report CSV path.")
    parser.add_argument("--sample-id-key", default="sample_id", help="Sample id key in raw JSONL.")
    parser.add_argument("--scene-id-key", default="scene_id", help="Scene id key in raw JSONL.")
    parser.add_argument("--sample-token-key", default="sample_token", help="Sample token key in raw JSONL.")
    parser.add_argument("--timestamp-key", default="timestamp", help="Timestamp key in raw JSONL.")
    parser.add_argument("--primary-scene-key", default="专题", help="Primary scene label key in --scene-map.")
    parser.add_argument("--secondary-scene-key", default="二级场景", help="Secondary scene label key in --scene-map.")
    parser.add_argument(
        "--target-column",
        default="suggested_target_count",
        help="Column in sampling_buckets.csv used as per-bucket target. Default: suggested_target_count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    parser.add_argument(
        "--sampling-mode",
        choices=["without_replacement", "with_replacement"],
        default="without_replacement",
        help="How to handle target > available. Default caps at available rows.")
    parser.add_argument(
        "--write-empty-report-rows",
        action="store_true",
        help="Include bucket report rows for targets that had no matching raw rows.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report) if args.report else output_path.with_suffix(output_path.suffix + ".report.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    scene_map = read_json(Path(args.scene_map))
    raw_lookup = load_raw_meta_lookup(
        args.raw_input,
        args.sample_id_key,
        args.scene_id_key,
        args.sample_token_key,
        args.timestamp_key,
    )
    meta_rows = load_feature_meta(
        Path(args.feature_dir),
        scene_map,
        raw_lookup,
        primary_key=args.primary_scene_key,
        secondary_key=args.secondary_scene_key,
    )
    assignments = load_assignments(resolve_assignments_path(Path(args.cluster_dir)))
    targets = load_sampling_targets(Path(args.sampling_buckets), args.target_column)
    row_to_bucket = build_row_to_bucket(meta_rows, assignments)
    bucket_to_rows = collect_bucket_rows(row_to_bucket)
    selected_rows, report_rows = select_rows(targets, bucket_to_rows, args.sampling_mode, rng,
                                             args.write_empty_report_rows)
    write_sampled_jsonl(output_path, args.raw_input, selected_rows, rng)
    write_csv(report_path, report_rows)
    print(
        f"sampled {len(selected_rows)} rows from {sum(len(v) for v in bucket_to_rows.values())} assigned rows "
        f"across {len(report_rows)} buckets -> {output_path}")
    print(f"sampling report -> {report_path}")


def load_assignments(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path):
        rows.append(row)
    rows.sort(key=lambda item: int(item.get("row", len(rows))))
    return rows


def load_sampling_targets(path: Path, target_column: str) -> Dict[Bucket, int]:
    targets: Dict[Bucket, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"primary_scene", "label", "cluster_id", target_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns in {path}: {sorted(missing)}")
        for row in reader:
            bucket = (row["primary_scene"], row["label"], int(float(row["cluster_id"])))
            target = max(0, int(round(float(row[target_column]))))
            targets[bucket] = target
    return targets


def build_row_to_bucket(meta_rows: Sequence[Any], assignments: Sequence[Dict[str, Any]]) -> Dict[int, Bucket]:
    row_to_bucket = {}
    for line_idx, assignment in enumerate(assignments):
        row = int(assignment.get("row", line_idx))
        if row >= len(meta_rows):
            raise ValueError(f"assignment row {row} out of meta range {len(meta_rows)}")
        meta = meta_rows[row]
        label = decision_label(str(assignment.get("lateral") or UNKNOWN), str(assignment.get("longitudinal") or UNKNOWN))
        bucket = (meta.primary_scene, label, int(assignment["cluster_id"]))
        row_to_bucket[row] = bucket
    return row_to_bucket


def collect_bucket_rows(row_to_bucket: Dict[int, Bucket]) -> Dict[Bucket, List[int]]:
    bucket_to_rows: DefaultDict[Bucket, List[int]] = defaultdict(list)
    for row, bucket in row_to_bucket.items():
        bucket_to_rows[bucket].append(row)
    return dict(bucket_to_rows)


def select_rows(
        targets: Dict[Bucket, int],
        bucket_to_rows: Dict[Bucket, List[int]],
        sampling_mode: str,
        rng: random.Random,
        write_empty_report_rows: bool) -> Tuple[List[int], List[Dict[str, Any]]]:
    selected = []
    report_rows = []
    for bucket, target in sorted(targets.items()):
        available_rows = bucket_to_rows.get(bucket, [])
        available = len(available_rows)
        if available == 0:
            if write_empty_report_rows or target > 0:
                report_rows.append(report_row(bucket, target, available, 0, "missing_bucket"))
            continue
        if sampling_mode == "with_replacement" and target > available:
            chosen = [rng.choice(available_rows) for _ in range(target)]
            status = "sampled_with_replacement"
        else:
            count = min(target, available)
            chosen = rng.sample(available_rows, count) if count < available else list(available_rows)
            status = "capped_at_available" if target > available else "sampled"
        selected.extend(chosen)
        report_rows.append(report_row(bucket, target, available, len(chosen), status))
    rng.shuffle(selected)
    return selected, report_rows


def report_row(bucket: Bucket, target: int, available: int, sampled: int, status: str) -> Dict[str, Any]:
    primary_scene, label, cluster_id = bucket
    lateral, longitudinal = split_label(label)
    return {
        "primary_scene": primary_scene,
        "lateral": lateral,
        "longitudinal": longitudinal,
        "label": label,
        "cluster_id": cluster_id,
        "target": target,
        "available": available,
        "sampled": sampled,
        "sample_ratio": safe_div(sampled, available),
        "status": status,
    }


def write_sampled_jsonl(output_path: Path, raw_input: str, selected_rows: Sequence[int], rng: random.Random) -> None:
    selected_counter = Counter(selected_rows)
    selected_set = set(selected_counter)
    written_counter: Counter = Counter()
    with output_path.open("w", encoding="utf-8") as out:
        for row_idx, row in enumerate(iter_jsonl_files(resolve_input_files(Path(raw_input)))):
            if row_idx not in selected_set:
                continue
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for _ in range(selected_counter[row_idx]):
                out.write(line)
                written_counter[row_idx] += 1
    missing = selected_counter - written_counter
    if missing:
        examples = list(missing.items())[:5]
        raise ValueError(f"failed to write {sum(missing.values())} selected rows; examples: {examples}")


def iter_jsonl_files(paths: Sequence[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        yield from iter_jsonl(path)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {e}") from e


if __name__ == "__main__":
    main()
