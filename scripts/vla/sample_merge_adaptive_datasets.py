#!/usr/bin/env python3
"""Offline sample adaptive leaf datasets from dataset_args.txt and merge into shard JSONL files."""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from analyze_cluster_mining import resolve_input_files, safe_div, write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline sample and merge adaptive leaf JSONL datasets using dataset_args.txt.")
    parser.add_argument("--dataset-args", required=True, help="dataset_args.txt from build_adaptive_dataset_info.py.")
    parser.add_argument(
        "--dataset-info",
        required=True,
        help="adaptive_leaf_dataset_info.json from build_adaptive_dataset_info.py.")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged shard JSONL files.")
    parser.add_argument("--report", default=None, help="Optional sampling report CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    parser.add_argument(
        "--sampling-mode",
        choices=["capped", "repeat", "with_replacement"],
        default="capped",
        help="How to handle target > available. Default: capped.")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of output JSONL shards. Default: 1.")
    parser.add_argument(
        "--output-prefix",
        default="merged",
        help="Output shard filename prefix. Default: merged.")
    parser.add_argument(
        "--shuffle-datasets",
        action="store_true",
        help="Shuffle dataset processing order before merging. Sample order within each dataset is preserved.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report) if args.report else output_dir / "sampling_report.csv"

    dataset_map = load_dataset_info(Path(args.dataset_info))
    requests = load_dataset_args(Path(args.dataset_args))
    order = list(requests)
    if args.shuffle_datasets:
        rng.shuffle(order)

    shard_paths = [output_dir / f"{args.output_prefix}_{idx:05d}.jsonl" for idx in range(max(1, args.num_shards))]
    report_rows = []
    total_written = 0
    next_shard = 0
    with ExitStackFiles(shard_paths) as shard_files:
        for request in order:
            dataset_path = dataset_map.get(request["dataset_name"])
            if dataset_path is None:
                raise ValueError(f"dataset_name not found in dataset_info: {request['dataset_name']}")
            selected_lines, report_row = sample_dataset_lines(Path(dataset_path), request["target"], args.sampling_mode, rng)
            report_row["dataset_name"] = request["dataset_name"]
            report_row["dataset_path"] = str(dataset_path)
            report_rows.append(report_row)
            for line in selected_lines:
                shard_files[next_shard].write(line)
                total_written += 1
                next_shard = (next_shard + 1) % len(shard_files)

    write_csv(report_path, report_rows)
    print(f"sampled and merged {total_written} rows from {len(order)} datasets -> {output_dir}")
    print(f"sampling report -> {report_path}")


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
        for f in self.files:
            f.close()


def load_dataset_info(path: Path) -> Dict[str, Path]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    dataset_map = {}
    for row in rows:
        dataset_name = str(row["dataset_name"])
        dataset_path = Path(row["dataset_path"]).expanduser()
        dataset_map[dataset_name] = dataset_path
    return dataset_map


def load_dataset_args(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            dataset_name, target = parse_dataset_arg(line)
            rows.append({"dataset_name": dataset_name, "target": target})
    if not rows:
        raise ValueError(f"no dataset args found in {path}")
    return rows


def parse_dataset_arg(value: str) -> Tuple[str, Optional[int]]:
    if "#" not in value:
        return value, None
    dataset_name, sample = value.rsplit("#", 1)
    return dataset_name, int(sample)


def sample_dataset_lines(
        dataset_path: Path,
        target: Optional[int],
        sampling_mode: str,
        rng: random.Random) -> Tuple[List[str], Dict[str, Any]]:
    all_lines = list(iter_jsonl_lines(resolve_input_files(dataset_path)))
    available = len(all_lines)
    if target is None:
        chosen = list(all_lines)
        status = "full"
        target_count = available
    elif sampling_mode == "with_replacement" and target > available:
        chosen = [rng.choice(all_lines) for _ in range(target)] if available else []
        status = "with_replacement" if available else "missing_dataset"
        target_count = target
    elif sampling_mode == "repeat" and target > available:
        chosen = []
        full_repeats = target // max(1, available)
        remainder = target % max(1, available)
        for _ in range(full_repeats):
            chosen.extend(all_lines)
        if remainder > 0 and available > 0:
            chosen.extend(rng.sample(all_lines, remainder) if remainder < available else list(all_lines))
        status = "repeat" if available else "missing_dataset"
        target_count = target
    else:
        count = min(target, available)
        chosen = rng.sample(all_lines, count) if target is not None and count < available else list(all_lines[:count])
        status = "capped" if target is not None and target > available else "sampled"
        target_count = 0 if target is None else target
    return chosen, {
        "requested": "" if target is None else target_count,
        "available": available,
        "sampled": len(chosen),
        "sample_ratio": safe_div(len(chosen), available),
        "status": status,
    }


def iter_jsonl_lines(paths: Sequence[Path]) -> Iterable[str]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                if not line.endswith("\n"):
                    line += "\n"
                yield line


if __name__ == "__main__":
    main()
