#!/usr/bin/env python3
"""Split raw JSONL rows by adaptive clustering leaves or derived groups."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, TextIO, Tuple

from analyze_cluster_mining import UNKNOWN, resolve_input_files, safe_div, split_label, write_csv


GroupKey = str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split original JSONL rows using adaptive_leaf_assignments.jsonl.")
    parser.add_argument("--raw-input", required=True, help="Original JSONL file or directory used for feature dump.")
    parser.add_argument(
        "--adaptive-dir",
        required=True,
        help="Adaptive report dir containing adaptive_leaf_assignments.jsonl and adaptive_leaf_summary.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for split JSONL files and manifest CSV.")
    parser.add_argument(
        "--group-by",
        choices=[
            "leaf",
            "primary_scene",
            "secondary_scene",
            "label",
            "dominant_label",
            "suggested_action",
            "primary_scene_leaf",
            "secondary_scene_leaf",
            "primary_scene_label",
            "primary_scene_dominant_label",
            "label_leaf",
        ],
        default="leaf",
        help="Grouping key used for output JSONL files. Default: leaf.")
    parser.add_argument(
        "--assignments",
        default=None,
        help="Optional path to adaptive_leaf_assignments.jsonl. Defaults to --adaptive-dir/adaptive_leaf_assignments.jsonl.")
    parser.add_argument(
        "--leaf-summary",
        default=None,
        help="Optional path to adaptive_leaf_summary.csv. Defaults to --adaptive-dir/adaptive_leaf_summary.csv.")
    parser.add_argument("--manifest", default="split_manifest.csv", help="Manifest CSV name/path. Default: split_manifest.csv.")
    parser.add_argument("--filename-prefix", default="adaptive", help="Output JSONL filename prefix. Default: adaptive.")
    parser.add_argument("--max-open-files", type=int, default=64, help="Max concurrently open output files. Default: 64.")
    parser.add_argument(
        "--add-mining-meta",
        action="store_true",
        help="Add an _adaptive_mining field to each output JSON object. By default raw rows are kept unchanged.")
    parser.add_argument(
        "--add-channel",
        action="store_true",
        help="Add a channel field to each output row for ms-swift --enable_channel_loss monitoring.")
    parser.add_argument("--channel-key", default="channel", help="Channel field name to write. Default: channel.")
    parser.add_argument("--channel-prefix", default="vla_adaptive", help="Channel name prefix. Default: vla_adaptive.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    adaptive_dir = Path(args.adaptive_dir)
    assignments_path = Path(args.assignments) if args.assignments else adaptive_dir / "adaptive_leaf_assignments.jsonl"
    summary_path = Path(args.leaf_summary) if args.leaf_summary else adaptive_dir / "adaptive_leaf_summary.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = output_dir / manifest_path

    leaf_summaries = read_leaf_summaries(summary_path)
    assignments = read_assignments(assignments_path)
    group_paths = build_group_paths(assignments, leaf_summaries, args.group_by, output_dir, args.filename_prefix)
    stats = split_raw_rows(
        Path(args.raw_input),
        assignments,
        leaf_summaries,
        group_paths,
        args.group_by,
        args.max_open_files,
        args.add_mining_meta,
        args.add_channel,
        args.channel_key,
        args.channel_prefix,
    )
    manifest_rows = build_manifest_rows(stats, group_paths, leaf_summaries)
    write_csv(manifest_path, manifest_rows)
    total = sum(row.count for row in stats.values())
    print(f"split {total} rows into {len(manifest_rows)} JSONL files -> {output_dir}")
    print(f"manifest -> {manifest_path}")


class GroupStats:
    def __init__(self) -> None:
        self.count = 0
        self.leaf_counts: Counter = Counter()
        self.label_counts: Counter = Counter()
        self.primary_scene_counts: Counter = Counter()
        self.secondary_scene_counts: Counter = Counter()
        self.actions: Counter = Counter()
        self.stop_reasons: Counter = Counter()
        self.purity_weighted_sum = 0.0


class WriterPool:
    def __init__(self, max_open_files: int) -> None:
        self.max_open_files = max(1, max_open_files)
        self.handles: "OrderedDict[Path, TextIO]" = OrderedDict()
        self.initialized: Set[Path] = set()

    def write(self, path: Path, line: str) -> None:
        handle = self._handle(path)
        handle.write(line)

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def _handle(self, path: Path) -> TextIO:
        handle = self.handles.pop(path, None)
        if handle is not None:
            self.handles[path] = handle
            return handle
        if len(self.handles) >= self.max_open_files:
            _, old_handle = self.handles.popitem(last=False)
            old_handle.close()
        mode = "a" if path in self.initialized else "w"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open(mode, encoding="utf-8")
        self.initialized.add(path)
        self.handles[path] = handle
        return handle


def read_leaf_summaries(path: Path) -> Dict[int, Dict[str, str]]:
    if not path.exists():
        return {}
    summaries: Dict[int, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "leaf_id" not in (reader.fieldnames or []):
            raise ValueError(f"missing leaf_id column in {path}")
        for row in reader:
            summaries[int(float(row["leaf_id"]))] = dict(row)
    return summaries


def read_assignments(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"adaptive assignments not found: {path}")
    assignments: Dict[int, Dict[str, Any]] = {}
    for item in iter_jsonl(path):
        row = int(item["row"])
        if row in assignments:
            raise ValueError(f"duplicate row assignment for row {row} in {path}")
        assignments[row] = item
    if not assignments:
        raise ValueError(f"no assignments found in {path}")
    return assignments


def build_group_paths(
        assignments: Dict[int, Dict[str, Any]],
        leaf_summaries: Dict[int, Dict[str, str]],
        group_by: str,
        output_dir: Path,
        filename_prefix: str) -> Dict[GroupKey, Path]:
    key_to_items: DefaultDict[GroupKey, List[Dict[str, Any]]] = defaultdict(list)
    for item in assignments.values():
        key_to_items[group_key(item, leaf_summaries, group_by)].append(item)
    used_names: Set[str] = set()
    paths = {}
    for key in sorted(key_to_items):
        filename = group_filename(filename_prefix, key, key_to_items[key], leaf_summaries, used_names)
        paths[key] = output_dir / filename
    return paths


def group_filename(
        prefix: str,
        key: GroupKey,
        items: Sequence[Dict[str, Any]],
        leaf_summaries: Dict[int, Dict[str, str]],
        used_names: Set[str]) -> str:
    leaf_ids = sorted({int(item["leaf_id"]) for item in items})
    dominant, _ = top_item(Counter(str(item.get("dominant_label") or UNKNOWN) for item in items))
    purity = average_leaf_purity(items, leaf_summaries)
    if len(leaf_ids) == 1:
        base = f"{prefix}_leaf_{leaf_ids[0]:06d}_{purity_token(purity)}_{dominant}"
    else:
        base = f"{prefix}_{sanitize_channel(key, max_len=80)}_{purity_token(purity)}_{dominant}"
    name = sanitize_channel(base)
    if name in used_names:
        name = f"{name}_{short_hash(key)}"
    used_names.add(name)
    return f"{name}.jsonl"


def average_leaf_purity(items: Sequence[Dict[str, Any]], leaf_summaries: Dict[int, Dict[str, str]]) -> float:
    if not items:
        return 0.0
    total = 0.0
    for item in items:
        leaf_id = int(item["leaf_id"])
        summary = leaf_summaries.get(leaf_id, {})
        total += to_float(item.get("leaf_purity") or summary.get("purity") or summary.get("avg_leaf_purity"))
    return total / len(items)


def split_raw_rows(
        raw_input: Path,
        assignments: Dict[int, Dict[str, Any]],
        leaf_summaries: Dict[int, Dict[str, str]],
        group_paths: Dict[GroupKey, Path],
        group_by: str,
        max_open_files: int,
        add_mining_meta: bool,
        add_channel: bool,
        channel_key: str,
        channel_prefix: str) -> Dict[GroupKey, GroupStats]:
    stats: DefaultDict[GroupKey, GroupStats] = defaultdict(GroupStats)
    writer = WriterPool(max_open_files)
    seen_rows: Set[int] = set()
    try:
        for row_idx, line in iter_jsonl_lines(resolve_input_files(raw_input)):
            assignment = assignments.get(row_idx)
            if assignment is None:
                continue
            key = group_key(assignment, leaf_summaries, group_by)
            writer.write(group_paths[key],
                         output_line(line, assignment, leaf_summaries, add_mining_meta, add_channel, channel_key,
                                     channel_prefix))
            update_stats(stats[key], assignment, leaf_summaries)
            seen_rows.add(row_idx)
    finally:
        writer.close()
    missing_rows = sorted(set(assignments) - seen_rows)
    if missing_rows:
        raise ValueError(f"{len(missing_rows)} assigned rows were not found in raw input; first rows: {missing_rows[:10]}")
    return dict(stats)


def output_line(
        raw_line: str,
        assignment: Dict[str, Any],
        leaf_summaries: Dict[int, Dict[str, str]],
        add_mining_meta: bool,
        add_channel: bool,
        channel_key: str,
        channel_prefix: str) -> str:
    if not add_mining_meta and not add_channel:
        return raw_line.rstrip("\n") + "\n"
    row = json.loads(raw_line)
    leaf_id = int(assignment["leaf_id"])
    summary = leaf_summaries.get(leaf_id, {})
    if add_channel:
        row[channel_key] = channel_name(channel_prefix, assignment, summary)
    if add_mining_meta:
        row["_adaptive_mining"] = {
            "leaf_id": leaf_id,
            "node_id": assignment.get("node_id"),
            "depth": assignment.get("depth"),
            "path": assignment.get("path"),
            "label": assignment.get("label"),
            "dominant_label": assignment.get("dominant_label"),
            "leaf_purity": assignment.get("leaf_purity"),
            "closeness": assignment.get("closeness"),
            "is_dominant_label": assignment.get("is_dominant_label"),
            "suggested_action": summary.get("suggested_action"),
            "suggested_target_count": to_int(summary.get("suggested_target_count")),
        }
    return json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"


def channel_name(prefix: str, assignment: Dict[str, Any], summary: Dict[str, str]) -> str:
    leaf_id = int(assignment["leaf_id"])
    purity = to_float(assignment.get("leaf_purity") or summary.get("purity") or summary.get("avg_leaf_purity"))
    dominant = str(assignment.get("dominant_label") or summary.get("dominant_label") or UNKNOWN)
    return sanitize_channel(f"{prefix}_leaf_{leaf_id:06d}_{purity_token(purity)}_{dominant}")


def sanitize_channel(value: str, max_len: int = 120) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return (value or "channel")[:max_len]


def purity_token(value: float) -> str:
    value = max(0.0, min(1.0, value))
    return f"p{int(round(value * 1000)):04d}"


def update_stats(stats: GroupStats, assignment: Dict[str, Any], leaf_summaries: Dict[int, Dict[str, str]]) -> None:
    leaf_id = int(assignment["leaf_id"])
    summary = leaf_summaries.get(leaf_id, {})
    stats.count += 1
    stats.leaf_counts[leaf_id] += 1
    stats.label_counts[str(assignment.get("label") or UNKNOWN)] += 1
    stats.primary_scene_counts[str(assignment.get("primary_scene") or UNKNOWN)] += 1
    stats.secondary_scene_counts[str(assignment.get("secondary_scene") or UNKNOWN)] += 1
    stats.actions[str(summary.get("suggested_action") or UNKNOWN)] += 1
    stats.stop_reasons[str(summary.get("stop_reason") or UNKNOWN)] += 1
    stats.purity_weighted_sum += to_float(assignment.get("leaf_purity"))


def build_manifest_rows(
        stats_by_group: Dict[GroupKey, GroupStats],
        group_paths: Dict[GroupKey, Path],
        leaf_summaries: Dict[int, Dict[str, str]]) -> List[Dict[str, Any]]:
    rows = []
    for key, stats in sorted(stats_by_group.items()):
        leaf_ids = sorted(int(v) for v in stats.leaf_counts)
        suggested_target = suggested_target_for_group(stats, leaf_summaries)
        dominant_label, dominant_label_count = top_item(stats.label_counts)
        primary_scene, primary_scene_count = top_item(stats.primary_scene_counts)
        secondary_scene, secondary_scene_count = top_item(stats.secondary_scene_counts)
        action, _ = top_item(stats.actions)
        stop_reason, _ = top_item(stats.stop_reasons)
        lateral, longitudinal = split_label(dominant_label)
        rows.append({
            "group_key": key,
            "file": str(group_paths[key]),
            "count": stats.count,
            "suggested_target_count": suggested_target,
            "sample_ratio": safe_div(suggested_target, stats.count),
            "num_leaves": len(leaf_ids),
            "leaf_ids": " ".join(str(v) for v in leaf_ids),
            "dominant_label": dominant_label,
            "dominant_lateral": lateral,
            "dominant_longitudinal": longitudinal,
            "dominant_label_count": dominant_label_count,
            "dominant_label_ratio": safe_div(dominant_label_count, stats.count),
            "primary_scene_top": primary_scene,
            "primary_scene_top_count": primary_scene_count,
            "primary_scene_top_ratio": safe_div(primary_scene_count, stats.count),
            "secondary_scene_top": secondary_scene,
            "secondary_scene_top_count": secondary_scene_count,
            "secondary_scene_top_ratio": safe_div(secondary_scene_count, stats.count),
            "avg_leaf_purity": safe_div(stats.purity_weighted_sum, stats.count),
            "suggested_action_top": action,
            "stop_reason_top": stop_reason,
        })
    return rows


def suggested_target_for_group(stats: GroupStats, leaf_summaries: Dict[int, Dict[str, str]]) -> int:
    total = 0.0
    for leaf_id, group_leaf_count in stats.leaf_counts.items():
        summary = leaf_summaries.get(int(leaf_id), {})
        leaf_size = max(1, to_int(summary.get("size")))
        leaf_target = to_int(summary.get("suggested_target_count"))
        total += leaf_target * safe_div(group_leaf_count, leaf_size)
    return int(round(total))


def group_key(assignment: Dict[str, Any], leaf_summaries: Dict[int, Dict[str, str]], group_by: str) -> str:
    leaf_id = int(assignment["leaf_id"])
    summary = leaf_summaries.get(leaf_id, {})
    label = str(assignment.get("label") or UNKNOWN)
    dominant_label = str(assignment.get("dominant_label") or summary.get("dominant_label") or UNKNOWN)
    primary_scene = str(assignment.get("primary_scene") or UNKNOWN)
    secondary_scene = str(assignment.get("secondary_scene") or UNKNOWN)
    action = str(summary.get("suggested_action") or UNKNOWN)
    if group_by == "leaf":
        return join_key("leaf", f"{leaf_id:06d}", dominant_label)
    if group_by == "primary_scene":
        return join_key("primary", primary_scene)
    if group_by == "secondary_scene":
        return join_key("secondary", secondary_scene)
    if group_by == "label":
        return join_key("label", label)
    if group_by == "dominant_label":
        return join_key("dominant", dominant_label)
    if group_by == "suggested_action":
        return join_key("action", action)
    if group_by == "primary_scene_leaf":
        return join_key("primary", primary_scene, "leaf", f"{leaf_id:06d}", dominant_label)
    if group_by == "secondary_scene_leaf":
        return join_key("secondary", secondary_scene, "leaf", f"{leaf_id:06d}", dominant_label)
    if group_by == "primary_scene_label":
        return join_key("primary", primary_scene, "label", label)
    if group_by == "primary_scene_dominant_label":
        return join_key("primary", primary_scene, "dominant", dominant_label)
    if group_by == "label_leaf":
        return join_key("label", label, "leaf", f"{leaf_id:06d}")
    raise ValueError(f"unsupported group-by: {group_by}")


def join_key(*parts: str) -> str:
    return "__".join(str(part) for part in parts)


def sanitize_filename(value: str, max_len: int = 180) -> str:
    value = re.sub(r"[^0-9A-Za-z._+=@-]+", "_", value)
    value = value.strip("._")
    return (value or "group")[:max_len]


def short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def top_item(counter: Counter) -> Tuple[str, int]:
    if not counter:
        return UNKNOWN, 0
    key, value = counter.most_common(1)[0]
    return str(key), int(value)


def to_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(round(float(value)))


def to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def iter_jsonl_lines(paths: Sequence[Path]) -> Iterable[Tuple[int, str]]:
    row_idx = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                if not line.endswith("\n"):
                    line += "\n"
                yield row_idx, line
                row_idx += 1


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
