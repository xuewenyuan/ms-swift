#!/usr/bin/env python3
"""Analyze clustered VLA data for sampling balance and review mining."""

import argparse
import csv
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np


UNKNOWN = "UNKNOWN"


class MetaInfo(NamedTuple):
    sample_id: str
    row_idx: Optional[int]
    scene_id: str
    primary_scene: str
    secondary_scene: str


class Stats(NamedTuple):
    total: int
    cluster_labels: Dict[int, Counter]
    cluster_primary_scenes: Dict[int, Counter]
    cluster_secondary_scenes: Dict[int, Counter]
    label_clusters: Dict[str, Counter]
    primary_scene_labels: Dict[str, Counter]
    secondary_scene_labels: Dict[str, Counter]
    buckets: Counter
    bucket_secondary_scenes: Dict[Tuple[str, str, int], Counter]
    bucket_closeness_sum: Dict[Tuple[str, str, int], float]
    cluster_closeness_sum: Dict[int, float]
    label_counts: Counter
    primary_scene_counts: Counter
    secondary_scene_counts: Counter
    closeness_values: np.ndarray
    missing_scene_ids: int
    scene_ids_not_in_map: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build VLA cluster mining reports for sampling balance and human review.")
    parser.add_argument("--feature-dir", required=True, help="Feature dump dir containing meta.jsonl.")
    parser.add_argument(
        "--cluster-dir",
        required=True,
        help="Cluster report dir containing cluster_assignments.jsonl, or the assignments JSONL path.")
    parser.add_argument(
        "--scene-map",
        required=True,
        help="JSON mapping scene_id to scene labels, e.g. {scene_id: {'专题': ..., '二级场景': ...}}.")
    parser.add_argument("--output-dir", required=True, help="Output dir for mining reports.")
    parser.add_argument(
        "--raw-input",
        default=None,
        help="Optional original JSONL file/dir. Used to recover scene_id when old meta.jsonl does not contain it.")
    parser.add_argument("--sample-id-key", default="sample_id", help="Sample id key in raw JSONL.")
    parser.add_argument("--scene-id-key", default="scene_id", help="Scene id key in raw JSONL.")
    parser.add_argument("--primary-scene-key", default="专题", help="Primary scene label key in --scene-map.")
    parser.add_argument("--secondary-scene-key", default="二级场景", help="Secondary scene label key in --scene-map.")
    parser.add_argument("--topn", type=int, default=20, help="Top rows/items in markdown summaries. Default: 20.")
    parser.add_argument(
        "--review-topn-per-type",
        type=int,
        default=2000,
        help="Max review candidates kept for each candidate type. Default: 2000.")
    parser.add_argument(
        "--low-closeness-quantile",
        type=float,
        default=0.02,
        help="Bottom quantile of centroid closeness treated as far/hard candidates. Default: 0.02.")
    parser.add_argument(
        "--minority-max-ratio",
        type=float,
        default=0.2,
        help="Cluster label ratio threshold for minority-in-cluster review candidates. Default: 0.2.")
    parser.add_argument(
        "--rare-scene-label-ratio",
        type=float,
        default=0.01,
        help="Scene-level label ratio threshold for rare scene/decision candidates. Default: 0.01.")
    parser.add_argument(
        "--min-cluster-count",
        type=int,
        default=100,
        help="Minimum cluster size before minority/mixed-cluster heuristics are applied. Default: 100.")
    parser.add_argument(
        "--min-scene-count",
        type=int,
        default=500,
        help="Minimum scene size before rare scene/decision heuristics are applied. Default: 500.")
    parser.add_argument(
        "--tail-bucket-count",
        type=int,
        default=200,
        help="Buckets at or below this count are treated as tail and suggested to keep. Default: 200.")
    parser.add_argument(
        "--high-bucket-count",
        type=int,
        default=10000,
        help="Buckets at or above this count are treated as frequent and suggested to downsample. Default: 10000.")
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="Optional target total for suggested bucket sample counts. If omitted, per-bucket heuristic counts are used.")
    parser.add_argument(
        "--downsample-power",
        type=float,
        default=0.5,
        help="Power used for target allocation/downsampling. 0.5 means sqrt-style downsampling. Default: 0.5.")
    parser.add_argument(
        "--min-conditional-count",
        type=int,
        default=5000,
        help="Minimum (primary scene, label) count to suggest label-conditioned clustering. Default: 5000.")
    parser.add_argument(
        "--max-conditional-clusters",
        type=int,
        default=50,
        help="Maximum suggested clusters for one label-conditioned clustering job. Default: 50.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_map = read_json(Path(args.scene_map))
    raw_scene_lookup = load_raw_scene_lookup(args.raw_input, args.sample_id_key, args.scene_id_key)
    meta_rows = load_feature_meta(
        Path(args.feature_dir),
        scene_map,
        raw_scene_lookup,
        primary_key=args.primary_scene_key,
        secondary_key=args.secondary_scene_key,
    )
    assignments_path = resolve_assignments_path(Path(args.cluster_dir))
    stats = collect_stats(meta_rows, assignments_path)
    derived = build_derived_stats(stats)
    thresholds = {
        "low_closeness": float(np.quantile(stats.closeness_values, args.low_closeness_quantile))
        if len(stats.closeness_values) else None
    }

    write_cluster_decision_distribution(output_dir / "cluster_decision_distribution.csv", stats, derived)
    write_cluster_scene_distribution(output_dir / "cluster_primary_scene_distribution.csv", stats.cluster_primary_scenes)
    write_cluster_scene_distribution(output_dir / "cluster_secondary_scene_distribution.csv", stats.cluster_secondary_scenes)
    write_label_cluster_distribution(output_dir / "label_cluster_distribution.csv", stats)
    write_scene_decision_counts(output_dir / "scene_decision_counts.csv", stats)
    sampling_rows = build_sampling_rows(stats, derived, args)
    write_csv(output_dir / "sampling_buckets.csv", sampling_rows)
    conditional_jobs = build_conditional_cluster_jobs(stats, args)
    write_jsonl(output_dir / "conditional_cluster_jobs.jsonl", conditional_jobs)
    review_counts = write_review_candidates(
        output_dir / "review_candidates.jsonl",
        meta_rows,
        assignments_path,
        stats,
        derived,
        thresholds,
        args,
    )

    summary = build_summary(stats, sampling_rows, conditional_jobs, review_counts, thresholds, args)
    (output_dir / "mining_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "mining_summary.md").write_text(render_markdown_summary(summary, args.topn), encoding="utf-8")
    print(f"wrote mining reports for {stats.total} rows -> {output_dir}")


def load_feature_meta(
        feature_dir: Path,
        scene_map: Dict[str, Any],
        raw_scene_lookup: Tuple[Dict[str, str], Dict[int, str]],
        *,
        primary_key: str,
        secondary_key: str) -> List[MetaInfo]:
    rows: List[Tuple[int, MetaInfo]] = []
    sample_to_scene, row_to_scene = raw_scene_lookup
    for meta_path in resolve_meta_files(feature_dir):
        for local_idx, meta in enumerate(iter_jsonl(meta_path)):
            row_idx = to_optional_int(meta.get("row_idx"))
            sample_id = str(meta.get("sample_id") or f"row_{len(rows):08d}")
            scene_id = scene_id_from_meta(meta)
            if not scene_id:
                scene_id = sample_to_scene.get(sample_id) or (row_to_scene.get(row_idx) if row_idx is not None else None)
            scene_id = str(scene_id or "")
            labels = scene_map.get(scene_id) if scene_id else None
            primary = scene_label(labels, primary_key)
            secondary = scene_label(labels, secondary_key)
            sort_key = row_idx if row_idx is not None else len(rows)
            rows.append((sort_key, MetaInfo(sample_id, row_idx, scene_id or UNKNOWN, primary, secondary)))
    rows.sort(key=lambda item: item[0])
    return [item[1] for item in rows]


def resolve_meta_files(feature_dir: Path) -> List[Path]:
    if (feature_dir / "shards").is_dir():
        files = sorted(p / "meta.jsonl" for p in (feature_dir / "shards").iterdir() if (p / "meta.jsonl").is_file())
        if not files:
            raise ValueError(f"no shard meta.jsonl files found under {feature_dir / 'shards'}")
        return files
    meta_path = feature_dir / "meta.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.jsonl not found: {meta_path}")
    return [meta_path]


def load_raw_scene_lookup(
        raw_input: Optional[str],
        sample_id_key: str,
        scene_id_key: str) -> Tuple[Dict[str, str], Dict[int, str]]:
    if not raw_input:
        return {}, {}
    sample_to_scene: Dict[str, str] = {}
    row_to_scene: Dict[int, str] = {}
    for row_idx, row in enumerate(iter_jsonl_files(resolve_input_files(Path(raw_input)))):
        scene_id = row.get(scene_id_key)
        if scene_id is None:
            continue
        scene_id = str(scene_id)
        row_to_scene[row_idx] = scene_id
        sample_id = row.get(sample_id_key) or row.get("id")
        if sample_id is not None:
            sample_to_scene[str(sample_id)] = scene_id
    return sample_to_scene, row_to_scene


def collect_stats(meta_rows: Sequence[MetaInfo], assignments_path: Path) -> Stats:
    cluster_labels: DefaultDict[int, Counter] = defaultdict(Counter)
    cluster_primary_scenes: DefaultDict[int, Counter] = defaultdict(Counter)
    cluster_secondary_scenes: DefaultDict[int, Counter] = defaultdict(Counter)
    label_clusters: DefaultDict[str, Counter] = defaultdict(Counter)
    primary_scene_labels: DefaultDict[str, Counter] = defaultdict(Counter)
    secondary_scene_labels: DefaultDict[str, Counter] = defaultdict(Counter)
    buckets: Counter = Counter()
    bucket_secondary_scenes: DefaultDict[Tuple[str, str, int], Counter] = defaultdict(Counter)
    bucket_closeness_sum: DefaultDict[Tuple[str, str, int], float] = defaultdict(float)
    cluster_closeness_sum: DefaultDict[int, float] = defaultdict(float)
    label_counts: Counter = Counter()
    primary_scene_counts: Counter = Counter()
    secondary_scene_counts: Counter = Counter()
    closeness_values: List[float] = []
    missing_scene_ids = 0
    scene_ids_not_in_map = 0

    total = 0
    for line_idx, assignment in enumerate(iter_jsonl(assignments_path)):
        row = int(assignment.get("row", line_idx))
        if row >= len(meta_rows):
            raise ValueError(f"assignment row {row} out of meta range {len(meta_rows)}")
        meta = meta_rows[row]
        cluster_id = int(assignment["cluster_id"])
        lateral = str(assignment.get("lateral") or UNKNOWN)
        longitudinal = str(assignment.get("longitudinal") or UNKNOWN)
        label = decision_label(lateral, longitudinal)
        closeness = float(assignment.get("closeness", 0.0))
        bucket = (meta.primary_scene, label, cluster_id)

        cluster_labels[cluster_id][label] += 1
        cluster_primary_scenes[cluster_id][meta.primary_scene] += 1
        cluster_secondary_scenes[cluster_id][meta.secondary_scene] += 1
        label_clusters[label][cluster_id] += 1
        primary_scene_labels[meta.primary_scene][label] += 1
        secondary_scene_labels[meta.secondary_scene][label] += 1
        buckets[bucket] += 1
        bucket_secondary_scenes[bucket][meta.secondary_scene] += 1
        bucket_closeness_sum[bucket] += closeness
        cluster_closeness_sum[cluster_id] += closeness
        label_counts[label] += 1
        primary_scene_counts[meta.primary_scene] += 1
        secondary_scene_counts[meta.secondary_scene] += 1
        closeness_values.append(closeness)
        if meta.scene_id == UNKNOWN:
            missing_scene_ids += 1
        if meta.scene_id != UNKNOWN and meta.primary_scene == UNKNOWN and meta.secondary_scene == UNKNOWN:
            scene_ids_not_in_map += 1
        total += 1

    return Stats(
        total=total,
        cluster_labels=dict(cluster_labels),
        cluster_primary_scenes=dict(cluster_primary_scenes),
        cluster_secondary_scenes=dict(cluster_secondary_scenes),
        label_clusters=dict(label_clusters),
        primary_scene_labels=dict(primary_scene_labels),
        secondary_scene_labels=dict(secondary_scene_labels),
        buckets=buckets,
        bucket_secondary_scenes=dict(bucket_secondary_scenes),
        bucket_closeness_sum=dict(bucket_closeness_sum),
        cluster_closeness_sum=dict(cluster_closeness_sum),
        label_counts=label_counts,
        primary_scene_counts=primary_scene_counts,
        secondary_scene_counts=secondary_scene_counts,
        closeness_values=np.asarray(closeness_values, dtype=np.float32),
        missing_scene_ids=missing_scene_ids,
        scene_ids_not_in_map=scene_ids_not_in_map,
    )


def build_derived_stats(stats: Stats) -> Dict[str, Any]:
    cluster_dominant_label = {}
    cluster_purity = {}
    cluster_entropy = {}
    for cluster_id, counter in stats.cluster_labels.items():
        total = sum(counter.values())
        dominant_label, dominant_count = counter.most_common(1)[0]
        cluster_dominant_label[cluster_id] = dominant_label
        cluster_purity[cluster_id] = safe_div(dominant_count, total)
        cluster_entropy[cluster_id] = entropy(counter)
    label_cluster_entropy = {label: entropy(counter) for label, counter in stats.label_clusters.items()}
    return {
        "cluster_dominant_label": cluster_dominant_label,
        "cluster_purity": cluster_purity,
        "cluster_entropy": cluster_entropy,
        "label_cluster_entropy": label_cluster_entropy,
    }


def build_sampling_rows(stats: Stats, derived: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = []
    raw_weights = {}
    for bucket, count in stats.buckets.items():
        primary_scene, label, cluster_id = bucket
        secondary_counter = stats.bucket_secondary_scenes.get(bucket, Counter())
        cluster_count = sum(stats.cluster_labels.get(cluster_id, Counter()).values())
        label_count = stats.label_counts[label]
        scene_label_count = stats.primary_scene_labels[primary_scene][label]
        cluster_purity = derived["cluster_purity"].get(cluster_id, 0.0)
        label_ratio_in_cluster = safe_div(stats.cluster_labels[cluster_id][label], cluster_count)
        mean_closeness = safe_div(stats.bucket_closeness_sum.get(bucket, 0.0), count)
        action, modifier = suggest_sampling_action(count, cluster_purity, args)
        raw_weight = (count ** args.downsample_power) * modifier
        raw_weights[bucket] = raw_weight
        row = {
            "primary_scene": primary_scene,
            "secondary_scene_top": format_top_distribution(secondary_counter, topn=5),
            "lateral": split_label(label)[0],
            "longitudinal": split_label(label)[1],
            "label": label,
            "cluster_id": cluster_id,
            "count": count,
            "label_count": label_count,
            "scene_label_count": scene_label_count,
            "cluster_count": cluster_count,
            "bucket_ratio_in_label": safe_div(count, label_count),
            "bucket_ratio_in_scene_label": safe_div(count, scene_label_count),
            "label_ratio_in_cluster": label_ratio_in_cluster,
            "cluster_purity": cluster_purity,
            "mean_closeness": mean_closeness,
            "is_tail_bucket": count <= args.tail_bucket_count,
            "suggested_action": action,
            "suggested_sample_weight": modifier,
        }
        rows.append(row)

    if args.target_total:
        total_weight = sum(raw_weights.values())
        for row in rows:
            bucket = (row["primary_scene"], row["label"], int(row["cluster_id"]))
            row["suggested_target_count"] = int(round(args.target_total * safe_div(raw_weights[bucket], total_weight)))
    else:
        for row in rows:
            count = int(row["count"])
            if count <= args.tail_bucket_count:
                target = count
            else:
                target = int(math.ceil((count ** args.downsample_power) * (args.tail_bucket_count ** (1 - args.downsample_power))))
                target = min(count, max(args.tail_bucket_count, target))
            if row["suggested_action"] == "review_first":
                target = min(target, args.tail_bucket_count)
            row["suggested_target_count"] = target

    rows.sort(key=lambda x: (str(x["primary_scene"]), str(x["label"]), int(x["cluster_id"])))
    return rows


def suggest_sampling_action(count: int, cluster_purity: float, args: argparse.Namespace) -> Tuple[str, float]:
    if cluster_purity < 0.5 and count >= args.min_cluster_count:
        return "review_first", 0.2
    if count <= args.tail_bucket_count:
        return "keep_tail", 1.5
    if count >= args.high_bucket_count and cluster_purity >= 0.8:
        return "downsample_frequent", 0.6
    if cluster_purity < 0.7:
        return "keep_hard_or_review", 1.2
    return "balanced_keep", 1.0


def build_conditional_cluster_jobs(stats: Stats, args: argparse.Namespace) -> List[Dict[str, Any]]:
    jobs = []
    for primary_scene, counter in stats.primary_scene_labels.items():
        for label, count in counter.items():
            if count < args.min_conditional_count:
                continue
            suggested_k = max(2, int(round(math.sqrt(count) / 20.0)))
            suggested_k = min(args.max_conditional_clusters, suggested_k)
            jobs.append({
                "primary_scene": primary_scene,
                "lateral": split_label(label)[0],
                "longitudinal": split_label(label)[1],
                "label": label,
                "count": count,
                "suggested_num_clusters": suggested_k,
                "reason": "large primary_scene + decision bucket; useful for label-conditioned clustering",
            })
    jobs.sort(key=lambda x: x["count"], reverse=True)
    return jobs


def write_review_candidates(
        path: Path,
        meta_rows: Sequence[MetaInfo],
        assignments_path: Path,
        stats: Stats,
        derived: Dict[str, Any],
        thresholds: Dict[str, Optional[float]],
        args: argparse.Namespace) -> Dict[str, int]:
    heaps: DefaultDict[str, List[Tuple[float, int, Dict[str, Any]]]] = defaultdict(list)
    seq = 0
    low_threshold = thresholds["low_closeness"]

    for line_idx, assignment in enumerate(iter_jsonl(assignments_path)):
        row = int(assignment.get("row", line_idx))
        meta = meta_rows[row]
        cluster_id = int(assignment["cluster_id"])
        lateral = str(assignment.get("lateral") or UNKNOWN)
        longitudinal = str(assignment.get("longitudinal") or UNKNOWN)
        label = decision_label(lateral, longitudinal)
        closeness = float(assignment.get("closeness", 0.0))
        cluster_count = sum(stats.cluster_labels[cluster_id].values())
        label_ratio_in_cluster = safe_div(stats.cluster_labels[cluster_id][label], cluster_count)
        cluster_purity = derived["cluster_purity"].get(cluster_id, 0.0)
        dominant_label = derived["cluster_dominant_label"].get(cluster_id, UNKNOWN)
        primary_label_ratio = safe_div(
            stats.primary_scene_labels[meta.primary_scene][label], stats.primary_scene_counts[meta.primary_scene])
        secondary_label_ratio = safe_div(
            stats.secondary_scene_labels[meta.secondary_scene][label], stats.secondary_scene_counts[meta.secondary_scene])

        base = {
            "row": row,
            "row_idx": meta.row_idx,
            "sample_id": meta.sample_id,
            "scene_id": meta.scene_id,
            "primary_scene": meta.primary_scene,
            "secondary_scene": meta.secondary_scene,
            "cluster_id": cluster_id,
            "lateral": lateral,
            "longitudinal": longitudinal,
            "label": label,
            "closeness": closeness,
            "cluster_purity": cluster_purity,
            "cluster_label_ratio": label_ratio_in_cluster,
            "dominant_label": dominant_label,
            "primary_scene_label_ratio": primary_label_ratio,
            "secondary_scene_label_ratio": secondary_label_ratio,
        }

        if low_threshold is not None and closeness <= low_threshold:
            candidate = dict(base)
            candidate.update({
                "candidate_type": "low_cluster_closeness",
                "score": float(low_threshold - closeness),
                "reason": "far from assigned cluster centroid; likely hard or outlier sample",
            })
            seq = push_candidate(heaps, candidate, args.review_topn_per_type, seq)

        if (cluster_count >= args.min_cluster_count and label != dominant_label
                and label_ratio_in_cluster <= args.minority_max_ratio):
            candidate = dict(base)
            candidate.update({
                "candidate_type": "minority_in_cluster",
                "score": float((1.0 - label_ratio_in_cluster) + max(0.0, (low_threshold or closeness) - closeness)),
                "reason": "decision label is a small minority inside a cluster dominated by another decision",
            })
            seq = push_candidate(heaps, candidate, args.review_topn_per_type, seq)

        if (stats.primary_scene_counts[meta.primary_scene] >= args.min_scene_count
                and primary_label_ratio <= args.rare_scene_label_ratio):
            candidate = dict(base)
            candidate.update({
                "candidate_type": "rare_scene_decision",
                "score": float(1.0 - primary_label_ratio),
                "reason": "decision label is rare inside its primary scene; review for tail value or label conflict",
            })
            seq = push_candidate(heaps, candidate, args.review_topn_per_type, seq)

    counts = {}
    with path.open("w", encoding="utf-8") as f:
        for candidate_type in sorted(heaps):
            rows = [item[2] for item in sorted(heaps[candidate_type], key=lambda x: x[0], reverse=True)]
            counts[candidate_type] = len(rows)
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return counts


def push_candidate(
        heaps: DefaultDict[str, List[Tuple[float, int, Dict[str, Any]]]],
        candidate: Dict[str, Any],
        limit: int,
        seq: int) -> int:
    candidate_type = candidate["candidate_type"]
    item = (float(candidate["score"]), seq, candidate)
    heap = heaps[candidate_type]
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)
    return seq + 1


def build_summary(
        stats: Stats,
        sampling_rows: Sequence[Dict[str, Any]],
        conditional_jobs: Sequence[Dict[str, Any]],
        review_counts: Dict[str, int],
        thresholds: Dict[str, Optional[float]],
        args: argparse.Namespace) -> Dict[str, Any]:
    largest_buckets = sorted(sampling_rows, key=lambda x: int(x["count"]), reverse=True)[:args.topn]
    tail_buckets = [row for row in sampling_rows if row["is_tail_bucket"]]
    return {
        "num_rows": stats.total,
        "num_labels": len(stats.label_counts),
        "num_primary_scenes": len(stats.primary_scene_counts),
        "num_secondary_scenes": len(stats.secondary_scene_counts),
        "num_clusters": len(stats.cluster_labels),
        "num_sampling_buckets": len(sampling_rows),
        "num_tail_buckets": len(tail_buckets),
        "missing_scene_ids": stats.missing_scene_ids,
        "scene_ids_not_in_map": stats.scene_ids_not_in_map,
        "low_closeness_threshold": thresholds["low_closeness"],
        "review_candidate_counts": review_counts,
        "num_conditional_cluster_jobs": len(conditional_jobs),
        "largest_sampling_buckets": largest_buckets,
        "largest_labels": counter_rows(stats.label_counts, stats.total, args.topn),
        "largest_primary_scenes": counter_rows(stats.primary_scene_counts, stats.total, args.topn),
        "outputs": [
            "cluster_decision_distribution.csv",
            "cluster_primary_scene_distribution.csv",
            "cluster_secondary_scene_distribution.csv",
            "label_cluster_distribution.csv",
            "scene_decision_counts.csv",
            "sampling_buckets.csv",
            "conditional_cluster_jobs.jsonl",
            "review_candidates.jsonl",
        ],
    }


def render_markdown_summary(summary: Dict[str, Any], topn: int) -> str:
    lines = [
        "# VLA Cluster Mining Summary",
        "",
        "## Overview",
        "",
        f"- Rows: {summary['num_rows']}",
        f"- Labels: {summary['num_labels']}",
        f"- Primary scenes: {summary['num_primary_scenes']}",
        f"- Secondary scenes: {summary['num_secondary_scenes']}",
        f"- Clusters: {summary['num_clusters']}",
        f"- Sampling buckets: {summary['num_sampling_buckets']}",
        f"- Tail buckets: {summary['num_tail_buckets']}",
        f"- Missing scene ids: {summary['missing_scene_ids']}",
        f"- Scene ids not in map: {summary['scene_ids_not_in_map']}",
        f"- Low closeness threshold: {summary['low_closeness_threshold']}",
        "",
        "## Review Candidates",
        "",
    ]
    for candidate_type, count in sorted(summary["review_candidate_counts"].items()):
        lines.append(f"- `{candidate_type}`: {count}")
    lines.extend(["", "## Largest Sampling Buckets", ""])
    lines.append("| primary_scene | label | cluster | count | action | target |")
    lines.append("| --- | --- | ---: | ---: | --- | ---: |")
    for row in summary["largest_sampling_buckets"][:topn]:
        lines.append(
            f"| `{row['primary_scene']}` | `{row['label']}` | {row['cluster_id']} | {row['count']} | "
            f"{row['suggested_action']} | {row['suggested_target_count']} |")
    lines.extend(["", "## Outputs", ""])
    for output in summary["outputs"]:
        lines.append(f"- `{output}`")
    lines.append("")
    return "\n".join(lines)


def write_cluster_decision_distribution(path: Path, stats: Stats, derived: Dict[str, Any]) -> None:
    rows = []
    for cluster_id, counter in stats.cluster_labels.items():
        total = sum(counter.values())
        dominant = derived["cluster_dominant_label"][cluster_id]
        for label, count in counter.most_common():
            lateral, longitudinal = split_label(label)
            rows.append({
                "cluster_id": cluster_id,
                "lateral": lateral,
                "longitudinal": longitudinal,
                "label": label,
                "count": count,
                "ratio": safe_div(count, total),
                "is_dominant": label == dominant,
                "cluster_purity": derived["cluster_purity"][cluster_id],
                "cluster_entropy": derived["cluster_entropy"][cluster_id],
            })
    rows.sort(key=lambda x: (int(x["cluster_id"]), -int(x["count"])))
    write_csv(path, rows)


def write_cluster_scene_distribution(path: Path, cluster_scenes: Dict[int, Counter]) -> None:
    rows = []
    for cluster_id, counter in cluster_scenes.items():
        total = sum(counter.values())
        for scene, count in counter.most_common():
            rows.append({"cluster_id": cluster_id, "scene": scene, "count": count, "ratio": safe_div(count, total)})
    rows.sort(key=lambda x: (int(x["cluster_id"]), -int(x["count"])))
    write_csv(path, rows)


def write_label_cluster_distribution(path: Path, stats: Stats) -> None:
    rows = []
    for label, counter in stats.label_clusters.items():
        total = sum(counter.values())
        lateral, longitudinal = split_label(label)
        for cluster_id, count in counter.most_common():
            rows.append({
                "lateral": lateral,
                "longitudinal": longitudinal,
                "label": label,
                "cluster_id": cluster_id,
                "count": count,
                "ratio_in_label": safe_div(count, total),
                "label_cluster_entropy": entropy(counter),
            })
    rows.sort(key=lambda x: (str(x["label"]), -int(x["count"])))
    write_csv(path, rows)


def write_scene_decision_counts(path: Path, stats: Stats) -> None:
    rows = []
    for level, scene_labels in [("primary", stats.primary_scene_labels), ("secondary", stats.secondary_scene_labels)]:
        for scene, counter in scene_labels.items():
            total = sum(counter.values())
            for label, count in counter.most_common():
                lateral, longitudinal = split_label(label)
                rows.append({
                    "scene_level": level,
                    "scene": scene,
                    "lateral": lateral,
                    "longitudinal": longitudinal,
                    "label": label,
                    "count": count,
                    "ratio_in_scene": safe_div(count, total),
                })
    rows.sort(key=lambda x: (str(x["scene_level"]), str(x["scene"]), -int(x["count"])))
    write_csv(path, rows)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def resolve_assignments_path(path: Path) -> Path:
    if path.is_file():
        return path
    assignments = path / "cluster_assignments.jsonl"
    if not assignments.is_file():
        raise FileNotFoundError(f"cluster_assignments.jsonl not found: {assignments}")
    return assignments


def scene_id_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    for key in ("scene_id", "scene"):
        value = meta.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def scene_label(labels: Any, key: str) -> str:
    if not isinstance(labels, dict):
        return UNKNOWN
    value = labels.get(key)
    if value in (None, ""):
        return UNKNOWN
    return str(value)


def decision_label(lateral: str, longitudinal: str) -> str:
    return f"{lateral}|{longitudinal}"


def split_label(label: str) -> Tuple[str, str]:
    parts = label.split("|", 1)
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
            value -= p * math.log(p)
    return float(value)


def counter_rows(counter: Counter, total: int, topn: int) -> List[Dict[str, Any]]:
    return [{"key": key, "count": count, "ratio": safe_div(count, total)} for key, count in counter.most_common(topn)]


def format_top_distribution(counter: Counter, topn: int) -> str:
    total = sum(counter.values())
    return ";".join(f"{key}:{count}:{safe_div(count, total):.4f}" for key, count in counter.most_common(topn))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def to_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_input_files(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix != ".jsonl":
            raise ValueError(f"input file must be a .jsonl file: {path}")
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.rglob("*.jsonl") if p.is_file())
        if not files:
            raise ValueError(f"no .jsonl files found under input directory: {path}")
        return files
    raise FileNotFoundError(f"input path does not exist: {path}")


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
