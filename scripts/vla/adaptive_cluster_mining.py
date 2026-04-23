#!/usr/bin/env python3
"""Adaptive recursive clustering for VLA sampling coverage and label-error mining."""

import argparse
import csv
import heapq
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from analyze_cluster_mining import (
    UNKNOWN,
    decision_label,
    entropy,
    format_top_distribution,
    load_feature_meta,
    load_raw_meta_lookup,
    read_json,
    safe_div,
    split_label,
    write_csv,
    write_jsonl,
)
from cluster_features_faiss import (
    configure_cuda_visible_devices,
    import_faiss,
    load_dump,
    l2_normalize_matrix,
    parse_gpu_devices,
    validate_faiss_gpu,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recursively split VLA clusters for sampling diversity first, then label-error review.")
    parser.add_argument("--feature-dir", required=True, help="Feature dump dir from dump_messages_features.py.")
    parser.add_argument("--output-dir", required=True, help="Output dir for adaptive clustering reports.")
    parser.add_argument("--matrix", default="input_text_hash", help="Matrix name/path used for clustering.")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine", help="Clustering metric.")
    parser.add_argument("--scene-map", default=None, help="Optional scene_id -> scene labels JSON.")
    parser.add_argument("--raw-input", default=None, help="Optional raw JSONL file/dir to recover source metadata.")
    parser.add_argument("--sample-id-key", default="sample_id", help="Sample id key in raw JSONL.")
    parser.add_argument("--scene-id-key", default="scene_id", help="Scene id key in raw JSONL.")
    parser.add_argument("--sample-token-key", default="sample_token", help="Sample token key in raw JSONL.")
    parser.add_argument("--timestamp-key", default="timestamp", help="Timestamp key in raw JSONL.")
    parser.add_argument("--primary-scene-key", default="专题", help="Primary scene label key in --scene-map.")
    parser.add_argument("--secondary-scene-key", default="二级场景", help="Secondary scene label key in --scene-map.")

    parser.add_argument("--purity-threshold", type=float, default=0.7, help="Leaf label-purity target. Default: 0.7.")
    parser.add_argument("--min-leaf-size", type=int, default=500, help="Small leaves are treated as tail. Default: 500.")
    parser.add_argument(
        "--max-leaf-size",
        type=int,
        default=50000,
        help="Pure leaves larger than this are split for diversity when possible. Default: 50000.")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum recursive split depth. Default: 3.")
    parser.add_argument(
        "--min-split-size",
        type=int,
        default=None,
        help="Minimum node size that can be split. Default: 2 * min_leaf_size.")
    parser.add_argument("--root-k", type=int, default=None, help="Optional child k for the root split.")
    parser.add_argument("--child-k", type=int, default=None, help="Optional fixed child k for non-root splits.")
    parser.add_argument("--min-child-k", type=int, default=2, help="Minimum child clusters per split. Default: 2.")
    parser.add_argument("--max-child-k", type=int, default=50, help="Maximum child clusters per split. Default: 50.")
    parser.add_argument(
        "--large-pure-action",
        choices=["split", "keep"],
        default="split",
        help="Whether high-purity but oversized nodes should be split for diversity. Default: split.")

    parser.add_argument("--niter", type=int, default=50, help="FAISS k-means iterations. Default: 50.")
    parser.add_argument("--nredo", type=int, default=3, help="FAISS k-means restarts. Default: 3.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    parser.add_argument("--max-points-per-centroid", type=int, default=256, help="FAISS training cap per centroid.")
    parser.add_argument("--gpu", action="store_true", help="Use FAISS GPU k-means when selected.")
    parser.add_argument("--gpu-devices", default=None, help="Comma-separated GPU ids, e.g. 0,1,2,3.")
    parser.add_argument(
        "--kmeans-device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Device used for FAISS k-means training. Default: auto.")
    parser.add_argument(
        "--gpu-min-train-points",
        type=int,
        default=100000,
        help="In auto mode, use CPU when sampled train points are below this threshold. Default: 100000.")

    parser.add_argument("--target-total", type=int, default=None, help="Optional total target count for leaf sampling.")
    parser.add_argument("--downsample-power", type=float, default=0.5, help="Power for target allocation. Default: 0.5.")
    parser.add_argument(
        "--review-topn-per-type",
        type=int,
        default=50000,
        help="Max review candidates kept for each candidate type. Default: 50000.")
    parser.add_argument(
        "--minority-max-ratio",
        type=float,
        default=0.3,
        help="Minority label ratio threshold in high-purity leaves. Default: 0.3.")
    parser.add_argument("--no-assignments", action="store_true", help="Do not write per-row leaf assignments JSONL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.min_split_size is None:
        args.min_split_size = max(args.min_leaf_size * 2, args.min_leaf_size + 1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_devices = parse_gpu_devices(args.gpu_devices)
    configure_cuda_visible_devices(gpu_devices if wants_gpu(args, gpu_devices) else None)
    faiss = import_faiss()
    validate_faiss_gpu(faiss, wants_gpu(args, gpu_devices))

    dump = load_dump(Path(args.feature_dir), args.matrix)
    x = np.ascontiguousarray(dump["matrix"].astype(np.float32, copy=False))
    if args.metric == "cosine":
        x = l2_normalize_matrix(x)

    scene_map = read_json(Path(args.scene_map)) if args.scene_map else {}
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
    if len(meta_rows) != x.shape[0]:
        raise ValueError(f"metadata rows ({len(meta_rows)}) != matrix rows ({x.shape[0]})")

    labels = load_decision_labels(dump["meta_rows"])
    if len(labels) != x.shape[0]:
        raise ValueError(f"label rows ({len(labels)}) != matrix rows ({x.shape[0]})")

    tree, leaves, leaf_ids, leaf_closeness = build_adaptive_tree(faiss, x, labels, args, gpu_devices)
    enrich_leaf_summaries(leaves, labels, meta_rows, args)
    assign_sampling_targets(leaves, args)

    write_csv(output_dir / "adaptive_leaf_summary.csv", [leaf["summary"] for leaf in leaves])
    write_csv(output_dir / "adaptive_leaf_decision_distribution.csv", leaf_decision_distribution(leaves))
    write_csv(output_dir / "adaptive_leaf_scene_distribution.csv", leaf_scene_distribution(leaves))
    (output_dir / "adaptive_tree.json").write_text(json.dumps(tree, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    if not args.no_assignments:
        write_leaf_assignments(output_dir / "adaptive_leaf_assignments.jsonl", meta_rows, labels, leaf_ids,
                               leaf_closeness, leaves)
    review_counts = write_review_candidates(output_dir / "adaptive_review_candidates.jsonl", meta_rows, labels, leaf_ids,
                                            leaf_closeness, leaves, args)
    summary = build_summary(tree, leaves, labels, review_counts, args)
    (output_dir / "adaptive_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                       encoding="utf-8")
    (output_dir / "adaptive_summary.md").write_text(render_markdown_summary(summary), encoding="utf-8")
    print(f"wrote adaptive clustering reports for {x.shape[0]} rows, {len(leaves)} leaves -> {output_dir}")


def load_decision_labels(meta_rows: Sequence[Dict[str, Any]]) -> List[str]:
    labels = []
    for row in meta_rows:
        output = row.get("output") or {}
        labels.append(decision_label(str(output.get("lateral") or UNKNOWN), str(output.get("longitudinal") or UNKNOWN)))
    return labels


def build_adaptive_tree(faiss: Any, x: np.ndarray, labels: Sequence[str], args: argparse.Namespace,
                        gpu_devices: Optional[Sequence[int]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray,
                                                                       np.ndarray]:
    n = x.shape[0]
    root_indices = np.arange(n, dtype=np.int64)
    queue = deque([{
        "node_id": 0,
        "parent_id": None,
        "depth": 0,
        "path": "0",
        "indices": root_indices,
        "incoming_closeness": np.zeros(n, dtype=np.float32),
    }])
    next_node_id = 1
    next_leaf_id = 0
    nodes = []
    leaves = []
    leaf_ids = np.full(n, -1, dtype=np.int32)
    leaf_closeness = np.full(n, np.nan, dtype=np.float32)

    while queue:
        node = queue.popleft()
        indices = node["indices"]
        label_counter = Counter(labels[int(i)] for i in indices)
        stats = node_stats(label_counter, len(indices))
        split_reason = split_reason_for_node(stats, node["depth"], args)
        child_k = choose_child_k(len(indices), node["depth"], args) if split_reason else 0
        can_split = split_reason is not None and child_k >= 2 and len(indices) >= child_k
        node_record = {
            "node_id": node["node_id"],
            "parent_id": node["parent_id"],
            "depth": node["depth"],
            "path": node["path"],
            "size": int(len(indices)),
            "dominant_label": stats["dominant_label"],
            "dominant_count": stats["dominant_count"],
            "purity": stats["purity"],
            "num_labels": stats["num_labels"],
            "label_entropy": stats["label_entropy"],
            "split_reason": split_reason,
            "child_k": child_k if can_split else 0,
            "children": [],
        }
        nodes.append(node_record)

        if can_split:
            child_labels, child_closeness = run_node_kmeans(faiss, x, indices, child_k, node["node_id"], args,
                                                            gpu_devices)
            unique_child_ids = sorted(set(int(v) for v in child_labels))
            if len(unique_child_ids) <= 1:
                make_leaf(node, node_record, indices, stats, "unsplittable", next_leaf_id, leaf_ids, leaf_closeness,
                          leaves)
                next_leaf_id += 1
                continue
            for child_rank, child_cluster in enumerate(unique_child_ids):
                mask = child_labels == child_cluster
                child_indices = indices[mask]
                child_node_id = next_node_id
                next_node_id += 1
                child_path = f"{node['path']}.{child_rank}"
                node_record["children"].append(child_node_id)
                queue.append({
                    "node_id": child_node_id,
                    "parent_id": node["node_id"],
                    "depth": node["depth"] + 1,
                    "path": child_path,
                    "indices": child_indices,
                    "incoming_closeness": child_closeness[mask],
                })
            continue

        stop_reason = stop_reason_for_node(stats, node["depth"], args)
        make_leaf(node, node_record, indices, stats, stop_reason, next_leaf_id, leaf_ids, leaf_closeness, leaves)
        next_leaf_id += 1

    return {"nodes": nodes}, leaves, leaf_ids, leaf_closeness


def make_leaf(node: Dict[str, Any], node_record: Dict[str, Any], indices: np.ndarray, stats: Dict[str, Any],
              stop_reason: str, leaf_id: int, leaf_ids: np.ndarray, leaf_closeness: np.ndarray,
              leaves: List[Dict[str, Any]]) -> None:
    leaf_ids[indices] = leaf_id
    incoming_closeness = node["incoming_closeness"]
    if len(incoming_closeness) == len(indices):
        leaf_closeness[indices] = incoming_closeness
    node_record["leaf_id"] = leaf_id
    node_record["stop_reason"] = stop_reason
    leaves.append({
        "leaf_id": leaf_id,
        "node_id": node["node_id"],
        "parent_id": node["parent_id"],
        "depth": node["depth"],
        "path": node["path"],
        "indices": indices,
        "label_counter": Counter(),  # filled by enrich_leaf_summaries
        "summary": {
            "leaf_id": leaf_id,
            "node_id": node["node_id"],
            "parent_id": node["parent_id"],
            "depth": node["depth"],
            "path": node["path"],
            "size": int(len(indices)),
            "dominant_label": stats["dominant_label"],
            "dominant_lateral": split_label(stats["dominant_label"])[0],
            "dominant_longitudinal": split_label(stats["dominant_label"])[1],
            "dominant_count": stats["dominant_count"],
            "purity": stats["purity"],
            "num_labels": stats["num_labels"],
            "label_entropy": stats["label_entropy"],
            "stop_reason": stop_reason,
        },
    })


def node_stats(label_counter: Counter, size: int) -> Dict[str, Any]:
    if not label_counter:
        return {
            "size": size,
            "dominant_label": UNKNOWN,
            "dominant_count": 0,
            "purity": 0.0,
            "num_labels": 0,
            "label_entropy": 0.0,
        }
    dominant_label, dominant_count = label_counter.most_common(1)[0]
    return {
        "size": size,
        "dominant_label": dominant_label,
        "dominant_count": int(dominant_count),
        "purity": safe_div(dominant_count, size),
        "num_labels": len(label_counter),
        "label_entropy": entropy(label_counter),
    }


def split_reason_for_node(stats: Dict[str, Any], depth: int, args: argparse.Namespace) -> Optional[str]:
    size = int(stats["size"])
    if depth >= args.max_depth:
        return None
    if size < args.min_split_size:
        return None
    if stats["purity"] < args.purity_threshold:
        return "low_purity"
    if args.large_pure_action == "split" and size > args.max_leaf_size:
        return "large_pure_diversity"
    return None


def stop_reason_for_node(stats: Dict[str, Any], depth: int, args: argparse.Namespace) -> str:
    size = int(stats["size"])
    if size < args.min_leaf_size:
        return "tail_leaf"
    if stats["purity"] < args.purity_threshold:
        if depth >= args.max_depth:
            return "impure_max_depth"
        return "impure_unsplit"
    if size > args.max_leaf_size:
        return "large_pure_leaf"
    return "pure_leaf"


def choose_child_k(size: int, depth: int, args: argparse.Namespace) -> int:
    if depth == 0 and args.root_k:
        return min(max(2, args.root_k), size)
    if depth > 0 and args.child_k:
        return min(max(2, args.child_k), size)
    by_target_size = int(math.ceil(size / max(1, args.max_leaf_size)))
    k = max(args.min_child_k, by_target_size)
    return min(args.max_child_k, k, size)


def run_node_kmeans(faiss: Any, x: np.ndarray, indices: np.ndarray, k: int, node_id: int, args: argparse.Namespace,
                    gpu_devices: Optional[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    subset = np.ascontiguousarray(x[indices].astype(np.float32, copy=False))
    use_gpu = resolve_node_gpu(args, gpu_devices, subset.shape[0], k)
    validate_faiss_gpu(faiss, use_gpu)
    gpu_setting: Any = len(gpu_devices) if use_gpu and gpu_devices else use_gpu
    kmeans = faiss.Kmeans(
        subset.shape[1],
        k,
        niter=args.niter,
        nredo=args.nredo,
        seed=args.seed + int(node_id),
        verbose=True,
        gpu=gpu_setting,
        max_points_per_centroid=args.max_points_per_centroid,
        spherical=args.metric == "cosine",
    )
    kmeans.train(subset)
    centroids = np.ascontiguousarray(kmeans.centroids.astype(np.float32))
    if args.metric == "cosine":
        centroids = l2_normalize_matrix(centroids)
        index = faiss.IndexFlatIP(subset.shape[1])
    else:
        index = faiss.IndexFlatL2(subset.shape[1])
    index.add(centroids)
    scores, child_ids = index.search(subset, 1)
    child_ids = child_ids.reshape(-1).astype(np.int32)
    scores = scores.reshape(-1).astype(np.float32)
    closeness = scores if args.metric == "cosine" else -scores
    return child_ids, closeness


def wants_gpu(args: argparse.Namespace, gpu_devices: Optional[Sequence[int]]) -> bool:
    return bool(args.kmeans_device == "gpu" or args.gpu or gpu_devices)


def resolve_node_gpu(args: argparse.Namespace, gpu_devices: Optional[Sequence[int]], node_size: int, k: int) -> bool:
    gpu_requested = bool(args.gpu or gpu_devices)
    sampled_train_points = min(node_size, max(1, k) * max(1, args.max_points_per_centroid))
    if args.kmeans_device == "cpu":
        return False
    if args.kmeans_device == "gpu":
        return True
    if not gpu_requested:
        return False
    return sampled_train_points >= args.gpu_min_train_points


def enrich_leaf_summaries(leaves: List[Dict[str, Any]], labels: Sequence[str], meta_rows: Sequence[Any],
                          args: argparse.Namespace) -> None:
    for leaf in leaves:
        indices = leaf["indices"]
        label_counter = Counter(labels[int(i)] for i in indices)
        primary_counter = Counter(meta_rows[int(i)].primary_scene for i in indices)
        secondary_counter = Counter(meta_rows[int(i)].secondary_scene for i in indices)
        leaf["label_counter"] = label_counter
        leaf["primary_scene_counter"] = primary_counter
        leaf["secondary_scene_counter"] = secondary_counter
        summary = leaf["summary"]
        summary["primary_scene_top"] = format_top_distribution(primary_counter, topn=5)
        summary["secondary_scene_top"] = format_top_distribution(secondary_counter, topn=5)
        summary["suggested_action"] = suggested_action(summary, args)
        summary["suggested_sample_weight"] = suggested_weight(summary)


def suggested_action(summary: Dict[str, Any], args: argparse.Namespace) -> str:
    size = int(summary["size"])
    purity = float(summary["purity"])
    stop_reason = str(summary["stop_reason"])
    if stop_reason in {"impure_max_depth", "impure_unsplit"} or purity < args.purity_threshold:
        return "review_first"
    if size < args.min_leaf_size:
        return "keep_tail"
    if size > args.max_leaf_size:
        return "downsample_frequent"
    if stop_reason == "large_pure_leaf":
        return "downsample_frequent"
    return "balanced_keep"


def suggested_weight(summary: Dict[str, Any]) -> float:
    action = summary["suggested_action"]
    if action == "review_first":
        return 0.2
    if action == "keep_tail":
        return 1.5
    if action == "downsample_frequent":
        return 0.6
    return 1.0


def assign_sampling_targets(leaves: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    raw_weights = []
    for leaf in leaves:
        summary = leaf["summary"]
        raw_weights.append((float(summary["size"]) ** args.downsample_power) * float(summary["suggested_sample_weight"]))
    if args.target_total:
        total_weight = sum(raw_weights)
        for leaf, weight in zip(leaves, raw_weights):
            leaf["summary"]["suggested_target_count"] = int(round(args.target_total * safe_div(weight, total_weight)))
        return
    for leaf in leaves:
        summary = leaf["summary"]
        size = int(summary["size"])
        action = summary["suggested_action"]
        if action == "keep_tail":
            target = size
        elif action == "review_first":
            target = min(size, args.min_leaf_size)
        elif action == "downsample_frequent":
            target = min(size, args.max_leaf_size)
        else:
            target = size
        summary["suggested_target_count"] = int(target)


def leaf_decision_distribution(leaves: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for leaf in leaves:
        total = int(leaf["summary"]["size"])
        for label, count in leaf["label_counter"].most_common():
            lateral, longitudinal = split_label(label)
            rows.append({
                "leaf_id": leaf["leaf_id"],
                "node_id": leaf["node_id"],
                "depth": leaf["depth"],
                "lateral": lateral,
                "longitudinal": longitudinal,
                "label": label,
                "count": int(count),
                "ratio": safe_div(count, total),
                "is_dominant": label == leaf["summary"]["dominant_label"],
            })
    return rows


def leaf_scene_distribution(leaves: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for leaf in leaves:
        total = int(leaf["summary"]["size"])
        for level, counter in [("primary", leaf["primary_scene_counter"]), ("secondary", leaf["secondary_scene_counter"])]:
            for scene, count in counter.most_common():
                rows.append({
                    "leaf_id": leaf["leaf_id"],
                    "node_id": leaf["node_id"],
                    "depth": leaf["depth"],
                    "scene_level": level,
                    "scene": scene,
                    "count": int(count),
                    "ratio": safe_div(count, total),
                })
    return rows


def write_leaf_assignments(path: Path, meta_rows: Sequence[Any], labels: Sequence[str], leaf_ids: np.ndarray,
                           leaf_closeness: np.ndarray, leaves: Sequence[Dict[str, Any]]) -> None:
    leaf_lookup = {leaf["leaf_id"]: leaf for leaf in leaves}
    with path.open("w", encoding="utf-8") as f:
        for row, meta in enumerate(meta_rows):
            leaf_id = int(leaf_ids[row])
            leaf = leaf_lookup[leaf_id]
            lateral, longitudinal = split_label(labels[row])
            item = {
                "row": row,
                "row_idx": meta.row_idx,
                "sample_id": meta.sample_id,
                "scene_id": meta.scene_id,
                "sample_token": meta.sample_token,
                "timestamp": meta.timestamp,
                "primary_scene": meta.primary_scene,
                "secondary_scene": meta.secondary_scene,
                "leaf_id": leaf_id,
                "node_id": leaf["node_id"],
                "depth": leaf["depth"],
                "path": leaf["path"],
                "lateral": lateral,
                "longitudinal": longitudinal,
                "label": labels[row],
                "dominant_label": leaf["summary"]["dominant_label"],
                "leaf_purity": leaf["summary"]["purity"],
                "closeness": none_if_nan(float(leaf_closeness[row])),
                "is_dominant_label": labels[row] == leaf["summary"]["dominant_label"],
            }
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_review_candidates(path: Path, meta_rows: Sequence[Any], labels: Sequence[str], leaf_ids: np.ndarray,
                            leaf_closeness: np.ndarray, leaves: Sequence[Dict[str, Any]],
                            args: argparse.Namespace) -> Dict[str, int]:
    leaf_lookup = {leaf["leaf_id"]: leaf for leaf in leaves}
    heaps: DefaultDict[str, List[Tuple[float, int, Dict[str, Any]]]] = defaultdict(list)
    seq = 0
    for row, label in enumerate(labels):
        leaf_id = int(leaf_ids[row])
        leaf = leaf_lookup[leaf_id]
        summary = leaf["summary"]
        label_count = leaf["label_counter"][label]
        label_ratio = safe_div(label_count, int(summary["size"]))
        base = review_base(row, meta_rows[row], label, leaf, leaf_closeness[row], label_ratio)
        if (float(summary["purity"]) >= args.purity_threshold and label != summary["dominant_label"]
                and label_ratio <= args.minority_max_ratio):
            candidate = dict(base)
            candidate.update({
                "candidate_type": "minority_in_pure_leaf",
                "score": float(summary["purity"]) - label_ratio,
                "reason": "sample label is a minority inside a high-purity leaf",
            })
            seq = push_candidate(heaps, candidate, args.review_topn_per_type, seq)
        if summary["stop_reason"] in {"impure_max_depth", "impure_unsplit"}:
            candidate = dict(base)
            candidate.update({
                "candidate_type": "in_impure_leaf",
                "score": 1.0 - float(summary["purity"]),
                "reason": "leaf remains mixed after adaptive splitting; review for multi-solution or label conflict",
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


def review_base(row: int, meta: Any, label: str, leaf: Dict[str, Any], closeness: float,
                label_ratio: float) -> Dict[str, Any]:
    lateral, longitudinal = split_label(label)
    return {
        "row": row,
        "row_idx": meta.row_idx,
        "sample_id": meta.sample_id,
        "scene_id": meta.scene_id,
        "sample_token": meta.sample_token,
        "timestamp": meta.timestamp,
        "primary_scene": meta.primary_scene,
        "secondary_scene": meta.secondary_scene,
        "leaf_id": leaf["leaf_id"],
        "node_id": leaf["node_id"],
        "depth": leaf["depth"],
        "path": leaf["path"],
        "lateral": lateral,
        "longitudinal": longitudinal,
        "label": label,
        "dominant_label": leaf["summary"]["dominant_label"],
        "leaf_purity": leaf["summary"]["purity"],
        "leaf_label_ratio": label_ratio,
        "closeness": none_if_nan(float(closeness)),
        "stop_reason": leaf["summary"]["stop_reason"],
    }


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


def build_summary(tree: Dict[str, Any], leaves: Sequence[Dict[str, Any]], labels: Sequence[str], review_counts: Dict[str,
                                                                                                                      int],
                  args: argparse.Namespace) -> Dict[str, Any]:
    target_total = sum(int(leaf["summary"]["suggested_target_count"]) for leaf in leaves)
    raw_total = sum(int(leaf["summary"]["size"]) for leaf in leaves)
    leaf_actions = Counter(leaf["summary"]["suggested_action"] for leaf in leaves)
    stop_reasons = Counter(leaf["summary"]["stop_reason"] for leaf in leaves)
    pure_coverage = sum(int(leaf["summary"]["size"]) for leaf in leaves
                        if float(leaf["summary"]["purity"]) >= args.purity_threshold)
    return {
        "num_rows": raw_total,
        "num_labels": len(set(labels)),
        "num_nodes": len(tree["nodes"]),
        "num_leaves": len(leaves),
        "purity_threshold": args.purity_threshold,
        "sample_coverage_at_purity": safe_div(pure_coverage, raw_total),
        "raw_total": raw_total,
        "suggested_total": target_total,
        "keep_ratio": safe_div(target_total, raw_total),
        "leaf_actions": dict(leaf_actions),
        "stop_reasons": dict(stop_reasons),
        "review_candidate_counts": review_counts,
        "largest_leaves": sorted([leaf["summary"] for leaf in leaves], key=lambda x: int(x["size"]), reverse=True)[:20],
        "lowest_purity_leaves": sorted([leaf["summary"] for leaf in leaves], key=lambda x: float(x["purity"]))[:20],
        "outputs": [
            "adaptive_leaf_summary.csv",
            "adaptive_leaf_decision_distribution.csv",
            "adaptive_leaf_scene_distribution.csv",
            "adaptive_tree.json",
            "adaptive_leaf_assignments.jsonl",
            "adaptive_review_candidates.jsonl",
        ],
    }


def render_markdown_summary(summary: Dict[str, Any]) -> str:
    lines = [
        "# Adaptive VLA Cluster Mining Summary",
        "",
        "## Overview",
        "",
        f"- Rows: {summary['num_rows']}",
        f"- Labels: {summary['num_labels']}",
        f"- Nodes: {summary['num_nodes']}",
        f"- Leaves: {summary['num_leaves']}",
        f"- Purity threshold: {summary['purity_threshold']}",
        f"- Sample coverage at purity: {summary['sample_coverage_at_purity']:.4f}",
        f"- Suggested total: {summary['suggested_total']}",
        f"- Keep ratio: {summary['keep_ratio']:.4f}",
        "",
        "## Leaf Actions",
        "",
    ]
    for action, count in sorted(summary["leaf_actions"].items()):
        lines.append(f"- `{action}`: {count}")
    lines.extend(["", "## Review Candidates", ""])
    for candidate_type, count in sorted(summary["review_candidate_counts"].items()):
        lines.append(f"- `{candidate_type}`: {count}")
    lines.extend(["", "## Largest Leaves", ""])
    lines.append("| leaf | size | purity | dominant_label | action | target | stop_reason |")
    lines.append("| ---: | ---: | ---: | --- | --- | ---: | --- |")
    for row in summary["largest_leaves"]:
        lines.append(
            f"| {row['leaf_id']} | {row['size']} | {float(row['purity']):.4f} | `{row['dominant_label']}` | "
            f"{row['suggested_action']} | {row['suggested_target_count']} | {row['stop_reason']} |")
    lines.append("")
    return "\n".join(lines)


def none_if_nan(value: float) -> Optional[float]:
    if math.isnan(value):
        return None
    return value


if __name__ == "__main__":
    main()
