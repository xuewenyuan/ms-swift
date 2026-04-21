#!/usr/bin/env python3
"""Cluster VLA feature dumps with FAISS and write an analysis report."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster FAISS-friendly VLA feature dumps and report potential abnormal classes.")
    parser.add_argument("--input-dir", required=True, help="Feature dump dir from dump_messages_features.py.")
    parser.add_argument("--output-dir", required=True, help="Output dir for cluster assignments and reports.")
    parser.add_argument(
        "--matrix",
        default="curation_fused",
        help="Matrix name in manifest.json or path to an N x D float32 .npy file. Default: curation_fused.")
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=None,
        help="Override cluster count. Default is the number of unique lateral/longitudinal decision combinations.")
    parser.add_argument(
        "--metric",
        choices=["cosine", "l2"],
        default="cosine",
        help="Distance metric for clustering/reporting. Use cosine for L2-normalized embeddings. Default: cosine.")
    parser.add_argument("--niter", type=int, default=50, help="FAISS k-means iterations. Default: 50.")
    parser.add_argument("--nredo", type=int, default=3, help="FAISS k-means restarts. Default: 3.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    parser.add_argument("--gpu", action="store_true", help="Use FAISS GPU resources when available.")
    parser.add_argument(
        "--max-points-per-centroid",
        type=int,
        default=256,
        help="FAISS training sample cap per centroid. Default: 256.")
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Clusters smaller than this are marked anomalous. Default: max(2, ceil(0.1%% of rows)).")
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="Top rows/clusters to include in human-readable reports. Default: 10.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss = import_faiss()
    dump = load_dump(input_dir, args.matrix)
    meta_rows = dump["meta_rows"]
    x = dump["matrix"]
    if len(meta_rows) != x.shape[0]:
        raise ValueError(f"metadata rows ({len(meta_rows)}) != matrix rows ({x.shape[0]})")
    if x.shape[0] == 0:
        raise ValueError("empty feature matrix")

    x = np.ascontiguousarray(x.astype(np.float32, copy=False))
    if args.metric == "cosine":
        x = l2_normalize_matrix(x)

    decision_pairs = [decision_pair(row) for row in meta_rows]
    pair_counts = Counter(decision_pairs)
    num_clusters = args.num_clusters or len(pair_counts)
    num_clusters = validate_num_clusters(num_clusters, x.shape[0])

    cluster_ids, closeness, centroids = run_faiss_kmeans(
        faiss,
        x,
        num_clusters,
        metric=args.metric,
        niter=args.niter,
        nredo=args.nredo,
        seed=args.seed,
        gpu=args.gpu,
        max_points_per_centroid=args.max_points_per_centroid,
    )

    summary = build_summary(
        meta_rows,
        decision_pairs,
        pair_counts,
        cluster_ids,
        closeness,
        centroids,
        metric=args.metric,
        matrix_name=dump["matrix_name"],
        num_clusters=num_clusters,
        min_cluster_size=args.min_cluster_size,
        topn=args.topn,
    )

    write_assignments(output_dir / "cluster_assignments.jsonl", meta_rows, decision_pairs, cluster_ids, closeness,
                      args.metric)
    (output_dir / "cluster_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "cluster_report.md").write_text(render_markdown(summary, args.topn), encoding="utf-8")
    np.save(output_dir / "centroids.npy", centroids.astype(np.float32))
    print(f"clustered {x.shape[0]} rows into {num_clusters} clusters -> {output_dir}")


def import_faiss():
    try:
        import faiss  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "faiss is required. Please install faiss-cpu or faiss-gpu in this environment.") from e
    return faiss


def load_dump(input_dir: Path, matrix_arg: str) -> Dict[str, Any]:
    if (input_dir / "shards").is_dir():
        shard_dirs = sorted(p for p in (input_dir / "shards").iterdir() if p.is_dir())
        if not shard_dirs:
            raise ValueError(f"no shard dirs found under {input_dir / 'shards'}")
        rows: List[Tuple[int, Dict[str, Any], np.ndarray]] = []
        matrix_name = matrix_arg
        for shard_dir in shard_dirs:
            shard = load_single_dump(shard_dir, matrix_arg)
            for local_idx, meta in enumerate(shard["meta_rows"]):
                rows.append((int(meta.get("row_idx", len(rows))), meta, shard["matrix"][local_idx]))
            matrix_name = shard["matrix_name"]
        rows.sort(key=lambda item: item[0])
        return {
            "meta_rows": [item[1] for item in rows],
            "matrix": np.stack([item[2] for item in rows]).astype(np.float32),
            "matrix_name": matrix_name,
        }
    return load_single_dump(input_dir, matrix_arg)


def load_single_dump(input_dir: Path, matrix_arg: str) -> Dict[str, Any]:
    manifest = read_json(input_dir / "manifest.json")
    matrix_path, matrix_name = resolve_matrix_path(input_dir, manifest, matrix_arg)
    meta_rows = list(iter_jsonl(input_dir / manifest.get("files", {}).get("meta", "meta.jsonl")))
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D: {matrix_path}, got shape={matrix.shape}")
    return {"meta_rows": meta_rows, "matrix": matrix, "matrix_name": matrix_name}


def resolve_matrix_path(input_dir: Path, manifest: Dict[str, Any], matrix_arg: str) -> Tuple[Path, str]:
    candidate = Path(matrix_arg)
    if candidate.exists():
        return candidate, candidate.stem
    input_relative_candidate = input_dir / matrix_arg
    if input_relative_candidate.exists():
        return input_relative_candidate, input_relative_candidate.stem
    matrices = manifest.get("matrices", {})
    if matrix_arg not in matrices:
        available = ", ".join(sorted(matrices))
        raise ValueError(f"matrix '{matrix_arg}' not found in manifest. Available matrices: {available}")
    return input_dir / matrices[matrix_arg]["path"], matrix_arg


def run_faiss_kmeans(
        faiss: Any,
        x: np.ndarray,
        k: int,
        *,
        metric: str,
        niter: int,
        nredo: int,
        seed: int,
        gpu: bool,
        max_points_per_centroid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kwargs = {
        "niter": niter,
        "nredo": nredo,
        "verbose": True,
        "seed": seed,
        "gpu": gpu,
        "max_points_per_centroid": max_points_per_centroid,
        "spherical": metric == "cosine",
    }
    kmeans = faiss.Kmeans(x.shape[1], k, **kwargs)
    kmeans.train(x)
    centroids = np.ascontiguousarray(kmeans.centroids.astype(np.float32))
    if metric == "cosine":
        centroids = l2_normalize_matrix(centroids)
        index = faiss.IndexFlatIP(x.shape[1])
    else:
        index = faiss.IndexFlatL2(x.shape[1])
    index.add(centroids)
    scores, ids = index.search(x, 1)
    ids = ids.reshape(-1).astype(np.int64)
    scores = scores.reshape(-1).astype(np.float32)
    closeness = scores if metric == "cosine" else -scores
    return ids, closeness, centroids


def build_summary(
        meta_rows: Sequence[Dict[str, Any]],
        decision_pairs: Sequence[Tuple[str, str]],
        pair_counts: Counter,
        cluster_ids: np.ndarray,
        closeness: np.ndarray,
        centroids: np.ndarray,
        *,
        metric: str,
        matrix_name: str,
        num_clusters: int,
        min_cluster_size: Optional[int],
        topn: int) -> Dict[str, Any]:
    n = len(meta_rows)
    if min_cluster_size is None:
        min_cluster_size = max(2, math.ceil(n * 0.001))
    cluster_to_rows: Dict[int, List[int]] = defaultdict(list)
    for i, cluster_id in enumerate(cluster_ids):
        cluster_to_rows[int(cluster_id)].append(i)

    global_mean = float(np.mean(closeness))
    global_std = float(np.std(closeness))
    clusters = []
    for cluster_id in range(num_clusters):
        idxs = cluster_to_rows.get(cluster_id, [])
        pair_counter = Counter(decision_pairs[i] for i in idxs)
        size = len(idxs)
        values = closeness[idxs] if idxs else np.array([], dtype=np.float32)
        dominant_pair, dominant_count = pair_counter.most_common(1)[0] if pair_counter else (("NA", "NA"), 0)
        low_examples = sorted(idxs, key=lambda i: float(closeness[i]))[:topn]
        clusters.append({
            "cluster_id": cluster_id,
            "size": size,
            "size_ratio": safe_div(size, n),
            "dominant_decision_pair": list(dominant_pair),
            "dominant_decision_count": dominant_count,
            "dominant_decision_ratio": safe_div(dominant_count, size),
            "num_decision_pairs": len(pair_counter),
            "mean_closeness": float(np.mean(values)) if size else None,
            "min_closeness": float(np.min(values)) if size else None,
            "max_closeness": float(np.max(values)) if size else None,
            "decision_pair_distribution": pair_distribution(pair_counter),
            "low_closeness_examples": [
                sample_view(meta_rows[i], decision_pairs[i], cluster_ids[i], closeness[i], metric) for i in low_examples
            ],
            "flags": cluster_flags(size, min_cluster_size, pair_counter, values, global_mean, global_std),
        })

    all_low_examples = sorted(range(n), key=lambda i: float(closeness[i]))[:topn]
    anomaly_clusters = [c for c in clusters if c["flags"]]
    anomaly_clusters.sort(key=lambda c: (c["size"], c["mean_closeness"] if c["mean_closeness"] is not None else -1.0))

    return {
        "matrix": matrix_name,
        "metric": metric,
        "num_rows": n,
        "num_clusters": num_clusters,
        "num_decision_pairs": len(pair_counts),
        "min_cluster_size": min_cluster_size,
        "global_closeness": {
            "mean": global_mean,
            "std": global_std,
            "min": float(np.min(closeness)),
            "max": float(np.max(closeness)),
        },
        "decision_pair_counts": pair_distribution(pair_counts),
        "clusters": clusters,
        "anomaly_clusters": anomaly_clusters[:topn],
        "global_low_closeness_examples": [
            sample_view(meta_rows[i], decision_pairs[i], cluster_ids[i], closeness[i], metric) for i in all_low_examples
        ],
        "notes": [
            "Default cluster count is the number of unique lateral/longitudinal decision combinations.",
            "For cosine, closeness is centroid cosine similarity; lower values are farther from the assigned centroid.",
            "For l2, closeness is negative squared L2 distance; lower values are farther from the assigned centroid.",
        ],
        "centroid_norms": [float(v) for v in np.linalg.norm(centroids, axis=1)],
    }


def cluster_flags(
        size: int,
        min_cluster_size: int,
        pair_counter: Counter,
        values: np.ndarray,
        global_mean: float,
        global_std: float) -> List[str]:
    flags = []
    if size == 0:
        flags.append("empty_cluster")
        return flags
    if size < min_cluster_size:
        flags.append("small_cluster")
    dominant_ratio = pair_counter.most_common(1)[0][1] / size if pair_counter else 0.0
    if dominant_ratio < 0.5 and len(pair_counter) > 1:
        flags.append("mixed_decisions")
    mean_value = float(np.mean(values))
    if global_std > 0 and mean_value < global_mean - 2.0 * global_std:
        flags.append("low_centroid_closeness")
    return flags


def write_assignments(
        path: Path,
        meta_rows: Sequence[Dict[str, Any]],
        decision_pairs: Sequence[Tuple[str, str]],
        cluster_ids: np.ndarray,
        closeness: np.ndarray,
        metric: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, meta in enumerate(meta_rows):
            row = {
                "row": i,
                "row_idx": meta.get("row_idx"),
                "sample_id": meta.get("sample_id"),
                "cluster_id": int(cluster_ids[i]),
                "lateral": decision_pairs[i][0],
                "longitudinal": decision_pairs[i][1],
                "closeness": float(closeness[i]),
                "metric": metric,
            }
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def render_markdown(summary: Dict[str, Any], topn: int) -> str:
    lines = [
        "# VLA FAISS Cluster Report",
        "",
        "## Overview",
        "",
        f"- Matrix: `{summary['matrix']}`",
        f"- Metric: `{summary['metric']}`",
        f"- Rows: {summary['num_rows']}",
        f"- Clusters: {summary['num_clusters']}",
        f"- Decision pairs: {summary['num_decision_pairs']}",
        f"- Min cluster size flag: {summary['min_cluster_size']}",
        f"- Global closeness mean/std: {summary['global_closeness']['mean']:.6f} / "
        f"{summary['global_closeness']['std']:.6f}",
        "",
        "## Potential Abnormal Clusters",
        "",
    ]
    if not summary["anomaly_clusters"]:
        lines.append("No clusters were flagged by the default heuristics.")
    else:
        lines.extend(cluster_table(summary["anomaly_clusters"]))
    lines.extend(["", "## Smallest Clusters", ""])
    lines.extend(cluster_table(sorted(summary["clusters"], key=lambda c: c["size"])[:topn]))
    lines.extend(["", "## Lowest Closeness Samples", ""])
    lines.extend(sample_table(summary["global_low_closeness_examples"]))
    lines.extend(["", "## Largest Clusters", ""])
    lines.extend(cluster_table(sorted(summary["clusters"], key=lambda c: c["size"], reverse=True)[:topn]))
    lines.extend(["", "## Cluster Decision Pair Distribution", ""])
    lines.extend(cluster_distribution_sections(summary["clusters"], topn))
    lines.extend(["", "## Decision Pair Counts", ""])
    lines.append("| lateral | longitudinal | count | ratio |")
    lines.append("| --- | --- | ---: | ---: |")
    for item in summary["decision_pair_counts"][:topn * 3]:
        lines.append(f"| `{item['lateral']}` | `{item['longitudinal']}` | {item['count']} | {item['ratio']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def cluster_table(clusters: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| cluster | size | dominant lateral | dominant longitudinal | dominant ratio | pairs | mean closeness | flags |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for cluster in clusters:
        lat, lon = cluster["dominant_decision_pair"]
        mean = cluster["mean_closeness"]
        mean_text = "" if mean is None else f"{mean:.6f}"
        flags = ", ".join(cluster["flags"])
        lines.append(
            f"| {cluster['cluster_id']} | {cluster['size']} | `{lat}` | `{lon}` | "
            f"{cluster['dominant_decision_ratio']:.4f} | {cluster['num_decision_pairs']} | {mean_text} | {flags} |")
    return lines


def cluster_distribution_sections(clusters: Sequence[Dict[str, Any]], topn: int) -> List[str]:
    lines = []
    for cluster in sorted(clusters, key=lambda c: c["cluster_id"]):
        lines.extend([
            f"### Cluster {cluster['cluster_id']}",
            "",
            f"- Size: {cluster['size']}",
            f"- Decision pairs: {cluster['num_decision_pairs']}",
            "",
            "| lateral | longitudinal | count | cluster ratio |",
            "| --- | --- | ---: | ---: |",
        ])
        distribution = cluster["decision_pair_distribution"]
        for item in distribution[:topn]:
            lines.append(
                f"| `{item['lateral']}` | `{item['longitudinal']}` | {item['count']} | {item['ratio']:.4f} |")
        if len(distribution) > topn:
            remaining_count = sum(item["count"] for item in distribution[topn:])
            remaining_ratio = sum(item["ratio"] for item in distribution[topn:])
            lines.append(f"| `<OTHER>` | `<OTHER>` | {remaining_count} | {remaining_ratio:.4f} |")
        lines.append("")
    return lines


def sample_table(samples: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| row_idx | sample_id | cluster | lateral | longitudinal | closeness |",
        "| ---: | --- | ---: | --- | --- | ---: |",
    ]
    for sample in samples:
        lines.append(
            f"| {sample['row_idx']} | `{sample['sample_id']}` | {sample['cluster_id']} | "
            f"`{sample['lateral']}` | `{sample['longitudinal']}` | {sample['closeness']:.6f} |")
    return lines


def pair_distribution(counter: Counter) -> List[Dict[str, Any]]:
    total = sum(counter.values())
    rows = []
    for (lateral, longitudinal), count in counter.most_common():
        rows.append({
            "lateral": lateral,
            "longitudinal": longitudinal,
            "count": int(count),
            "ratio": safe_div(count, total),
        })
    return rows


def sample_view(
        meta: Dict[str, Any],
        decision: Tuple[str, str],
        cluster_id: int,
        closeness: float,
        metric: str) -> Dict[str, Any]:
    return {
        "row_idx": meta.get("row_idx"),
        "sample_id": meta.get("sample_id"),
        "cluster_id": int(cluster_id),
        "lateral": decision[0],
        "longitudinal": decision[1],
        "closeness": float(closeness),
        "metric": metric,
    }


def decision_pair(row: Dict[str, Any]) -> Tuple[str, str]:
    output = row.get("output") or {}
    lateral = output.get("lateral")
    longitudinal = output.get("longitudinal")
    return str(lateral or "NA"), str(longitudinal or "NA")


def validate_num_clusters(k: int, n: int) -> int:
    if k <= 0:
        raise ValueError(f"num_clusters must be positive, got {k}")
    if k > n:
        print(f"num_clusters={k} is larger than rows={n}; using {n}.")
        return n
    return k


def l2_normalize_matrix(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
