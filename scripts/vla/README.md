# VLA Feature Dump

`dump_messages_features.py` extracts curation features from ms-swift messages-format VLA data.
`--input` accepts either one `.jsonl` file or a directory. For directory input,
the script recursively finds all `*.jsonl` files and processes them in sorted
path order.
By default, `input_text_hash.npy` is built from the ego-state segment between
`自车当前状态信息有：` and `自车的驾驶决策包括`, instead of the full prompt. Use
`--input-text-scope full` to restore full-prompt text features.

## FAISS-friendly dump

```shell
python3 scripts/vla/dump_messages_features.py \
  --input /path/to/train_jsonl_dir_or_file \
  --output-dir /path/to/feature_dump \
  --scene-id-key scene_id \
  --max-traj-points 10
```

The output directory contains:

```text
feature_dump/
  manifest.json
  meta.jsonl
  index.jsonl
  sample_ids.txt
  matrices/
    input_text_hash.npy
    output_decision_hash.npy
    output_traj.npy
    output_traj_mask.npy
    curation_fused.npy
```

All `.npy` matrices are `float32` and use the same row order as `index.jsonl` and
`sample_ids.txt`.

- `meta.jsonl`: human-readable metadata, media refs, label signature, counts.
- `input_text_hash.npy`: lightweight hashed prompt vector, L2-normalized.
- `output_decision_hash.npy`: lightweight decision-token vector, L2-normalized.
- `output_traj.npy`: fixed-length trajectory-token vector.
- `output_traj_mask.npy`: valid trajectory point mask.
- `curation_fused.npy`: cheap curation vector for smoke tests and early sampling.

These matrices are not meant to replace real VLM embeddings. For production
clustering, append real visual/text/fused embeddings as additional `N x D`
`float32` `.npy` files with the same row order, and add them to `manifest.json`.

## FAISS usage sketch

```python
import faiss
import numpy as np

x = np.load("/path/to/feature_dump/matrices/curation_fused.npy", mmap_mode="r")
x = np.ascontiguousarray(x, dtype="float32")

index = faiss.IndexFlatIP(x.shape[1])
index.add(x)
scores, ids = index.search(x[:10], 20)
```

Use `IndexFlatIP` with L2-normalized matrices for cosine-like similarity.

`meta.jsonl` keeps `scene_id` by default when the input row has a `scene_id`
field. Use `--scene-id-key` if the raw dataset uses another key.

## FAISS clustering report

`cluster_features_faiss.py` clusters the dumped feature matrix and writes
analysis files for abnormal-class inspection. By default, the cluster count is
the number of unique lateral/longitudinal decision combinations in `meta.jsonl`.
It requires `faiss-cpu` or `faiss-gpu` in the active Python environment.

```shell
python3 scripts/vla/cluster_features_faiss.py \
  --input-dir /path/to/feature_dump \
  --output-dir /path/to/feature_dump_cluster_report \
  --matrix curation_fused
```

To run FAISS k-means on specific GPUs:

```shell
python3 scripts/vla/cluster_features_faiss.py \
  --input-dir /path/to/feature_dump \
  --output-dir /path/to/feature_dump_cluster_report \
  --matrix input_text_hash \
  --gpu-devices 0,1,2,3
```

`--gpu-devices` sets `CUDA_VISIBLE_DEVICES` before importing FAISS and then
passes the visible GPU count to FAISS k-means. This avoids FAISS silently using
only the first visible GPU when a physical device list is desired.

The output directory contains:

```text
feature_dump_cluster_report/
  cluster_assignments.jsonl
  cluster_summary.json
  cluster_report.md
  centroids.npy
```

- `cluster_report.md`: human-readable report with potential abnormal clusters,
  smallest clusters, lowest-closeness samples, largest clusters, and decision
  pair counts. It also includes every cluster's lateral/longitudinal decision
  pair distribution and per-cluster ratios.
- `cluster_summary.json`: structured cluster statistics for downstream analysis.
- `cluster_assignments.jsonl`: per-row `sample_id`, original `row_idx`,
  cluster id, lateral/longitudinal labels, and centroid closeness.
- `centroids.npy`: learned FAISS k-means centroids.

For real embeddings appended to `matrices/` and `manifest.json`, pass their
matrix name with `--matrix`. Use `--metric cosine` for L2-normalized embeddings
and `--metric l2` for raw Euclidean features.

## Cluster mining for sampling and review

`analyze_cluster_mining.py` enriches completed clustering results with scene
labels, builds sampling buckets, and exports human-review candidates. It does
not rerun FAISS.

The scene mapping is passed as JSON:

```json
{
  "9f94da7ee64f4bfab4e270864aec0fce": {
    "专题": "调平",
    "二级场景": "导航换道lane_remain调平"
  }
}
```

Run it on an existing feature dump and cluster report:

```shell
python3 scripts/vla/analyze_cluster_mining.py \
  --feature-dir /path/to/feature_dump \
  --cluster-dir /path/to/feature_dump_cluster_report \
  --scene-map /path/to/scene_map.json \
  --output-dir /path/to/mining_report
```

If the feature dump was created before `scene_id` was kept in `meta.jsonl`, pass
the original JSONL file or directory to recover it:

```shell
python3 scripts/vla/analyze_cluster_mining.py \
  --feature-dir /path/to/old_feature_dump \
  --cluster-dir /path/to/feature_dump_cluster_report \
  --scene-map /path/to/scene_map.json \
  --raw-input /path/to/train_jsonl_dir_or_file \
  --output-dir /path/to/mining_report
```

The output directory contains:

```text
mining_report/
  mining_summary.md
  mining_summary.json
  cluster_decision_distribution.csv
  cluster_primary_scene_distribution.csv
  cluster_secondary_scene_distribution.csv
  label_cluster_distribution.csv
  scene_decision_counts.csv
  sampling_buckets.csv
  conditional_cluster_jobs.jsonl
  review_candidates.jsonl
```

- `sampling_buckets.csv`: recommended sampling unit
  `(一级场景, decision label, cluster_id)`, with secondary-scene distribution,
  cluster purity, bucket ratios, suggested action, sample weight, and target
  count.
- `review_candidates.jsonl`: candidates for human review, including
  `low_cluster_closeness`, `minority_in_cluster`, and `rare_scene_decision`.
- `conditional_cluster_jobs.jsonl`: large `(一级场景, decision label)` buckets
  worth running label-conditioned clustering on next.

## Distributed dump

The script can reuse ms-swift's distributed environment helpers. Launch it with
`torchrun` and pass `--distributed`; each rank processes rows where
`row_idx % WORLD_SIZE == RANK` and writes an independent shard dump.

```shell
NPROC_PER_NODE=8 \
python3 -m torch.distributed.run \
  --nproc_per_node 8 \
  scripts/vla/dump_messages_features.py \
  --input /path/to/train.jsonl \
  --output-dir /path/to/feature_dump \
  --distributed \
  --max-traj-points 10
```

The output layout becomes:

```text
feature_dump/
  shards/
    rank_00000_of_00008/
      manifest.json
      meta.jsonl
      matrices/*.npy
    rank_00001_of_00008/
      ...
```

These shard dumps can later be concatenated by row order into one global dump.

## Legacy JSONL

For debugging or backward compatibility:

```shell
python3 scripts/vla/dump_messages_features.py \
  --input /path/to/train.jsonl \
  --output /path/to/features.jsonl \
  --keep-text
```
