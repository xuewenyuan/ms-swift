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
