# VLA Feature Dump

`dump_messages_features.py` extracts curation features from ms-swift messages-format VLA data.

## FAISS-friendly dump

```shell
python3 scripts/vla/dump_messages_features.py \
  --input /path/to/train.jsonl \
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
