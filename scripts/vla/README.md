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
  --sample-token-key sample_token \
  --timestamp-key timestamp \
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

## CoC template analysis

`analyze_coc_templates.py` evaluates Chain-of-Causation / COT text wrapped by
`<think>...</think>` in assistant messages. It is independent from feature dump
and clustering reports, and can be run directly on a JSONL file or directory.

```shell
python3 scripts/vla/analyze_coc_templates.py \
  --input /path/to/coc_jsonl_dir_or_file \
  --output-dir /path/to/coc_template_report \
  --scene-map /path/to/scene_map.json
```

The report includes exact and normalized duplicate coverage, top normalized
templates, per-sample template scores, CoC length bins, and high-score review
candidates:

```text
coc_template_report/
  coc_summary.json
  coc_summary.md
  coc_sample_scores.csv
  coc_top_normalized_templates.csv
  coc_top_exact_templates.csv
  coc_length_bins.csv
  coc_review_candidates.jsonl
```

Normalization replaces dates, times, numbers, units, URLs, and special tokens
before duplicate counting, so repeated templates with different distances or
speeds can still be detected.

## Swift last-hidden-state dump

`dump_hidden_state_features.py` uses swift model/template loading to run the
model forward pass and dump pooled last-layer hidden states as a FAISS-friendly
matrix. It keeps the same `manifest.json`, `meta.jsonl`, `index.jsonl`, and
`sample_ids.txt` layout as `dump_messages_features.py`, so downstream clustering
can consume it directly.

```shell
python3 scripts/vla/dump_hidden_state_features.py \
  --input /path/to/train_jsonl_dir_or_file \
  --output-dir /path/to/hidden_feature_dump \
  --message-scope prompt \
  --pooling mean \
  --batch-size 1 \
  --model /path/to/model \
  --template qwen2_5_vl
```

The default `--message-scope prompt` removes assistant responses before encoding,
which avoids leaking the lateral/longitudinal labels into the clustering
features. Use `--message-scope full` only when you explicitly want the assistant
decision tokens included. The output matrix name is `llm_last_hidden` by
default:

```shell
python3 scripts/vla/adaptive_cluster_mining.py \
  --feature-dir /path/to/hidden_feature_dump \
  --output-dir /path/to/adaptive_report_hidden \
  --matrix llm_last_hidden \
  --scene-map /path/to/scene_map.json
```

For cloud distributed runs, launch one process per GPU. Each process writes a
rank shard under `output-dir/shards/`, and the existing clustering loader will
merge shards by original `row_idx`:

```shell
NPROC_PER_NODE=8 \
python3 -m torch.distributed.run \
  --nproc_per_node 8 \
  scripts/vla/dump_hidden_state_features.py \
  --input /path/to/train_jsonl_dir_or_file \
  --output-dir /path/to/hidden_feature_dump \
  --distributed \
  --batch-size 1 \
  --model /path/to/model \
  --template qwen2_5_vl
```

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

`meta.jsonl` keeps `scene_id`, `sample_token`, and `timestamp` by default when
the input row has those fields. Use `--scene-id-key`, `--sample-token-key`, and
`--timestamp-key` if the raw dataset uses other keys.

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
passes the visible GPU count to FAISS k-means when GPU training is selected.
By default, `--kmeans-device auto` uses CPU k-means if the sampled training set
is small, because FAISS GPU can hit CUBLAS failures on tiny GEMMs such as a
35-centroid, 256-dim clustering run. Use `--kmeans-device gpu` to force GPU
k-means, or lower `--gpu-min-train-points` if you want auto mode to use GPU
more aggressively.

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

## Adaptive clustering for sampling diversity

`adaptive_cluster_mining.py` recursively splits only the leaves that are not yet
useful for data mining: low-purity leaves are split to improve label separation,
and high-purity but oversized leaves are split to improve sampling diversity.
Leaves that are already pure enough and not too large stop early.

```shell
python3 scripts/vla/adaptive_cluster_mining.py \
  --feature-dir /path/to/feature_dump \
  --output-dir /path/to/adaptive_report \
  --matrix input_text_hash \
  --scene-map /path/to/scene_map.json \
  --purity-threshold 0.7 \
  --min-leaf-size 500 \
  --max-leaf-size 50000 \
  --max-depth 3
```

This workflow is intended for "sampling balance and diversity first, suspected
label-error mining second". It outputs:

```text
adaptive_report/
  adaptive_summary.md
  adaptive_summary.json
  adaptive_tree.json
  adaptive_leaf_summary.csv
  adaptive_leaf_decision_distribution.csv
  adaptive_leaf_scene_distribution.csv
  adaptive_leaf_assignments.jsonl
  adaptive_review_candidates.jsonl
```

- `adaptive_leaf_summary.csv`: final sampling leaves with purity, dominant
  decision, top scenes, stop reason, suggested action, and suggested target
  count.
- `adaptive_leaf_assignments.jsonl`: per-row leaf id and source metadata for
  building train manifests.
- `adaptive_review_candidates.jsonl`: suspected label issues from high-purity
  leaf minorities and leaves that remain impure after recursive splitting.

## Assign eval data to adaptive leaves

`assign_eval_to_adaptive_leaf.py` maps an eval feature dump onto train adaptive
leaves by nearest-neighbor search in the same feature space. Edit the config
area at the top of the script or pass the equivalent CLI arguments:

```shell
python3 scripts/vla/assign_eval_to_adaptive_leaf.py \
  --train-feature-dir /path/to/train_feature_dump \
  --eval-feature-dir /path/to/eval_feature_dump \
  --adaptive-dir /path/to/adaptive_report \
  --output /path/to/eval_leaf_assignments.jsonl \
  --matrix input_text_hash \
  --metric cosine
```

The script writes per-eval-row assignments and a leaf-level summary:

```text
eval_leaf_assignments.jsonl
eval_leaf_assignments.leaf_summary.csv
eval_leaf_assignments.summary.json
```

Use the same matrix and metric as adaptive clustering, otherwise the nearest
train leaf will not be comparable with the train bucket loss.

## Split raw JSONL by adaptive leaves

`split_raw_by_adaptive_leaf.py` maps `adaptive_leaf_assignments.jsonl` back to
the original JSONL row order and writes one JSONL per adaptive group. The default
group is one file per leaf, which makes each file a diversity/sampling unit.
For leaf grouping, output filenames use
`<prefix>_leaf_<leaf_id>_<purity>_<dominant_label>.jsonl`.

```shell
python3 scripts/vla/split_raw_by_adaptive_leaf.py \
  --raw-input /path/to/train_jsonl_dir_or_file \
  --adaptive-dir /path/to/adaptive_report \
  --output-dir /path/to/adaptive_leaf_jsonl
```

The output directory contains split JSONL files plus `split_manifest.csv` with
each file's row count, proportional `suggested_target_count`, dominant decision,
top scenes, average leaf purity, and suggested action. Use `--group-by` to
coarsen or refine the split, for example `primary_scene_leaf`, `label_leaf`,
`primary_scene_label`, or `suggested_action`. By default the raw JSON rows are
kept unchanged; use `--add-mining-meta` to attach an `_adaptive_mining` field.
Use `--add-channel` if you want ms-swift to log per-leaf channel loss with
`--enable_channel_loss true`.

`build_adaptive_dataset_info.py` converts `split_manifest.csv` into an
ms-swift custom dataset info JSON. Each leaf file becomes one `dataset_name`, and
leaf metadata is recorded in the supported `tags` and `help` fields.
Generated dataset names include the leaf id, purity token, and dominant
decision, for example `vla_ego_leaf_000012_p0950_LAT_LANE_KEEP_LON_MAINTAIN`.

```shell
python3 scripts/vla/build_adaptive_dataset_info.py \
  --split-manifest /path/to/adaptive_leaf_jsonl/split_manifest.csv \
  --output /path/to/adaptive_leaf_dataset_info.json \
  --dataset-name-prefix vla_ego \
  --min-purity 0.7 \
  --purity-ratio-threshold 0.9 \
  --purity-sample-ratio 0.2 \
  --non-purity-target-mode suggested_target_count
```

Use the generated JSON with `--custom_dataset_info`. The companion
`*.dataset_args.txt` contains recommended dataset arguments; with
`--downsample-high-purity`, entries whose target is smaller than their raw count
are written as `dataset_name#target_count`.
If `--purity-ratio-threshold` and `--purity-sample-ratio` are set, rows whose
`avg_leaf_purity` is greater than the threshold use `count * ratio` as their
target count, and are written with `#target_count` automatically. Rows that do
not match this rule use `--non-purity-target-mode`: `suggested_target_count`
applies the manifest target column, while `full` keeps all rows without
sampling.
The companion `*.dataset_config.yaml` can be pasted into a `swift --config`
YAML file so hundreds of leaf datasets do not need to be typed on the command
line.

`sample_merge_adaptive_datasets.py` applies `dataset_args.txt` offline, samples
each leaf JSONL, and merges the selected rows into one or more shard JSONL
files. This avoids training startup overhead from registering hundreds of leaf
datasets while preserving the per-row `channel` field for channel loss logging.

```shell
python3 scripts/vla/sample_merge_adaptive_datasets.py \
  --dataset-args /path/to/adaptive_leaf_dataset_info.json.dataset_args.txt \
  --dataset-info /path/to/adaptive_leaf_dataset_info.json \
  --output-dir /path/to/balanced_train \
  --num-shards 16 \
  --shuffle-datasets
```

The output directory contains `merged_00000.jsonl`, `merged_00001.jsonl`, ...
and `sampling_report.csv`. Train with the merged directory directly:

```shell
swift sft --dataset /path/to/balanced_train --enable_channel_loss true
```

`sample_adaptive_experiments.py` builds experiment datasets directly from
adaptive leaf JSONL files and `split_manifest.csv`. It can produce the sampling
plans discussed for VLA decision training:

- `A_high_purity_dedup`: purity >= 0.9 keeps 30%, 0.7 <= purity < 0.9 keeps
  50%, lower-purity leaves keep all rows.
- `B_remove_roundabout_dirty`: applies A, then removes low-purity leaves that
  are dominated by roundabout prompts or review-first leaves whose secondary
  scene matches roundabout keywords.
- `C_balanced_decision`: applies roundabout dirty removal, down-samples
  high-purity/high-frequency labels, keeps or lightly up-samples low-purity
  non-roundabout leaves, and up-samples low-frequency decision labels.
- `D_aggressive_purity_label_sampling`: applies the same roundabout dirty
  removal as B, then samples by decision label inside each leaf. For
  purity >= 0.7, the dominant label keeps 10% while non-dominant labels are kept. For
  0.4 <= purity < 0.7, labels with ratio p >= 10% keep 0.1 / p, labels below
  100 rows are up-sampled to 100, and lower-ratio labels are kept. Purity < 0.4
  leaves are kept unchanged.

```shell
python3 scripts/vla/sample_adaptive_experiments.py \
  --split-manifest /path/to/adaptive_leaf_jsonl/split_manifest.csv \
  --output-dir /path/to/adaptive_sampling_experiments \
  --strategies A,B,C \
  --num-shards 16 \
  --write-holdout
```

Each strategy directory contains sampled JSONL shards, `sampling_plan.csv`, and
`sampling_summary.json`. The holdout shards keep removed dirty roundabout leaves
for review or future roundabout-specific training.

Run the aggressive label-aware plan with:

```shell
python3 scripts/vla/sample_adaptive_experiments.py \
  --split-manifest /path/to/adaptive_leaf_jsonl/split_manifest.csv \
  --output-dir /path/to/adaptive_sampling_experiments \
  --strategies D \
  --num-shards 16 \
  --write-holdout
```

The D strategy additionally writes `label_sampling_plan.csv`, which records the
target count and reason for every `(leaf, decision label)` group. Dirty
roundabout leaves are always removed from sampled shards; `--write-holdout`
only controls whether those removed leaves are also written to holdout shards.

`visualize_adaptive_purity.py` renders a static HTML dashboard for
`adaptive_leaf_summary.csv`, including purity-bin leaf counts, row-volume
distribution, suggested target volume, and a purity-vs-size scatter plot.

```shell
python3 scripts/vla/visualize_adaptive_purity.py \
  --adaptive-dir /path/to/adaptive_report \
  --output /path/to/adaptive_purity_dashboard.html
```

`compare_adaptive_reports.py` compares two adaptive report directories and
summarizes whether relabeled data became cleaner overall, by decision label, and
by primary scene.

```shell
python3 scripts/vla/compare_adaptive_reports.py \
  --old-report /path/to/adaptive_report_old \
  --new-report /path/to/adaptive_report_new \
  --output-dir /path/to/adaptive_compare
```

The output directory contains:

```text
adaptive_compare/
  compare_overview.csv
  compare_label_metrics.csv
  compare_primary_scene_metrics.csv
  compare_summary.json
  compare_summary.md
```

`analyze_channel_loss_purity.py` joins training channel-loss curves with
adaptive leaf purity to check whether convergence is correlated with cluster
purity. It can read TensorBoard event files directly, or a scalar CSV exported
from TensorBoard.

```shell
python3 scripts/vla/analyze_channel_loss_purity.py \
  --tb-dir /path/to/train_output/runs \
  --split-manifest /path/to/adaptive_leaf_jsonl/split_manifest.csv \
  --channel-prefix vla_ego \
  --output-dir /path/to/channel_loss_purity
```

The output directory contains per-channel convergence metrics, purity/loss
correlations, purity-bin trends, unmatched TensorBoard tags, a Markdown summary,
and a static HTML dashboard. For hundreds of leaves, the purity-bin trend charts
are usually easier to read than the raw scatter plots.

```text
channel_loss_purity/
  channel_loss_metrics.csv
  channel_loss_correlations.csv
  channel_loss_purity_bins.csv
  channel_loss_unmatched.csv
  channel_loss_purity_summary.md
  channel_loss_purity_dashboard.html
```

`bucket_diagnosis.py` joins one or more sampling experiments with TensorBoard
channel loss, per-checkpoint eval prediction JSONL files, and eval-to-leaf
assignments. `tb_dir` / `loss_csv` are optional; if they are omitted, the script
still produces eval-only bucket diagnosis and the HTML dashboard. It computes
lateral/longitudinal decision correctness from each row's `response` and `labels`
fields with logic aligned to
`examples/train/plugins/lateral_decision_metric_plugin.py`, then aggregates the
metrics by adaptive leaf. It produces recommendations such as `downsample_more`,
`undertrained_upsample_or_replay`, `suspected_noisy_review`, and
`hard_but_valid_curriculum`.
Edit the config area at the top of the script or pass a JSON experiment list:

```json
[
  {
    "name": "sample_d",
    "tb_dir": "/path/to/train_output/runs",
    "eval_json_glob": [
      "/path/to/eval/ckpt-500/predictions.jsonl",
      "/path/to/eval/ckpt-1000/predictions.jsonl"
    ],
    "eval_leaf_assignments": "/path/to/eval_leaf_assignments.jsonl",
    "split_manifest": "/path/to/adaptive_leaf_jsonl/split_manifest.csv",
    "sampling_plan": "/path/to/D_aggressive_purity_label_sampling/sampling_plan.csv",
    "label_sampling_plan": "/path/to/D_aggressive_purity_label_sampling/label_sampling_plan.csv"
  }
]
```

```shell
python3 scripts/vla/bucket_diagnosis.py \
  --experiments-json /path/to/bucket_experiments.json \
  --output-dir /path/to/bucket_diagnosis \
  --primary-eval-metric decision_acc \
  --threshold-mode auto
```

Each eval JSONL row should contain `id`, `response`, and `labels`. The `id`
should be `scene_id_sample_token`; it is matched to `eval_leaf_assignments.jsonl`
through `eval_scene_id` + `_` + `eval_sample_token`. The script also keeps a
fallback JSON parser for common `records` / `samples` / `predictions` structures.
`eval_json_glob` can be a single glob string or a list of paths/globs; checkpoint
steps are inferred from the full path, so paths such as `ckpt-500/predictions.jsonl`
are supported. If your eval output has a custom structure, adapt
`read_eval_records()` and `record_assignment_key()`.

Loss smoothing and trend parameters:

- `MIN_LOSS_POINTS`: minimum scalar points required before a leaf loss curve is
  diagnosed.
- `WINDOW_POINTS`: number of points averaged at the start/end of a curve to
  reduce noise in `loss_first` and `loss_final`.
- `TAIL_FRACTION`: final fraction of the curve used to judge whether loss is
  still descending, flat, or volatile.

Diagnosis thresholds can be fixed with `--threshold-mode manual` or derived from
the current experiment with `--threshold-mode auto`. Auto mode uses bucket
distribution quantiles: low final-loss buckets become the easy-loss group, high
final-loss buckets become hard/noisy candidates, eval high/low thresholds come
from final eval metric quantiles, plateau slope comes from the distribution of
absolute tail slopes, and high volatility comes from the tail-loss std
distribution.

```text
bucket_diagnosis/
  bucket_loss_metrics.csv
  bucket_loss_timeseries.csv
  bucket_eval_timeseries.csv
  bucket_diagnosis.csv
  bucket_eval_unmatched.csv
  bucket_diagnosis_action_summary.csv
  bucket_diagnosis_summary.json
  bucket_diagnosis_summary.md
  bucket_diagnosis_dashboard.html
  bucket_diagnosis_dashboard_data/
```

The dashboard keeps large per-leaf eval/loss time series in
`bucket_diagnosis_dashboard_data/` and loads them only when a leaf is selected.
Because browsers often block `fetch()` from `file://` pages, serve the output
directory over HTTP for the interactive leaf trend chart:

```shell
cd /path/to/bucket_diagnosis
python3 -m http.server 8000
```

## Sampling raw JSONL from bucket targets

`sample_from_buckets.py` applies `sampling_buckets.csv` targets back to the
original JSONL data. It reconstructs each row's mining bucket
`(一级场景, decision label, cluster_id)` from `meta.jsonl`,
`cluster_assignments.jsonl`, and the scene map, then samples up to
`suggested_target_count` rows per bucket.

```shell
python3 scripts/vla/sample_from_buckets.py \
  --raw-input /path/to/train_jsonl_dir_or_file \
  --feature-dir /path/to/feature_dump \
  --cluster-dir /path/to/feature_dump_cluster_report \
  --scene-map /path/to/scene_map.json \
  --sampling-buckets /path/to/mining_report/sampling_buckets.csv \
  --output /path/to/balanced_train.jsonl
```

By default sampling is without replacement, so targets larger than the available
bucket size are capped. Use `--sampling-mode with_replacement` if you want true
upsampling. A per-bucket report is written next to the output JSONL unless
`--report` is set.

## Mining dashboard

`visualize_cluster_mining.py` turns the mining report directory into a static
HTML dashboard. The Python script uses only the standard library; the generated
HTML loads Plotly.js in the browser.

```shell
python3 scripts/vla/visualize_cluster_mining.py \
  --mining-dir /path/to/mining_report \
  --output /path/to/mining_dashboard.html
```

The dashboard includes:

- overview cards and top-level distributions;
- sampling treemap and raw-vs-target sampling charts;
- cluster size/purity and label-dispersion charts;
- primary-scene x decision-label heatmap;
- review-candidate scatter plot and table;
- conditional clustering job table.

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
