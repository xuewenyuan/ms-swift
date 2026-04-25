#!/usr/bin/env python3
"""Dump last-layer model hidden-state features for VLA JSONL data."""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dump_messages_features import (
    build_feature,
    iter_jsonl,
    iter_jsonl_files,
    make_meta_row,
    resolve_input_files,
    select_input_feature_text,
    split_input_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract last-layer hidden-state features with swift model/template loading. "
            "Pass swift model arguments after this script's arguments, for example --model /path/to/model."))
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file or directory. Directories are searched recursively for *.jsonl.")
    parser.add_argument("--output-dir", required=True, help="FAISS-friendly output dump directory.")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Shard rows by global row_idx %% WORLD_SIZE == RANK and write one dump per rank.")
    parser.add_argument("--batch-size", type=int, default=1, help="Forward batch size per process. Default: 1.")
    parser.add_argument(
        "--message-scope",
        choices=["prompt", "full", "ego_state"],
        default="prompt",
        help=(
            "Messages used for the model forward. prompt removes assistant responses; full keeps all messages; "
            "ego_state keeps only the extracted ego-state text. Default: prompt."))
    parser.add_argument(
        "--pooling",
        choices=["mean", "last_non_pad"],
        default="mean",
        help="Pooling over the last hidden layer. Default: mean over non-padding tokens.")
    parser.add_argument(
        "--matrix-name",
        default="llm_last_hidden",
        help="Matrix name stored in manifest.json. Default: llm_last_hidden.")
    parser.add_argument(
        "--output-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Numpy dtype used on disk. Default: float16.")
    parser.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="Do not L2-normalize embeddings before writing. Default: normalize for cosine/IndexFlatIP.")
    parser.add_argument("--sample-id-key", default="sample_id", help="Sample id key in raw JSONL.")
    parser.add_argument("--scene-id-key", default="scene_id", help="Scene id key in raw JSONL.")
    parser.add_argument("--sample-token-key", default="sample_token", help="Sample token key in raw JSONL.")
    parser.add_argument("--timestamp-key", default="timestamp", help="Timestamp key in raw JSONL.")
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip rows that fail encoding/forward. This leaves fewer rows in the shard dump.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress every N written rows per process. Default: 100.")
    parser.add_argument(
        "swift_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to swift BaseArguments, for example: --model /path/to/model --template qwen2_5.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    rank, local_rank, world_size, _ = get_runtime_dist()
    if not args.distributed:
        rank, world_size = 0, 1

    setup_device(local_rank if args.distributed else None)
    model, template, swift_args = load_swift_model_template(args.swift_args)
    template.set_mode("train" if args.message_scope == "full" else "pt")
    model.eval()

    input_path = Path(args.input)
    input_files = resolve_input_files(input_path)
    total_rows = count_assigned_rows(input_files, rank, world_size)
    output_dir = distributed_output_dir(Path(args.output_dir), rank, world_size) if args.distributed else Path(
        args.output_dir)
    writer = HiddenStateDumpWriter(
        output_dir,
        total_rows=total_rows,
        matrix_name=args.matrix_name,
        output_dtype=np.dtype(args.output_dtype),
        l2_normalized=not args.no_l2_normalize,
    )

    pending: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    seen = 0
    written = 0
    try:
        for row_idx, row in enumerate(iter_jsonl_files(input_files)):
            seen += 1
            if row_idx % world_size != rank:
                continue
            try:
                request_row = prepare_request_row(row, args.message_scope)
                pending.append((row_idx, row, request_row))
                if len(pending) >= args.batch_size:
                    written += flush_batch_with_error_policy(pending, writer, model, template, feature_args(args), args)
                    pending.clear()
                    maybe_print_progress(written, total_rows, rank, args.progress_interval)
            except Exception as e:
                if not args.skip_errors:
                    raise
                print(f"[rank {rank}] warning: skipped row_idx={row_idx}: {e}")
        if pending:
            written += flush_batch_with_error_policy(pending, writer, model, template, feature_args(args), args)
            pending.clear()
            maybe_print_progress(written, total_rows, rank, args.progress_interval, force=True)
    finally:
        writer.close(
            input_path=str(input_path),
            input_files=[str(path) for path in input_files],
            swift_args=swift_args,
            model=str(getattr(model, "name_or_path", None) or getattr(model.config, "name_or_path", "") or ""),
            message_scope=args.message_scope,
            pooling=args.pooling,
            dist={"rank": rank, "world_size": world_size},
            expected_rows=total_rows,
            written_rows=written,
        )
    print(f"[rank {rank}/{world_size}] dumped {written}/{total_rows} hidden-state rows -> {writer.output_dir}")


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")
    if args.swift_args and args.swift_args[0] == "--":
        args.swift_args = args.swift_args[1:]
    if "--model" not in args.swift_args and not any(item.startswith("--model=") for item in args.swift_args):
        raise ValueError("Please pass swift model args after script args, at least: --model /path/to/model")


def get_runtime_dist() -> Tuple[int, int, int, int]:
    try:
        from swift.utils import get_dist_setting

        rank, local_rank, world_size, local_world_size = get_dist_setting()
    except ModuleNotFoundError:
        rank = int(os.getenv("RANK", "-1"))
        local_rank = int(os.getenv("LOCAL_RANK", "-1"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    rank = 0 if rank < 0 else rank
    local_rank = 0 if local_rank < 0 else local_rank
    return rank, local_rank, max(1, world_size), max(1, local_world_size)


def setup_device(local_rank: Optional[int]) -> None:
    try:
        from swift.utils import set_device

        set_device(local_rank)
    except Exception as e:
        print(f"warning: failed to set device for local_rank={local_rank}: {e}")


def load_swift_model_template(swift_args: Sequence[str]):
    from swift.llm import BaseArguments, prepare_model_template
    from swift.utils import parse_args

    parsed_args, remaining = parse_args(BaseArguments, list(swift_args))
    if remaining:
        raise ValueError(f"unrecognized swift args: {remaining}")
    model, template = prepare_model_template(parsed_args)
    return model, template, list(swift_args)


def distributed_output_dir(output_dir: Path, rank: int, world_size: int) -> Path:
    return output_dir / "shards" / f"rank_{rank:05d}_of_{world_size:05d}"


def count_assigned_rows(input_files: Sequence[Path], rank: int, world_size: int) -> int:
    count = 0
    row_idx = 0
    for path in input_files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if row_idx % world_size == rank:
                    count += 1
                row_idx += 1
    return count


def feature_args(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "sample_id_key": args.sample_id_key,
        "scene_id_key": args.scene_id_key,
        "sample_token_key": args.sample_token_key,
        "timestamp_key": args.timestamp_key,
    }


def flush_batch(
        batch: Sequence[Tuple[int, Dict[str, Any], Dict[str, Any]]],
        writer: "HiddenStateDumpWriter",
        model: Any,
        template: Any,
        meta_args: Dict[str, str],
        args: argparse.Namespace) -> int:
    features = [
        build_feature(
            raw_row,
            row_idx,
            meta_args["sample_id_key"],
            meta_args["scene_id_key"],
            meta_args["sample_token_key"],
            meta_args["timestamp_key"],
            tokenizer=None,
            keep_text=False,
            keep_token_ids=False,
            input_text_scope="ego_state") for row_idx, raw_row, _ in batch
    ]
    request_rows = [request_row for _, _, request_row in batch]
    embeddings = encode_hidden_embeddings(model, template, request_rows, args.pooling, not args.no_l2_normalize)
    writer.add(features, embeddings)
    return len(features)


def flush_batch_with_error_policy(
        batch: Sequence[Tuple[int, Dict[str, Any], Dict[str, Any]]],
        writer: "HiddenStateDumpWriter",
        model: Any,
        template: Any,
        meta_args: Dict[str, str],
        args: argparse.Namespace) -> int:
    try:
        return flush_batch(batch, writer, model, template, meta_args, args)
    except Exception:
        if not args.skip_errors or len(batch) == 1:
            raise
    written = 0
    for item in batch:
        try:
            written += flush_batch([item], writer, model, template, meta_args, args)
        except Exception as e:
            print(f"[rank warning] skipped row_idx={item[0]} after batch failure: {e}")
    return written


def prepare_request_row(row: Dict[str, Any], message_scope: str) -> Dict[str, Any]:
    item = deepcopy(row)
    messages = item.get("messages")
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list) or not messages:
        raise ValueError("missing non-empty messages")

    if message_scope == "full":
        item["messages"] = messages
        return item

    input_messages, _ = split_input_output(messages)
    if message_scope == "prompt":
        item["messages"] = input_messages
        return item

    full_input_text = "\n".join(str(message.get("content", "")) for message in input_messages)
    ego_text, found = select_input_feature_text(full_input_text, "ego_state")
    if not found:
        print("warning: ego_state marker not found; falling back to full prompt text for one row")
    system_messages = [message for message in input_messages if message.get("role") == "system"]
    item["messages"] = system_messages + [{"role": "user", "content": ego_text}]
    return item


def encode_hidden_embeddings(
        model: Any,
        template: Any,
        request_rows: Sequence[Dict[str, Any]],
        pooling: str,
        normalize: bool) -> np.ndarray:
    import torch
    from swift.llm import to_device

    encoded_rows = [template.encode(row, return_template_inputs=False) for row in request_rows]
    inputs = template.data_collator(encoded_rows)
    inputs.pop("labels", None)
    inputs.pop("loss_scale", None)
    inputs.pop("channel", None)
    device = get_model_device(model)
    inputs = to_device(inputs, device)
    if getattr(getattr(model, "model_meta", None), "is_multimodal", False):
        _, inputs = template.pre_forward_hook(model, None, inputs)
    forward_inputs = dict(inputs)
    with torch.inference_mode():
        outputs = forward_with_hidden_states(model, forward_inputs)
    hidden = extract_last_hidden(outputs)
    attention_mask = inputs.get("attention_mask")
    pooled = pool_hidden(hidden, attention_mask, pooling)
    matrix = pooled.detach().float().cpu().numpy()
    if normalize:
        matrix = l2_normalize_matrix(matrix)
    return matrix


def get_model_device(model: Any):
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def forward_with_hidden_states(model: Any, inputs: Dict[str, Any]) -> Any:
    kwargs = dict(inputs)
    kwargs.update({"output_hidden_states": True, "return_dict": True, "use_cache": False})
    try:
        return model(**kwargs)
    except TypeError:
        kwargs.pop("use_cache", None)
        try:
            return model(**kwargs)
        except TypeError:
            kwargs.pop("output_hidden_states", None)
            old_value = getattr(model.config, "output_hidden_states", None)
            model.config.output_hidden_states = True
            try:
                return model(**kwargs)
            finally:
                if old_value is not None:
                    model.config.output_hidden_states = old_value


def extract_last_hidden(outputs: Any):
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and isinstance(outputs, dict):
        hidden_states = outputs.get("hidden_states")
    if hidden_states is not None:
        return hidden_states[-1]
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is None and isinstance(outputs, dict):
        last_hidden = outputs.get("last_hidden_state")
    if last_hidden is None:
        raise RuntimeError("model output does not contain hidden_states or last_hidden_state")
    return last_hidden


def pool_hidden(hidden: Any, attention_mask: Optional[Any], pooling: str):
    import torch

    if attention_mask is None or attention_mask.shape[:2] != hidden.shape[:2]:
        if pooling == "last_non_pad":
            return hidden[:, -1, :]
        return hidden.mean(dim=1)

    mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
    if pooling == "last_non_pad":
        rows = []
        for i in range(hidden.shape[0]):
            valid = torch.nonzero(mask[i], as_tuple=False).reshape(-1)
            idx = int(valid[-1]) if valid.numel() else hidden.shape[1] - 1
            rows.append(hidden[i, idx, :])
        return torch.stack(rows, dim=0)

    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denom


def l2_normalize_matrix(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, eps)).astype(np.float32)


def maybe_print_progress(written: int, total: int, rank: int, interval: int, force: bool = False) -> None:
    if force or written % interval == 0:
        print(f"[rank {rank}] wrote {written}/{total} rows")


class HiddenStateDumpWriter:

    def __init__(
            self,
            output_dir: Path,
            *,
            total_rows: int,
            matrix_name: str,
            output_dtype: np.dtype,
            l2_normalized: bool) -> None:
        self.output_dir = output_dir
        self.matrix_dir = output_dir / "matrices"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.matrix_dir.mkdir(parents=True, exist_ok=True)
        self.total_rows = total_rows
        self.matrix_name = matrix_name
        self.output_dtype = output_dtype
        self.l2_normalized = l2_normalized
        self.meta_f = (output_dir / "meta.jsonl").open("w", encoding="utf-8")
        self.index_f = (output_dir / "index.jsonl").open("w", encoding="utf-8")
        self.sample_id_f = (output_dir / "sample_ids.txt").open("w", encoding="utf-8")
        self.matrix = None
        self.dim = None
        self.count = 0
        self.matrix_path = self.matrix_dir / f"{self.matrix_name}.npy"

    def add(self, features: Sequence[Dict[str, Any]], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape={embeddings.shape}")
        if len(features) != embeddings.shape[0]:
            raise ValueError(f"features ({len(features)}) != embeddings ({embeddings.shape[0]})")
        self._ensure_matrix(embeddings.shape[1])
        end = self.count + embeddings.shape[0]
        if end > self.total_rows:
            raise ValueError(f"writer overflow: writing {end} rows into total_rows={self.total_rows}")
        self.matrix[self.count:end] = embeddings.astype(self.output_dtype, copy=False)
        for feature in features:
            row = self.count
            sample_id = str(feature["sample_id"])
            self.meta_f.write(json.dumps(make_meta_row(feature), ensure_ascii=False, separators=(",", ":")) + "\n")
            self.index_f.write(json.dumps({"row": row, "sample_id": sample_id}, ensure_ascii=False) + "\n")
            self.sample_id_f.write(sample_id + "\n")
            self.count += 1

    def _ensure_matrix(self, dim: int) -> None:
        if self.matrix is not None:
            if self.dim != dim:
                raise ValueError(f"embedding dim changed from {self.dim} to {dim}")
            return
        self.dim = dim
        self.matrix = np.lib.format.open_memmap(
            self.matrix_path,
            mode="w+",
            dtype=self.output_dtype,
            shape=(self.total_rows, dim),
        )

    def close(
            self,
            *,
            input_path: str,
            input_files: Sequence[str],
            swift_args: Sequence[str],
            model: str,
            message_scope: str,
            pooling: str,
            dist: Dict[str, int],
            expected_rows: int,
            written_rows: int) -> None:
        self.meta_f.close()
        self.index_f.close()
        self.sample_id_f.close()
        if self.matrix is not None:
            self.matrix.flush()
            if self.count != self.total_rows:
                trimmed = np.asarray(self.matrix[:self.count])
                del self.matrix
                np.save(self.matrix_path, trimmed.astype(self.output_dtype, copy=False))
        matrix_specs = {}
        if self.dim is not None:
            matrix_specs[self.matrix_name] = {
                "path": f"matrices/{self.matrix_name}.npy",
                "shape": [self.count, self.dim],
                "dtype": str(self.output_dtype),
                "faiss_metric": "inner_product" if self.l2_normalized else "l2",
                "l2_normalized": self.l2_normalized,
                "source": "swift_model_last_hidden_state",
                "pooling": pooling,
                "message_scope": message_scope,
            }
        manifest = {
            "format": "vla-faiss-feature-dump",
            "version": 2,
            "input_path": input_path,
            "input_files": list(input_files),
            "model": model,
            "swift_args": list(swift_args),
            "distributed": dist,
            "num_rows": self.count,
            "expected_rows": expected_rows,
            "written_rows": written_rows,
            "row_alignment": "All matrices use the same row order as index.jsonl and sample_ids.txt.",
            "files": {
                "meta": "meta.jsonl",
                "index": "index.jsonl",
                "sample_ids": "sample_ids.txt",
            },
            "matrices": matrix_specs,
            "notes": [
                "This dump stores pooled last-layer hidden states produced through swift model/template loading.",
                "Use --message-scope prompt to avoid assistant-label leakage when clustering by input semantics.",
            ],
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
