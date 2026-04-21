#!/usr/bin/env python3
"""Dump lightweight curation features from ms-swift messages-format VLA data.

This script intentionally keeps the first stage cheap and dependency-light:
it does not run a VLM forward pass. It extracts stable input/output signatures,
media references, optional tokenizer ids, and structured VLA labels from the
assistant special-token response. Visual embeddings can be appended later using
the same sample_id/media fields.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


MEDIA_TAG_RE = re.compile(r"<(image|video|audio)>")
SPECIAL_TOKEN_RE = re.compile(r"<[^<>\s]+>")
LOC_TOKEN_RE = re.compile(r"<LOC_(\d+)>")
POINT_TOKEN_RE = re.compile(r"<P_(\d+)>")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract curation features from messages-format VLA JSONL data.")
    parser.add_argument("--input", required=True, help="Input JSONL in ms-swift messages format.")
    parser.add_argument("--output", default=None, help="Legacy output feature JSONL.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="FAISS-friendly dump dir. Writes meta/index JSONL, manifest JSON, and matrices/*.npy.")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Shard rows by global row_idx %% WORLD_SIZE == RANK and write one dump per rank.")
    parser.add_argument("--model", default=None, help="Optional model id/path for tokenizer ids only.")
    parser.add_argument("--sample-id-key", default="sample_id", help="Preferred sample id key.")
    parser.add_argument("--text-hash-dim", type=int, default=256, help="Dimension for hashed input text vector.")
    parser.add_argument("--decision-hash-dim", type=int, default=64, help="Dimension for hashed decision token vector.")
    parser.add_argument("--max-traj-points", type=int, default=10, help="Max trajectory points kept in dense arrays.")
    parser.add_argument(
        "--keep-text",
        action="store_true",
        help="Keep raw input/output text in feature JSONL. By default only hashes and lengths are saved.")
    parser.add_argument(
        "--keep-token-ids",
        action="store_true",
        help="Keep tokenizer ids if --model is provided. By default only token counts are saved.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output is None and args.output_dir is None:
        raise ValueError("Please set at least one of --output or --output-dir.")
    rank, world_size = get_runtime_rank_world(args.distributed)
    tokenizer = load_tokenizer(args.model) if args.model else None
    rows = iter_jsonl(Path(args.input))
    legacy_f = None
    if args.output:
        out_path = rank_path(Path(args.output), rank, world_size) if args.distributed else Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_f = out_path.open("w", encoding="utf-8")

    writer = FaissDumpWriter(
        distributed_output_dir(Path(args.output_dir), rank, world_size) if args.distributed else Path(args.output_dir),
        text_hash_dim=args.text_hash_dim,
        decision_hash_dim=args.decision_hash_dim,
        max_traj_points=args.max_traj_points,
    ) if args.output_dir else None

    count = 0
    seen = 0
    try:
        for row_idx, row in enumerate(rows):
            seen += 1
            if args.distributed and row_idx % world_size != rank:
                continue
            feature = build_feature(row, row_idx, args.sample_id_key, tokenizer, args.keep_text, args.keep_token_ids)
            if legacy_f is not None:
                legacy_f.write(json.dumps(feature, ensure_ascii=False, separators=(",", ":")) + "\n")
            if writer is not None:
                writer.add(feature)
            count += 1
    finally:
        if legacy_f is not None:
            legacy_f.close()
    if writer is not None:
        writer.close(input_path=str(Path(args.input)), model=args.model, dist={"rank": rank, "world_size": world_size})
        print(f"[rank {rank}/{world_size}] dumped {count}/{seen} feature rows -> {writer.output_dir}")
    if args.output:
        print(f"[rank {rank}/{world_size}] dumped {count}/{seen} feature rows -> {legacy_f.name if legacy_f else args.output}")


def get_runtime_rank_world(distributed: bool) -> Tuple[int, int]:
    if not distributed:
        return 0, 1
    try:
        from swift.utils import get_dist_setting

        rank, _, world_size, _ = get_dist_setting()
    except ModuleNotFoundError:
        rank = int(os.getenv("RANK", "-1"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
    if rank < 0:
        rank = 0
    world_size = max(1, world_size)
    return rank, world_size


def distributed_output_dir(output_dir: Path, rank: int, world_size: int) -> Path:
    return output_dir / "shards" / f"rank_{rank:05d}_of_{world_size:05d}"


def rank_path(path: Path, rank: int, world_size: int) -> Path:
    return path.with_name(f"{path.stem}.rank_{rank:05d}_of_{world_size:05d}{path.suffix}")


class FaissDumpWriter:

    def __init__(self, output_dir: Path, *, text_hash_dim: int, decision_hash_dim: int, max_traj_points: int):
        self.output_dir = output_dir
        self.matrix_dir = output_dir / "matrices"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.matrix_dir.mkdir(parents=True, exist_ok=True)
        self.text_hash_dim = text_hash_dim
        self.decision_hash_dim = decision_hash_dim
        self.max_traj_points = max_traj_points
        self.meta_f = (output_dir / "meta.jsonl").open("w", encoding="utf-8")
        self.index_f = (output_dir / "index.jsonl").open("w", encoding="utf-8")
        self.sample_id_f = (output_dir / "sample_ids.txt").open("w", encoding="utf-8")
        self.input_text_hash: List[np.ndarray] = []
        self.output_decision_hash: List[np.ndarray] = []
        self.output_traj: List[np.ndarray] = []
        self.output_traj_mask: List[np.ndarray] = []
        self.curation_fused: List[np.ndarray] = []
        self.count = 0

    def add(self, feature: Dict[str, Any]) -> None:
        row = self.count
        sample_id = str(feature["sample_id"])
        meta = make_meta_row(feature)
        self.meta_f.write(json.dumps(meta, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.index_f.write(json.dumps({"row": row, "sample_id": sample_id}, ensure_ascii=False) + "\n")
        self.sample_id_f.write(sample_id + "\n")

        text_vec = hashed_text_vector(feature["input"]["text_ngram_hashes"], self.text_hash_dim)
        decision_vec = decision_hash_vector(feature["output"]["vla"], self.decision_hash_dim)
        traj_vec, traj_mask = trajectory_vector(feature["output"]["vla"], self.max_traj_points)
        fused_vec = l2_normalize(np.concatenate([text_vec * 0.5, decision_vec * 0.2, traj_vec * 0.3, traj_mask * 0.1]))

        self.input_text_hash.append(text_vec)
        self.output_decision_hash.append(decision_vec)
        self.output_traj.append(traj_vec)
        self.output_traj_mask.append(traj_mask)
        self.curation_fused.append(fused_vec)
        self.count += 1

    def close(self, *, input_path: str, model: Optional[str], dist: Optional[Dict[str, int]] = None) -> None:
        self.meta_f.close()
        self.index_f.close()
        self.sample_id_f.close()

        matrices = {
            "input_text_hash": stack_or_empty(self.input_text_hash, self.text_hash_dim),
            "output_decision_hash": stack_or_empty(self.output_decision_hash, self.decision_hash_dim),
            "output_traj": stack_or_empty(self.output_traj, self.max_traj_points * 2),
            "output_traj_mask": stack_or_empty(self.output_traj_mask, self.max_traj_points),
            "curation_fused": stack_or_empty(
                self.curation_fused,
                self.text_hash_dim + self.decision_hash_dim + self.max_traj_points * 3,
            ),
        }
        matrix_specs = {}
        for name, matrix in matrices.items():
            path = self.matrix_dir / f"{name}.npy"
            np.save(path, np.ascontiguousarray(matrix.astype(np.float32)))
            matrix_specs[name] = {
                "path": str(path.relative_to(self.output_dir)),
                "shape": list(matrix.shape),
                "dtype": "float32",
                "faiss_metric": "inner_product",
                "l2_normalized": bool(name in {"input_text_hash", "output_decision_hash", "curation_fused"}),
            }

        manifest = {
            "format": "vla-faiss-feature-dump",
            "version": 1,
            "input_path": input_path,
            "model": model,
            "distributed": dist or {"rank": 0, "world_size": 1},
            "num_rows": self.count,
            "row_alignment": "All matrices use the same row order as index.jsonl and sample_ids.txt.",
            "files": {
                "meta": "meta.jsonl",
                "index": "index.jsonl",
                "sample_ids": "sample_ids.txt",
            },
            "matrices": matrix_specs,
            "notes": [
                "input_text_hash/output_decision_hash/curation_fused are lightweight curation vectors, not model embeddings.",
                "Append real visual/text/fused embeddings as additional N x D float32 .npy files with the same row order.",
                "For FAISS IndexFlatIP, use l2-normalized matrices for cosine-like similarity.",
            ],
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_meta_row(feature: Dict[str, Any]) -> Dict[str, Any]:
    vla = feature["output"]["vla"]
    return {
        "sample_id": feature["sample_id"],
        "row_idx": feature["row_idx"],
        "channel": feature.get("channel"),
        "input": {
            "sha1": feature["input"]["sha1"],
            "char_len": feature["input"]["char_len"],
            "token_len": feature["input"]["token_len"],
            "media_tags": feature["input"]["media_tags"],
        },
        "media": feature["media"],
        "output": {
            "sha1": feature["output"]["sha1"],
            "char_len": feature["output"]["char_len"],
            "token_len": feature["output"]["token_len"],
            "lateral": vla["lateral"],
            "longitudinal": vla["longitudinal"],
            "trajectory_points": vla["trajectory_points"],
            "signature": vla["signature"],
            "trajectory_special_tokens": vla["trajectory_special_tokens"],
        },
    }


def hashed_text_vector(hash_ids: Sequence[int], dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for hash_id in hash_ids:
        vec[int(hash_id) % dim] += 1.0
    return l2_normalize(vec)


def decision_hash_vector(vla: Dict[str, Any], dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in vla.get("lateral_tokens", []) + vla.get("longitudinal_tokens", []):
        idx = int(hashlib.sha1(str(token).encode("utf-8")).hexdigest()[:8], 16) % dim
        vec[idx] += 1.0
    return l2_normalize(vec)


def trajectory_vector(vla: Dict[str, Any], max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    vec = np.zeros(max_points * 2, dtype=np.float32)
    mask = np.zeros(max_points, dtype=np.float32)
    pairs = vla.get("trajectory_loc_pairs") or []
    if pairs:
        for i, pair in enumerate(pairs[:max_points]):
            if len(pair) < 2:
                continue
            vec[i * 2] = float(pair[0]) / 1000.0
            vec[i * 2 + 1] = float(pair[1]) / 1000.0
            mask[i] = 1.0
        return vec, mask

    trajectory_tokens = vla.get("trajectory_special_tokens") or []
    for i, token in enumerate(trajectory_tokens[:max_points]):
        idx_value = int(hashlib.sha1(str(token).encode("utf-8")).hexdigest()[:8], 16) % 1000
        vec[i * 2] = idx_value / 1000.0
        mask[i] = 1.0
    return vec, mask


def stack_or_empty(rows: Sequence[np.ndarray], dim: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def load_tokenizer(model: str):
    from swift.llm import get_model_tokenizer

    _, tokenizer = get_model_tokenizer(model, load_model=False)
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    return tokenizer


def build_feature(
        row: Dict[str, Any],
        row_idx: int,
        sample_id_key: str,
        tokenizer: Any = None,
        keep_text: bool = False,
        keep_token_ids: bool = False) -> Dict[str, Any]:
    messages = row.get("messages")
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {row_idx}: missing non-empty messages")

    input_messages, output_messages = split_input_output(messages)
    input_text = "\n".join(str(m.get("content", "")) for m in input_messages)
    output_text = "\n".join(str(m.get("content", "")) for m in output_messages)

    input_ids = encode(tokenizer, input_text)
    output_ids = encode(tokenizer, output_text)
    parsed_output = parse_vla_output(output_text)
    media = extract_media(row, input_text)

    feature = {
        "sample_id": str(row.get(sample_id_key) or row.get("id") or f"row_{row_idx:08d}"),
        "row_idx": row_idx,
        "channel": row.get("channel"),
        "input": {
            "sha1": sha1(input_text),
            "char_len": len(input_text),
            "token_len": len(input_ids) if input_ids is not None else None,
            "media_tags": count_media_tags(input_text),
            "text_ngram_hashes": ngram_hashes(input_text),
        },
        "media": media,
        "output": {
            "sha1": sha1(output_text),
            "char_len": len(output_text),
            "token_len": len(output_ids) if output_ids is not None else None,
            "special_tokens": SPECIAL_TOKEN_RE.findall(output_text),
            "vla": parsed_output,
        },
    }
    if keep_text:
        feature["input"]["text"] = input_text
        feature["output"]["text"] = output_text
    if keep_token_ids:
        if input_ids is not None:
            feature["input"]["token_ids"] = input_ids
        if output_ids is not None:
            feature["output"]["token_ids"] = output_ids
    return feature


def split_input_output(messages: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    last_assistant = None
    for i, message in enumerate(messages):
        if message.get("role") == "assistant":
            last_assistant = i
    if last_assistant is None:
        return list(messages), []
    return list(messages[:last_assistant]), list(messages[last_assistant:])


def parse_vla_output(text: str) -> Dict[str, Any]:
    special_tokens = SPECIAL_TOKEN_RE.findall(text)
    lateral_tokens = [t for t in special_tokens if t.startswith("<LAT_")]
    longitudinal_tokens = [t for t in special_tokens if t.startswith("<LON_")]
    decision_tokens = set(lateral_tokens + longitudinal_tokens)
    trajectory_special_tokens = [
        t for t in special_tokens if t not in decision_tokens and t not in {"<TRAJ_BOS>", "<TRAJ_EOS>"}
    ]
    loc_values = [int(x) for x in LOC_TOKEN_RE.findall(text)]
    point_values = [int(x) for x in POINT_TOKEN_RE.findall(text)]
    trajectory = []
    for i in range(0, len(loc_values) - 1, 2):
        trajectory.append([loc_values[i], loc_values[i + 1]])
    return {
        "lateral": strip_token(lateral_tokens[0]) if lateral_tokens else None,
        "longitudinal": strip_token(longitudinal_tokens[0]) if longitudinal_tokens else None,
        "lateral_tokens": lateral_tokens,
        "longitudinal_tokens": longitudinal_tokens,
        "point_tokens": point_values,
        "loc_tokens": loc_values,
        "trajectory_special_tokens": trajectory_special_tokens,
        "trajectory_loc_pairs": trajectory,
        "trajectory_points": len(trajectory),
        "signature": "|".join([
            lateral_tokens[0] if lateral_tokens else "",
            longitudinal_tokens[0] if longitudinal_tokens else "",
            ",".join(f"{x}:{y}" for x, y in trajectory) if trajectory else ",".join(trajectory_special_tokens),
        ]),
    }


def extract_media(row: Dict[str, Any], input_text: str) -> Dict[str, Any]:
    images = media_list(row.get("images", row.get("image", [])))
    videos = row.get("videos", row.get("video", []))
    if isinstance(videos, str):
        try:
            videos = json.loads(videos)
        except json.JSONDecodeError:
            videos = [videos]
    if videos is None:
        videos = []
    if not isinstance(videos, list):
        videos = [videos]
    video_items = []
    video_frame_count = 0
    for video in videos:
        if isinstance(video, list):
            frames = [str(x) for x in video]
            video_items.append(frames)
            video_frame_count += len(frames)
        else:
            video_items.append(str(video))
    return {
        "images": images,
        "videos": video_items,
        "image_count": len(images),
        "video_count": len(video_items),
        "video_frame_count": video_frame_count,
        "media_ref_sha1": sha1(json.dumps({"images": images, "videos": video_items}, sort_keys=True)),
        "placeholder_mismatch": {
            "image": input_text.count("<image>") - len(images),
            "video": input_text.count("<video>") - len(video_items),
        },
    }


def media_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def encode(tokenizer: Any, text: str) -> Optional[List[int]]:
    if tokenizer is None:
        return None
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(x) for x in ids]


def count_media_tags(text: str) -> Dict[str, int]:
    counts = {"image": 0, "video": 0, "audio": 0}
    for tag in MEDIA_TAG_RE.findall(text):
        counts[tag] += 1
    return counts


def ngram_hashes(text: str, n: int = 3, buckets: int = 256) -> List[int]:
    tokens = re.findall(r"\w+|<[^<>]+>", text.lower())
    if not tokens:
        return []
    grams = []
    for i in range(max(1, len(tokens) - n + 1)):
        gram = " ".join(tokens[i:i + n])
        grams.append(int(hashlib.sha1(gram.encode("utf-8")).hexdigest()[:8], 16) % buckets)
    return sorted(set(grams))


def strip_token(token: str) -> str:
    return token.strip("<>")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


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
