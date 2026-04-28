#!/usr/bin/env python3
"""Evaluate Chain-of-Causation template patterns in assistant <think> blocks."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from dump_messages_features import parse_vla_output


UNKNOWN = "UNKNOWN"
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
OPEN_THINK_RE = re.compile(r"<think>", re.IGNORECASE)
CLOSE_THINK_RE = re.compile(r"</think>", re.IGNORECASE)
SPECIAL_TOKEN_RE = re.compile(r"<[^<>\s]+>")
URL_RE = re.compile(r"https?://\S+")
DATE_RE = re.compile(r"\b\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
NUMBER_WITH_UNIT_RE = re.compile(
    r"[-+]?\d+(?:\.\d+)?\s*(?:km/h|m/s|公里/小时|千米/小时|米/秒|公里|千米|米|m|秒|s|度|%|％)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class CocRecord:
    row_idx: int
    source_file: str
    sample_id: str
    sample_token: str
    timestamp: str
    scene_id: str
    primary_scene: str
    secondary_scene: str
    lateral: str
    longitudinal: str
    label: str
    exact_id: str
    normalized_id: str
    char_len: int
    byte_len: int
    unique_ngram_ratio: float
    gzip_ratio: float
    think_blocks: int
    malformed_think: bool
    preview: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze CoC/COT template severity from assistant <think>...</think> blocks in JSONL data.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file or directory. Directories are searched recursively for *.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Output directory for CoC reports.")
    parser.add_argument("--scene-map", default=None, help="Optional JSON mapping scene_id to scene labels.")
    parser.add_argument("--sample-id-key", default="sample_id", help="Sample id key in raw JSONL.")
    parser.add_argument("--scene-id-key", default="scene_id", help="Scene id key in raw JSONL.")
    parser.add_argument("--sample-token-key", default="sample_token", help="Sample token key in raw JSONL.")
    parser.add_argument("--timestamp-key", default="timestamp", help="Timestamp key in raw JSONL.")
    parser.add_argument("--primary-scene-key", default="专题", help="Primary scene key in row or --scene-map.")
    parser.add_argument("--secondary-scene-key", default="二级场景", help="Secondary scene key in row or --scene-map.")
    parser.add_argument(
        "--assistant-mode",
        choices=["last", "all"],
        default="last",
        help="Use the last assistant message or concatenate all assistant messages. Default: last.")
    parser.add_argument("--ngram-size", type=int, default=4, help="Character n-gram size for diversity. Default: 4.")
    parser.add_argument("--topn", type=int, default=100, help="Top templates to write. Default: 100.")
    parser.add_argument(
        "--review-topn",
        type=int,
        default=2000,
        help="Highest-template-score rows to write to coc_review_candidates.jsonl. Default: 2000.")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=240,
        help="Max characters kept in CSV/JSON previews. Default: 240.")
    parser.add_argument(
        "--min-review-template-count",
        type=int,
        default=10,
        help="Minimum normalized-template duplicate count before a row is considered strongly templated. Default: 10.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    input_files = resolve_input_files(input_path)
    scene_map = read_json(Path(args.scene_map)) if args.scene_map else {}

    stats = scan_coc_records(input_files, scene_map, args)
    records: List[CocRecord] = stats["records"]
    score_rows = build_score_rows(records, stats, args)
    template_rows = build_template_rows(stats["normalized_template_stats"], stats["normalized_counter"], args.topn)
    exact_rows = build_template_rows(stats["exact_template_stats"], stats["exact_counter"], args.topn)
    summary = build_summary(records, stats, template_rows, exact_rows, args)

    write_csv(output_dir / "coc_sample_scores.csv", score_rows)
    write_csv(output_dir / "coc_top_normalized_templates.csv", template_rows)
    write_csv(output_dir / "coc_top_exact_templates.csv", exact_rows)
    write_csv(output_dir / "coc_length_bins.csv", build_length_bins(records))
    write_jsonl(output_dir / "coc_review_candidates.jsonl", build_review_candidates(score_rows, args.review_topn))
    write_json(output_dir / "coc_summary.json", summary)
    (output_dir / "coc_summary.md").write_text(render_markdown_summary(summary, args.topn), encoding="utf-8")

    print(f"wrote CoC template reports for {summary['rows_with_coc']} CoC rows -> {output_dir}")


def validate_args(args: argparse.Namespace) -> None:
    if args.ngram_size <= 0:
        raise ValueError("--ngram-size must be positive")
    if args.topn <= 0:
        raise ValueError("--topn must be positive")
    if args.review_topn < 0:
        raise ValueError("--review-topn must be non-negative")
    if args.preview_chars <= 0:
        raise ValueError("--preview-chars must be positive")


def resolve_input_files(path: Path) -> List[Path]:
    path = path.expanduser()
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"no *.jsonl files found under {path}")
        return files
    raise FileNotFoundError(str(path))


def scan_coc_records(input_files: Sequence[Path], scene_map: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    records: List[CocRecord] = []
    exact_counter: Counter = Counter()
    normalized_counter: Counter = Counter()
    exact_template_stats: Dict[str, Dict[str, Any]] = {}
    normalized_template_stats: Dict[str, Dict[str, Any]] = {}
    rows_total = 0
    rows_with_assistant = 0
    rows_with_open_think = 0
    malformed_think_rows = 0

    for path in input_files:
        for obj in iter_jsonl(path):
            row_idx = rows_total
            rows_total += 1
            assistant_text = assistant_content(obj, args.assistant_mode)
            if not assistant_text:
                continue
            rows_with_assistant += 1
            if OPEN_THINK_RE.search(assistant_text):
                rows_with_open_think += 1

            coc_text, think_blocks, malformed = extract_coc_text(assistant_text)
            if malformed:
                malformed_think_rows += 1
            if not coc_text:
                continue

            exact_text = canonical_whitespace(coc_text)
            normalized_text = normalize_template_text(exact_text)
            exact_id = stable_id(exact_text)
            normalized_id = stable_id(normalized_text)
            decision = parse_vla_output(remove_think_blocks(assistant_text))
            lateral = str(decision.get("lateral") or UNKNOWN)
            longitudinal = str(decision.get("longitudinal") or UNKNOWN)
            label = f"{lateral}|{longitudinal}"
            meta = extract_meta(obj, scene_map, args, row_idx)
            record = CocRecord(
                row_idx=row_idx,
                source_file=str(path),
                sample_id=meta["sample_id"],
                sample_token=meta["sample_token"],
                timestamp=meta["timestamp"],
                scene_id=meta["scene_id"],
                primary_scene=meta["primary_scene"],
                secondary_scene=meta["secondary_scene"],
                lateral=lateral,
                longitudinal=longitudinal,
                label=label,
                exact_id=exact_id,
                normalized_id=normalized_id,
                char_len=len(exact_text),
                byte_len=len(exact_text.encode("utf-8")),
                unique_ngram_ratio=unique_char_ngram_ratio(exact_text, args.ngram_size),
                gzip_ratio=gzip_size_ratio(exact_text),
                think_blocks=think_blocks,
                malformed_think=malformed,
                preview=preview(exact_text, args.preview_chars),
            )
            records.append(record)
            exact_counter[exact_id] += 1
            normalized_counter[normalized_id] += 1
            update_template_stats(
                exact_template_stats,
                exact_id,
                exact_text,
                normalized_text,
                record,
                args.preview_chars,
            )
            update_template_stats(
                normalized_template_stats,
                normalized_id,
                exact_text,
                normalized_text,
                record,
                args.preview_chars,
            )

    return {
        "records": records,
        "rows_total": rows_total,
        "rows_with_assistant": rows_with_assistant,
        "rows_with_open_think": rows_with_open_think,
        "malformed_think_rows": malformed_think_rows,
        "exact_counter": exact_counter,
        "normalized_counter": normalized_counter,
        "exact_template_stats": exact_template_stats,
        "normalized_template_stats": normalized_template_stats,
    }


def assistant_content(row: Dict[str, Any], mode: str) -> str:
    messages = row.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return ""
    if not isinstance(messages, list):
        return ""
    contents = [str(message.get("content", "")) for message in messages if message.get("role") == "assistant"]
    if not contents:
        return ""
    if mode == "all":
        return "\n".join(contents)
    return contents[-1]


def extract_coc_text(text: str) -> Tuple[str, int, bool]:
    matches = THINK_BLOCK_RE.findall(text)
    open_count = len(OPEN_THINK_RE.findall(text))
    close_count = len(CLOSE_THINK_RE.findall(text))
    malformed = open_count != close_count
    if matches:
        return "\n\n".join(canonical_whitespace(match) for match in matches if canonical_whitespace(match)), len(matches), malformed
    open_match = OPEN_THINK_RE.search(text)
    if open_match:
        return canonical_whitespace(text[open_match.end():]), 1, True
    return "", 0, malformed


def remove_think_blocks(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", text)
    open_match = OPEN_THINK_RE.search(text)
    if open_match:
        text = text[:open_match.start()]
    return text


def extract_meta(row: Dict[str, Any], scene_map: Dict[str, Any], args: argparse.Namespace, row_idx: int) -> Dict[str, str]:
    sample_id = value_or_unknown(row.get(args.sample_id_key), f"row_{row_idx:08d}")
    scene_id = value_or_unknown(row.get(args.scene_id_key), UNKNOWN)
    sample_token = value_or_unknown(row.get(args.sample_token_key), "")
    timestamp = value_or_unknown(row.get(args.timestamp_key), "")
    labels = scene_map.get(scene_id) if scene_map else None
    primary_scene = scene_value(row, labels, args.primary_scene_key)
    secondary_scene = scene_value(row, labels, args.secondary_scene_key)
    return {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "sample_token": sample_token,
        "timestamp": timestamp,
        "primary_scene": primary_scene,
        "secondary_scene": secondary_scene,
    }


def scene_value(row: Dict[str, Any], labels: Any, key: str) -> str:
    value = row.get(key)
    if value not in (None, ""):
        return str(value)
    if isinstance(labels, dict) and labels.get(key) not in (None, ""):
        return str(labels[key])
    return UNKNOWN


def value_or_unknown(value: Any, fallback: str) -> str:
    if value in (None, ""):
        return fallback
    return str(value)


def update_template_stats(
        stats: Dict[str, Dict[str, Any]],
        template_id: str,
        exact_text: str,
        normalized_text: str,
        record: CocRecord,
        preview_chars: int) -> None:
    item = stats.get(template_id)
    if item is None:
        item = {
            "template_id": template_id,
            "preview": preview(exact_text, preview_chars),
            "normalized_preview": preview(normalized_text, preview_chars),
            "labels": Counter(),
            "primary_scenes": Counter(),
            "secondary_scenes": Counter(),
            "scene_ids": Counter(),
            "sample_ids": [],
            "char_len_sum": 0,
            "unique_ngram_ratio_sum": 0.0,
            "gzip_ratio_sum": 0.0,
        }
        stats[template_id] = item
    item["labels"][record.label] += 1
    item["primary_scenes"][record.primary_scene] += 1
    item["secondary_scenes"][record.secondary_scene] += 1
    item["scene_ids"][record.scene_id] += 1
    if len(item["sample_ids"]) < 5:
        item["sample_ids"].append(record.sample_id)
    item["char_len_sum"] += record.char_len
    item["unique_ngram_ratio_sum"] += record.unique_ngram_ratio
    item["gzip_ratio_sum"] += record.gzip_ratio


def build_score_rows(records: Sequence[CocRecord], stats: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    max_norm_count = max(stats["normalized_counter"].values(), default=1)
    rows = []
    total_coc = len(records)
    for record in records:
        normalized_stats = stats["normalized_template_stats"][record.normalized_id]
        exact_count = stats["exact_counter"][record.exact_id]
        normalized_count = stats["normalized_counter"][record.normalized_id]
        label_entropy = entropy(normalized_stats["labels"])
        scene_entropy = entropy(normalized_stats["secondary_scenes"])
        template_score = compute_template_score(
            normalized_count,
            max_norm_count,
            record.unique_ngram_ratio,
            record.gzip_ratio,
            label_entropy,
            scene_entropy,
            normalized_stats,
        )
        rows.append({
            "row_idx": record.row_idx,
            "source_file": record.source_file,
            "sample_id": record.sample_id,
            "sample_token": record.sample_token,
            "timestamp": record.timestamp,
            "scene_id": record.scene_id,
            "primary_scene": record.primary_scene,
            "secondary_scene": record.secondary_scene,
            "lateral": record.lateral,
            "longitudinal": record.longitudinal,
            "label": record.label,
            "char_len": record.char_len,
            "byte_len": record.byte_len,
            "think_blocks": record.think_blocks,
            "malformed_think": record.malformed_think,
            "exact_template_id": record.exact_id,
            "exact_dup_count": exact_count,
            "exact_dup_ratio": safe_div(exact_count, total_coc),
            "normalized_template_id": record.normalized_id,
            "normalized_dup_count": normalized_count,
            "normalized_dup_ratio": safe_div(normalized_count, total_coc),
            "normalized_label_entropy": label_entropy,
            "normalized_secondary_scene_entropy": scene_entropy,
            "normalized_unique_labels": len(normalized_stats["labels"]),
            "normalized_unique_secondary_scenes": len(normalized_stats["secondary_scenes"]),
            "unique_ngram_ratio": record.unique_ngram_ratio,
            "gzip_ratio": record.gzip_ratio,
            "template_score": template_score,
            "strong_template_candidate": normalized_count >= args.min_review_template_count,
            "coc_preview": record.preview,
        })
    rows.sort(key=lambda row: (-float(row["template_score"]), -int(row["normalized_dup_count"]), int(row["row_idx"])))
    return rows


def compute_template_score(
        normalized_count: int,
        max_norm_count: int,
        unique_ngram_ratio: float,
        gzip_ratio: float,
        label_entropy: float,
        scene_entropy: float,
        template_stats: Dict[str, Any]) -> float:
    duplicate_score = safe_div(math.log1p(normalized_count), math.log1p(max_norm_count))
    low_ngram_score = clamp01(1.0 - unique_ngram_ratio)
    low_gzip_score = clamp01((0.95 - gzip_ratio) / 0.95)
    label_entropy_score = normalized_entropy(label_entropy, len(template_stats["labels"]))
    scene_entropy_score = normalized_entropy(scene_entropy, len(template_stats["secondary_scenes"]))
    cross_context_score = max(label_entropy_score, scene_entropy_score)
    return round(
        0.50 * duplicate_score + 0.20 * low_ngram_score + 0.20 * low_gzip_score + 0.10 * cross_context_score,
        6,
    )


def build_template_rows(
        template_stats: Dict[str, Dict[str, Any]],
        counter: Counter,
        topn: int) -> List[Dict[str, Any]]:
    total = sum(counter.values())
    rows = []
    for template_id, count in counter.most_common(topn):
        item = template_stats[template_id]
        rows.append({
            "template_id": template_id,
            "count": count,
            "ratio": safe_div(count, total),
            "avg_char_len": safe_div(item["char_len_sum"], count),
            "avg_unique_ngram_ratio": safe_div(item["unique_ngram_ratio_sum"], count),
            "avg_gzip_ratio": safe_div(item["gzip_ratio_sum"], count),
            "label_entropy": entropy(item["labels"]),
            "secondary_scene_entropy": entropy(item["secondary_scenes"]),
            "unique_labels": len(item["labels"]),
            "unique_primary_scenes": len(item["primary_scenes"]),
            "unique_secondary_scenes": len(item["secondary_scenes"]),
            "unique_scene_ids": len(item["scene_ids"]),
            "top_labels": format_counter(item["labels"], 8),
            "top_primary_scenes": format_counter(item["primary_scenes"], 8),
            "top_secondary_scenes": format_counter(item["secondary_scenes"], 8),
            "sample_ids": ";".join(item["sample_ids"]),
            "preview": item["preview"],
            "normalized_preview": item["normalized_preview"],
        })
    return rows


def build_review_candidates(score_rows: Sequence[Dict[str, Any]], limit: int) -> Iterable[Dict[str, Any]]:
    for row in score_rows[:limit]:
        yield row


def build_length_bins(records: Sequence[CocRecord]) -> List[Dict[str, Any]]:
    bins = [
        (0, 50),
        (50, 100),
        (100, 200),
        (200, 400),
        (400, 800),
        (800, 1600),
        (1600, None),
    ]
    counts = []
    total = len(records)
    for lo, hi in bins:
        if hi is None:
            count = sum(1 for record in records if record.char_len >= lo)
            label = f">={lo}"
        else:
            count = sum(1 for record in records if lo <= record.char_len < hi)
            label = f"[{lo},{hi})"
        counts.append({"char_len_bin": label, "count": count, "ratio": safe_div(count, total)})
    return counts


def build_summary(
        records: Sequence[CocRecord],
        stats: Dict[str, Any],
        normalized_rows: Sequence[Dict[str, Any]],
        exact_rows: Sequence[Dict[str, Any]],
        args: argparse.Namespace) -> Dict[str, Any]:
    char_lens = [record.char_len for record in records]
    unique_ngram_ratios = [record.unique_ngram_ratio for record in records]
    gzip_ratios = [record.gzip_ratio for record in records]
    exact_counter = stats["exact_counter"]
    normalized_counter = stats["normalized_counter"]
    summary = {
        "input": args.input,
        "ngram_size": args.ngram_size,
        "rows_total": stats["rows_total"],
        "rows_with_assistant": stats["rows_with_assistant"],
        "rows_with_open_think": stats["rows_with_open_think"],
        "rows_with_coc": len(records),
        "coc_coverage_total": safe_div(len(records), stats["rows_total"]),
        "coc_coverage_assistant": safe_div(len(records), stats["rows_with_assistant"]),
        "malformed_think_rows": stats["malformed_think_rows"],
        "exact_unique_templates": len(exact_counter),
        "exact_unique_ratio": safe_div(len(exact_counter), len(records)),
        "normalized_unique_templates": len(normalized_counter),
        "normalized_unique_ratio": safe_div(len(normalized_counter), len(records)),
        "top_1_normalized_coverage": top_k_coverage(normalized_counter, 1),
        "top_10_normalized_coverage": top_k_coverage(normalized_counter, 10),
        "top_100_normalized_coverage": top_k_coverage(normalized_counter, 100),
        "top_1_exact_coverage": top_k_coverage(exact_counter, 1),
        "top_10_exact_coverage": top_k_coverage(exact_counter, 10),
        "top_100_exact_coverage": top_k_coverage(exact_counter, 100),
        "char_len": distribution_summary(char_lens),
        "unique_ngram_ratio": distribution_summary(unique_ngram_ratios),
        "gzip_ratio": distribution_summary(gzip_ratios),
        "template_risk": classify_template_risk(
            safe_div(len(normalized_counter), len(records)),
            top_k_coverage(normalized_counter, 100),
            top_k_coverage(normalized_counter, 10),
        ),
        "top_normalized_templates": list(normalized_rows[: min(10, len(normalized_rows))]),
        "top_exact_templates": list(exact_rows[: min(10, len(exact_rows))]),
        "outputs": {
            "summary_json": "coc_summary.json",
            "summary_md": "coc_summary.md",
            "sample_scores": "coc_sample_scores.csv",
            "top_normalized_templates": "coc_top_normalized_templates.csv",
            "top_exact_templates": "coc_top_exact_templates.csv",
            "length_bins": "coc_length_bins.csv",
            "review_candidates": "coc_review_candidates.jsonl",
        },
    }
    return summary


def classify_template_risk(normalized_unique_ratio: float, top100_coverage: float, top10_coverage: float) -> str:
    if top10_coverage >= 0.30 or top100_coverage >= 0.50 or normalized_unique_ratio <= 0.30:
        return "high"
    if top10_coverage >= 0.15 or top100_coverage >= 0.30 or normalized_unique_ratio <= 0.60:
        return "medium"
    return "low"


def render_markdown_summary(summary: Dict[str, Any], topn: int) -> str:
    lines = [
        "# CoC Template Analysis",
        "",
        "## Overview",
        "",
        f"- Rows total: {summary['rows_total']}",
        f"- Rows with assistant: {summary['rows_with_assistant']}",
        f"- Rows with CoC: {summary['rows_with_coc']} ({summary['coc_coverage_total']:.4f} of total)",
        f"- Malformed think rows: {summary['malformed_think_rows']}",
        f"- Template risk: **{summary['template_risk']}**",
        "",
        "## Duplicate Coverage",
        "",
        f"- Exact unique ratio: {summary['exact_unique_ratio']:.4f}",
        f"- Normalized unique ratio: {summary['normalized_unique_ratio']:.4f}",
        f"- Top-10 normalized coverage: {summary['top_10_normalized_coverage']:.4f}",
        f"- Top-100 normalized coverage: {summary['top_100_normalized_coverage']:.4f}",
        "",
        "## Length / Diversity",
        "",
        f"- Char length p50 / p90 / p99: {fmt_quantiles(summary['char_len'])}",
        f"- Unique {summary['ngram_size']}-gram ratio p50 / p90 / p99: {fmt_quantiles(summary['unique_ngram_ratio'])}",
        f"- Gzip ratio p50 / p90 / p99: {fmt_quantiles(summary['gzip_ratio'])}",
        "",
        f"## Top Normalized Templates",
        "",
    ]
    for row in summary["top_normalized_templates"][:topn]:
        lines.append(
            f"- `{row['template_id']}` count={row['count']} ratio={row['ratio']:.4f} "
            f"labels={row['unique_labels']} scenes={row['unique_secondary_scenes']} preview={row['normalized_preview']}"
        )
    lines.extend([
        "",
        "## Suggested Checks",
        "",
        "- 如果 top normalized templates 覆盖率很高，优先抽查 `coc_top_normalized_templates.csv`。",
        "- 如果同一 normalized template 横跨多个 label 或 secondary_scene，优先进入人工复核。",
        "- 如果高 `template_score` 样本训练 loss 很低但决策收益不明显，可以考虑下采样或不对 CoC 计 loss。",
        "",
    ])
    return "\n".join(lines)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def canonical_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", str(text).strip())


def normalize_template_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = URL_RE.sub("<URL>", text)
    text = DATE_RE.sub("<DATE>", text)
    text = TIME_RE.sub("<TIME>", text)
    text = SPECIAL_TOKEN_RE.sub("<TOKEN>", text)
    text = NUMBER_WITH_UNIT_RE.sub("<NUM>", text)
    text = NUMBER_RE.sub("<NUM>", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def preview(text: str, limit: int) -> str:
    text = canonical_whitespace(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def unique_char_ngram_ratio(text: str, n: int) -> float:
    compact = WHITESPACE_RE.sub("", text)
    if not compact:
        return 0.0
    if len(compact) <= n:
        return 1.0
    ngrams = [compact[i:i + n] for i in range(0, len(compact) - n + 1)]
    return safe_div(len(set(ngrams)), len(ngrams))


def gzip_size_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(gzip.compress(raw)) / len(raw)


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            value -= p * math.log(p)
    return value


def normalized_entropy(value: float, num_classes: int) -> float:
    if num_classes <= 1:
        return 0.0
    return safe_div(value, math.log(num_classes))


def format_counter(counter: Counter, topn: int) -> str:
    total = sum(counter.values())
    return ";".join(f"{key}:{count}:{safe_div(count, total):.4f}" for key, count in counter.most_common(topn))


def distribution_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "mean": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "p50": quantile(ordered, 0.50),
        "p90": quantile(ordered, 0.90),
        "p99": quantile(ordered, 0.99),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
    }


def quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight)


def top_k_coverage(counter: Counter, k: int) -> float:
    total = sum(counter.values())
    return safe_div(sum(count for _, count in counter.most_common(k)), total)


def fmt_quantiles(summary: Dict[str, float]) -> str:
    return f"{summary['p50']:.4g} / {summary['p90']:.4g} / {summary['p99']:.4g}"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


if __name__ == "__main__":
    main()
