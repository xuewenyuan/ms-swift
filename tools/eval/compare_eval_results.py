#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
import argparse
import math
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Optional


# =========================================================
# 顶部配置区：命令行未传参时，将使用这里的配置
# 命令行参数优先级更高
# =========================================================
CONFIG = {
    "base_va": "",
    "base_ref": "",
    "opsd_va": "",
    "sft_va": "",
    "output_dir": "",
}


# =========================
# 工具函数
# =========================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_div(a: int, b: int) -> float:
    return float(a) / float(b) if b else 0.0


def normalize_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        x = x.strip().lower()
        if x in {"true", "1", "yes", "y"}:
            return True
        if x in {"false", "0", "no", "n"}:
            return False
    return False


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"输入文件不是 dict 格式: {path}")
    return data


def convert_obj(obj):
    if isinstance(obj, Counter):
        new_obj = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                k = " -> ".join(map(str, k))
            else:
                k = str(k)
            new_obj[k] = convert_obj(v)
        return new_obj

    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                k = " -> ".join(map(str, k))
            elif not isinstance(k, (str, int, float, bool)) and k is not None:
                k = str(k)
            new_obj[k] = convert_obj(v)
        return new_obj

    if isinstance(obj, list):
        return [convert_obj(x) for x in obj]

    return obj


def write_json(path: Path, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert_obj(obj), f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_counter_csv(path: Path, counter: Counter, header: List[str]):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header + ["count", "ratio"])
        total = sum(counter.values())
        for k, v in counter.most_common():
            if isinstance(k, tuple):
                row = list(k)
            else:
                row = [k]
            writer.writerow(row + [v, round(safe_div(v, total), 6)])


def sanitize_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").replace("/", "_").lower()


def sorted_items_desc(counter_or_dict):
    if hasattr(counter_or_dict, "most_common"):
        return counter_or_dict.most_common()
    return sorted(counter_or_dict.items(), key=lambda x: x[1], reverse=True)


def frame_key_from_frame(frame: Dict[str, Any], fallback_index: int) -> str:
    """
    只用 scene_id + sample_token 匹配
    """
    scene_id = str(frame.get("scene_id", "")).strip()
    sample_token = frame.get("sample_token", None)

    if sample_token is None:
        sample_token = "NULL"
    else:
        sample_token = str(sample_token).strip()
        if sample_token == "" or sample_token.lower() == "null":
            sample_token = "NULL"

    return f"{scene_id}||{sample_token}"


def extract_clip_scene(clip_value: Dict[str, Any]) -> str:
    return str(clip_value.get("scene", ""))


def build_clip_map(data: Dict[str, Any], model_name: str) -> Dict[str, Dict[str, Any]]:
    clip_map = {}
    for clip_key, clip_value in data.items():
        if not isinstance(clip_value, dict):
            continue
        frame_info = clip_value.get("frame_info", [])
        clip_map[clip_key] = {
            "clip_key": clip_key,
            "scene": extract_clip_scene(clip_value),
            "checker": normalize_bool(clip_value.get("checker", False)),
            "num_frames": len(frame_info) if isinstance(frame_info, list) else 0,
            "model_name": model_name,
        }
    return clip_map


def build_frame_map(data: Dict[str, Any], model_name: str) -> Dict[str, Dict[str, Any]]:
    frame_map = {}
    dup_count = 0

    for clip_key, clip_value in data.items():
        if not isinstance(clip_value, dict):
            continue

        clip_scene = extract_clip_scene(clip_value)
        frames = clip_value.get("frame_info", [])
        if not isinstance(frames, list):
            continue

        for i, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue

            fk = frame_key_from_frame(frame, fallback_index=i)
            if fk in frame_map:
                dup_count += 1
                continue

            frame_map[fk] = {
                "frame_key": fk,
                "clip_key": clip_key,
                "scene_id": frame.get("scene_id"),
                "time_sample": frame.get("time_sample", frame.get("timestamp")),
                "sample_token": frame.get("sample_token"),
                "scene": frame.get("scene", clip_scene),
                "clip_section": frame.get("clip_section"),
                "index": frame.get("index", i),
                "checker": normalize_bool(frame.get("checker", False)),
                "prediction": frame.get("prediction"),
                "human_label": frame.get("human_label"),
                "model_name": model_name,
            }

    if dup_count > 0:
        print(f"[WARN] {model_name} 中发现重复 scene_id+sample_token，已忽略后续重复项: {dup_count}")

    return frame_map


def calc_accuracy_from_bool_list(bools: List[bool]) -> Dict[str, Any]:
    total = len(bools)
    correct = sum(1 for x in bools if x)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(safe_div(correct, total), 6)
    }


def calc_clip_accuracy(clip_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return calc_accuracy_from_bool_list([v["checker"] for v in clip_map.values()])


def calc_frame_accuracy(frame_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return calc_accuracy_from_bool_list([v["checker"] for v in frame_map.values()])


def subset_accuracy_from_keys(record_map: Dict[str, Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    bools = [record_map[k]["checker"] for k in keys if k in record_map]
    return calc_accuracy_from_bool_list(bools)


def count_present_keys(record_map: Dict[str, Dict[str, Any]], keys: List[str]) -> int:
    return sum(1 for k in keys if k in record_map)


def exact_two_sided_binom_pvalue(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return 1.0
    tail = sum(math.comb(n, i) * (p**i) * ((1 - p)**(n - i)) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail)


def mcnemar_test_from_discordant(left_only_correct: int, right_only_correct: int) -> Dict[str, Any]:
    discordant = left_only_correct + right_only_correct
    if discordant == 0:
        return {
            "left_only_correct": left_only_correct,
            "right_only_correct": right_only_correct,
            "discordant_total": 0,
            "exact_pvalue": 1.0,
            "chi2": 0.0,
            "chi2_pseudo_stat": 0.0,
            "winner": "tie",
            "significant_at_0_05": False,
        }

    smaller = min(left_only_correct, right_only_correct)
    exact_pvalue = exact_two_sided_binom_pvalue(smaller, discordant, p=0.5)
    chi2 = ((abs(left_only_correct - right_only_correct) - 1.0)**2) / discordant
    if left_only_correct > right_only_correct:
        winner = "left"
    elif right_only_correct > left_only_correct:
        winner = "right"
    else:
        winner = "tie"
    return {
        "left_only_correct": left_only_correct,
        "right_only_correct": right_only_correct,
        "discordant_total": discordant,
        "exact_pvalue": round(exact_pvalue, 8),
        "chi2": round(chi2, 8),
        "chi2_pseudo_stat": round(chi2, 8),
        "winner": winner,
        "significant_at_0_05": exact_pvalue < 0.05,
    }


def compare_pairwise_stats(
    left_map: Dict[str, Dict[str, Any]],
    right_map: Dict[str, Dict[str, Any]],
    left_name: str,
    right_name: str,
    compare_keys: List[str],
) -> Dict[str, Any]:
    left_only_correct = 0
    right_only_correct = 0
    both_correct = 0
    both_wrong = 0

    for k in compare_keys:
        l_ok = left_map[k]["checker"]
        r_ok = right_map[k]["checker"]
        if l_ok and r_ok:
            both_correct += 1
        elif l_ok and (not r_ok):
            left_only_correct += 1
        elif (not l_ok) and r_ok:
            right_only_correct += 1
        else:
            both_wrong += 1

    stat = mcnemar_test_from_discordant(left_only_correct, right_only_correct)
    stat.update({
        "left_name": left_name,
        "right_name": right_name,
        "num_common": len(compare_keys),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "left_accuracy": round(safe_div(left_only_correct + both_correct, len(compare_keys)), 6),
        "right_accuracy": round(safe_div(right_only_correct + both_correct, len(compare_keys)), 6),
        "accuracy_delta_right_minus_left": round(
            safe_div(right_only_correct - left_only_correct, len(compare_keys)), 6),
    })
    return stat


REFERENCE_BUCKET_LABELS = {
    "base错_ref对_target对": "initial model 在 reference-free 下错、在 reference-based 下对，且 target 也修复成功",
    "base错_ref错_target对": "initial model 在 reference-free/reference-based 下都错，但 target 修复成功",
    "base错_ref缺失_target对": "initial model 在 reference-free 下错，base_ref 缺失，且 target 修复成功",
    "base对_ref对_target错": "initial model 在 reference-free/reference-based 下都对，但 target 退化为错",
    "base对_ref错_target错": "initial model 在 reference-free 下对、在 reference-based 下反而错，且 target 也错",
    "base对_ref缺失_target错": "initial model 在 reference-free 下对，base_ref 缺失，且 target 错",
}


def format_reference_bucket_line(bucket_name: str, count: int, ratio: float) -> str:
    desc = REFERENCE_BUCKET_LABELS.get(bucket_name, bucket_name)
    return f"- {bucket_name}: {count} ({ratio:.6f})\n  含义：{desc}"


def merged_row_from_models(
    frame_key: str,
    model_to_record: Dict[str, Optional[Dict[str, Any]]]
) -> Dict[str, Any]:
    base_meta = None
    for rec in model_to_record.values():
        if rec is not None:
            base_meta = rec
            break

    row = {
        "frame_key": frame_key,
        "clip_key": base_meta.get("clip_key") if base_meta else None,
        "scene_id": base_meta.get("scene_id") if base_meta else None,
        "time_sample": base_meta.get("time_sample") if base_meta else None,
        "sample_token": base_meta.get("sample_token") if base_meta else None,
        "scene": base_meta.get("scene") if base_meta else None,
        "clip_section": base_meta.get("clip_section") if base_meta else None,
        "index": base_meta.get("index") if base_meta else None,
        "human_label": base_meta.get("human_label") if base_meta else None,
    }

    for model_name, rec in model_to_record.items():
        prefix = sanitize_name(model_name)
        row[f"{prefix}_checker"] = rec.get("checker") if rec else None
        row[f"{prefix}_prediction"] = rec.get("prediction") if rec else None

    return row


def compare_pairwise_four_buckets(
    left_map: Dict[str, Dict[str, Any]],
    right_map: Dict[str, Dict[str, Any]],
    left_name: str,
    right_name: str,
    compare_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    left_keys = set(left_map.keys())
    right_keys = set(right_map.keys())
    common_keys = sorted(compare_keys) if compare_keys is not None else sorted(left_keys & right_keys)

    both_correct = []
    left_only_correct = []
    right_only_correct = []
    both_wrong = []

    for k in common_keys:
        l_ok = left_map[k]["checker"]
        r_ok = right_map[k]["checker"]

        if l_ok and r_ok:
            both_correct.append(k)
        elif (not l_ok) and r_ok:
            right_only_correct.append(k)
        elif l_ok and (not r_ok):
            left_only_correct.append(k)
        else:
            both_wrong.append(k)

    return {
        "left_name": left_name,
        "right_name": right_name,
        "num_left_total": len(left_keys),
        "num_right_total": len(right_keys),
        "num_common": len(common_keys),
        "missing_in_right": len(left_keys - right_keys),
        "missing_in_left": len(right_keys - left_keys),
        "counts": {
            "both_correct": len(both_correct),
            f"{sanitize_name(left_name)}_only_correct": len(left_only_correct),
            f"{sanitize_name(right_name)}_only_correct": len(right_only_correct),
            "both_wrong": len(both_wrong),
        },
        "ratios_on_common": {
            "both_correct": round(safe_div(len(both_correct), len(common_keys)), 6),
            f"{sanitize_name(left_name)}_only_correct": round(safe_div(len(left_only_correct), len(common_keys)), 6),
            f"{sanitize_name(right_name)}_only_correct": round(safe_div(len(right_only_correct), len(common_keys)), 6),
            "both_wrong": round(safe_div(len(both_wrong), len(common_keys)), 6),
        },
        "keys": {
            "both_correct": both_correct,
            "left_only_correct": left_only_correct,
            "right_only_correct": right_only_correct,
            "both_wrong": both_wrong,
        }
    }


def compare_target_vs_base(
    base_map: Dict[str, Dict[str, Any]],
    target_map: Dict[str, Dict[str, Any]],
    base_name: str,
    target_name: str,
    ref_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ref_name: Optional[str] = None,
    compare_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    common_keys = sorted(compare_keys) if compare_keys is not None else sorted(set(base_map.keys()) & set(target_map.keys()))

    improved = []       # base错，target对
    regressed = []      # base对，target错
    both_correct = []
    both_wrong = []

    improved_by_scene = Counter()
    improved_by_label = Counter()
    improved_from_base_pred_to_label = Counter()

    regressed_by_scene = Counter()
    regressed_by_label = Counter()

    improved_ref_bucket = Counter()
    regressed_ref_bucket = Counter()

    for k in common_keys:
        b_ok = base_map[k]["checker"]
        t_ok = target_map[k]["checker"]

        scene = str(base_map[k].get("scene", ""))
        human_label = str(base_map[k].get("human_label", ""))
        base_pred = str(base_map[k].get("prediction", ""))

        if (not b_ok) and t_ok:
            improved.append(k)
            improved_by_scene[scene] += 1
            improved_by_label[human_label] += 1
            improved_from_base_pred_to_label[(base_pred, human_label)] += 1

            if ref_map is not None:
                if k in ref_map:
                    r_ok = ref_map[k]["checker"]
                    if r_ok:
                        improved_ref_bucket["base错_ref对_target对"] += 1
                    else:
                        improved_ref_bucket["base错_ref错_target对"] += 1
                else:
                    improved_ref_bucket["base错_ref缺失_target对"] += 1

        elif b_ok and (not t_ok):
            regressed.append(k)
            regressed_by_scene[scene] += 1
            regressed_by_label[human_label] += 1

            if ref_map is not None:
                if k in ref_map:
                    r_ok = ref_map[k]["checker"]
                    if r_ok:
                        regressed_ref_bucket["base对_ref对_target错"] += 1
                    else:
                        regressed_ref_bucket["base对_ref错_target错"] += 1
                else:
                    regressed_ref_bucket["base对_ref缺失_target错"] += 1

        elif b_ok and t_ok:
            both_correct.append(k)
        else:
            both_wrong.append(k)

    out = {
        "base_name": base_name,
        "target_name": target_name,
        "num_common": len(common_keys),
        "counts": {
            "improved": len(improved),
            "regressed": len(regressed),
            "both_correct": len(both_correct),
            "both_wrong": len(both_wrong),
        },
        "ratios_on_common": {
            "improved": round(safe_div(len(improved), len(common_keys)), 6),
            "regressed": round(safe_div(len(regressed), len(common_keys)), 6),
            "both_correct": round(safe_div(len(both_correct), len(common_keys)), 6),
            "both_wrong": round(safe_div(len(both_wrong), len(common_keys)), 6),
        },
        "keys": {
            "improved": improved,
            "regressed": regressed,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        },
        "breakdown": {
            "improved_by_scene": improved_by_scene,
            "improved_by_label": improved_by_label,
            "improved_from_base_pred_to_label": improved_from_base_pred_to_label,
            "regressed_by_scene": regressed_by_scene,
            "regressed_by_label": regressed_by_label,
        }
    }

    if ref_map is not None and ref_name is not None:
        out["reference_name"] = ref_name
        out["reference_coverage"] = {
            "num_compare_keys": len(common_keys),
            "num_ref_present": count_present_keys(ref_map, common_keys),
            "coverage_ratio": round(safe_div(count_present_keys(ref_map, common_keys), len(common_keys)), 6),
        }
        out["breakdown"]["improved_by_reference_bucket"] = improved_ref_bucket
        out["breakdown"]["regressed_by_reference_bucket"] = regressed_ref_bucket

    return out


def analyze_reference_helpfulness_on_main_set(
    base_va_map: Dict[str, Dict[str, Any]],
    base_ref_map: Dict[str, Dict[str, Any]],
    sft_map: Dict[str, Dict[str, Any]],
    opsd_map: Dict[str, Dict[str, Any]],
    compare_keys: List[str],
) -> Dict[str, Any]:
    covered_keys = [k for k in compare_keys if k in base_ref_map]

    buckets = {
        "ref_helpful_base_wrong_ref_right": [],
        "both_correct": [],
        "ref_harmful_base_right_ref_wrong": [],
        "both_wrong": [],
    }

    for k in covered_keys:
        b_ok = base_va_map[k]["checker"]
        r_ok = base_ref_map[k]["checker"]
        if (not b_ok) and r_ok:
            buckets["ref_helpful_base_wrong_ref_right"].append(k)
        elif b_ok and r_ok:
            buckets["both_correct"].append(k)
        elif b_ok and (not r_ok):
            buckets["ref_harmful_base_right_ref_wrong"].append(k)
        else:
            buckets["both_wrong"].append(k)

    bucket_ratios = {
        bucket_name: round(safe_div(len(keys), len(covered_keys)), 6)
        for bucket_name, keys in buckets.items()
    }

    per_bucket = {}
    model_maps = {
        "base_va": base_va_map,
        "base_ref": base_ref_map,
        "sft_va": sft_map,
        "opsd_va": opsd_map,
    }

    for bucket_name, keys in buckets.items():
        bucket_info = {
            "count": len(keys),
            "ratio_on_base_ref_covered_main_set": round(safe_div(len(keys), len(covered_keys)), 6),
            "accuracies": {
                model_name: subset_accuracy_from_keys(model_map, keys)
                for model_name, model_map in model_maps.items()
            },
            "sft_vs_base_va": compare_target_vs_base(
                base_map=base_va_map,
                target_map=sft_map,
                base_name="base_va",
                target_name="sft_va",
                compare_keys=keys,
            ),
            "opsd_vs_base_va": compare_target_vs_base(
                base_map=base_va_map,
                target_map=opsd_map,
                base_name="base_va",
                target_name="opsd_va",
                compare_keys=keys,
            ),
            "opsd_vs_sft": compare_pairwise_four_buckets(
                left_map=opsd_map,
                right_map=sft_map,
                left_name="opsd_va",
                right_name="sft_va",
                compare_keys=keys,
            ),
        }

        base_wrong_keys = [k for k in keys if not base_va_map[k]["checker"]]
        bucket_info["repair_on_base_wrong_subset"] = {
            "base_wrong_count": len(base_wrong_keys),
            "sft_repaired": sum(1 for k in base_wrong_keys if sft_map[k]["checker"]),
            "opsd_repaired": sum(1 for k in base_wrong_keys if opsd_map[k]["checker"]),
            "sft_repair_rate": round(
                safe_div(sum(1 for k in base_wrong_keys if sft_map[k]["checker"]), len(base_wrong_keys)), 6),
            "opsd_repair_rate": round(
                safe_div(sum(1 for k in base_wrong_keys if opsd_map[k]["checker"]), len(base_wrong_keys)), 6),
        }
        bucket_info["paired_stats"] = {
            "sft_vs_base_va": compare_pairwise_stats(
                left_map=base_va_map,
                right_map=sft_map,
                left_name="base_va",
                right_name="sft_va",
                compare_keys=keys,
            ),
            "opsd_vs_base_va": compare_pairwise_stats(
                left_map=base_va_map,
                right_map=opsd_map,
                left_name="base_va",
                right_name="opsd_va",
                compare_keys=keys,
            ),
            "opsd_vs_sft": compare_pairwise_stats(
                left_map=sft_map,
                right_map=opsd_map,
                left_name="sft_va",
                right_name="opsd_va",
                compare_keys=keys,
            ),
        }
        per_bucket[bucket_name] = bucket_info

    return {
        "num_main_3way_keys": len(compare_keys),
        "num_base_ref_covered_keys": len(covered_keys),
        "base_ref_coverage_ratio_on_main_3way": round(safe_div(len(covered_keys), len(compare_keys)), 6),
        "bucket_counts": {bucket_name: len(keys) for bucket_name, keys in buckets.items()},
        "bucket_ratios": bucket_ratios,
        "bucket_keys": buckets,
        "per_bucket": per_bucket,
    }


def save_pairwise_bucket_details(
    out_dir: Path,
    cmp_res: Dict[str, Any],
    left_map: Dict[str, Dict[str, Any]],
    right_map: Dict[str, Dict[str, Any]],
    left_name: str,
    right_name: str,
):
    ensure_dir(out_dir)

    bucket_to_keys = {
        "both_correct": cmp_res["keys"]["both_correct"],
        f"{sanitize_name(left_name)}_only_correct": cmp_res["keys"]["left_only_correct"],
        f"{sanitize_name(right_name)}_only_correct": cmp_res["keys"]["right_only_correct"],
        "both_wrong": cmp_res["keys"]["both_wrong"],
    }

    for bucket_name, keys in bucket_to_keys.items():
        rows = []
        for k in keys:
            rows.append(merged_row_from_models(k, {
                left_name: left_map.get(k),
                right_name: right_map.get(k),
            }))
        write_jsonl(out_dir / f"{bucket_name}.jsonl", rows)


def save_target_vs_base_details(
    out_dir: Path,
    cmp_res: Dict[str, Any],
    base_map: Dict[str, Dict[str, Any]],
    target_map: Dict[str, Dict[str, Any]],
    base_name: str,
    target_name: str,
    ref_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ref_name: Optional[str] = None,
):
    ensure_dir(out_dir)

    for bucket_name in ["improved", "regressed", "both_correct", "both_wrong"]:
        rows = []
        for k in cmp_res["keys"][bucket_name]:
            models = {
                base_name: base_map.get(k),
                target_name: target_map.get(k),
            }
            if ref_map is not None and ref_name is not None:
                models[ref_name] = ref_map.get(k)
            rows.append(merged_row_from_models(k, models))
        write_jsonl(out_dir / f"{bucket_name}.jsonl", rows)

    bd = cmp_res["breakdown"]
    write_counter_csv(out_dir / "improved_by_scene.csv", bd["improved_by_scene"], ["scene"])
    write_counter_csv(out_dir / "improved_by_label.csv", bd["improved_by_label"], ["human_label"])
    write_counter_csv(
        out_dir / "improved_from_base_pred_to_label.csv",
        bd["improved_from_base_pred_to_label"],
        ["base_prediction", "human_label"]
    )
    write_counter_csv(out_dir / "regressed_by_scene.csv", bd["regressed_by_scene"], ["scene"])
    write_counter_csv(out_dir / "regressed_by_label.csv", bd["regressed_by_label"], ["human_label"])

    if "improved_by_reference_bucket" in bd:
        write_counter_csv(
            out_dir / "improved_by_reference_bucket.csv",
            bd["improved_by_reference_bucket"],
            ["bucket"]
        )
    if "regressed_by_reference_bucket" in bd:
        write_counter_csv(
            out_dir / "regressed_by_reference_bucket.csv",
            bd["regressed_by_reference_bucket"],
            ["bucket"]
        )


def format_acc_line(name: str, acc_obj: Dict[str, Any]) -> str:
    return f"- {name}: correct={acc_obj['correct']}, total={acc_obj['total']}, acc={acc_obj['accuracy']:.6f}"


def format_stat_test_line(stat: Dict[str, Any]) -> str:
    winner_map = {
        "left": stat["left_name"],
        "right": stat["right_name"],
        "tie": "tie",
    }
    return (
        f"- {stat['left_name']} vs {stat['right_name']}: "
        f"delta({stat['right_name']}-{stat['left_name']})={stat['accuracy_delta_right_minus_left']:.6f}, "
        f"discordant={stat['discordant_total']}, "
        f"exact_p={stat['exact_pvalue']:.8f}, "
        f"winner={winner_map[stat['winner']]}, "
        f"significant={stat['significant_at_0_05']}"
    )


def summarize_stat_conclusion(stat: Dict[str, Any]) -> str:
    left_name = stat["left_name"]
    right_name = stat["right_name"]
    delta = stat["accuracy_delta_right_minus_left"]
    pvalue = stat["exact_pvalue"]
    significant = stat["significant_at_0_05"]
    winner = stat["winner"]
    discordant = stat["discordant_total"]

    if discordant == 0:
        return f"{left_name} 与 {right_name} 在公共样本上没有分歧样本，当前无法区分优劣。"

    if winner == "tie" or abs(delta) < 1e-12:
        if significant:
            return f"{left_name} 与 {right_name} 差异接近于 0，但统计结果异常地显示显著，建议复核数据。"
        return f"{left_name} 与 {right_name} 在公共样本上的差异不明显，未观察到统计显著差异。"

    better = right_name if winner == "right" else left_name
    worse = left_name if winner == "right" else right_name
    if significant:
        return (f"{better} 相比 {worse} 表现更好，且差异达到统计显著 "
                f"(McNemar exact p={pvalue:.8f})。")
    return (f"{better} 相比 {worse} 看起来更好，但差异尚未达到统计显著 "
            f"(McNemar exact p={pvalue:.8f})。")


def write_report_md(path: Path, summary: Dict[str, Any]):
    lines = []

    lines.append("# 模型评测对比报告\n")

    lines.append("## 0. 实验定义\n")
    lines.append("- `base model`：训练 OPSD 和 SFT 时共同使用的 initial model。")
    lines.append("- `base_va`：base model 在 reference-free 条件下的推理结果。")
    lines.append("- `base_ref`：base model 在 reference-based 条件下的推理结果。")
    lines.append("- `sft_va` / `opsd_va`：SFT / OPSD 训练后模型在 reference-free 条件下的推理结果。")
    lines.append("- 统计检验使用 paired McNemar exact test，仅在公共 frame 集上进行。")
    lines.append("")

    lines.append("## 1. 各模型准确率\n")
    lines.append("### 1.1 Clip 级准确率（各自全量）")
    for model_name, v in summary["accuracies"]["clip_all"].items():
        lines.append(format_acc_line(model_name, v))
    lines.append("")

    lines.append("### 1.2 Frame 级准确率（各自全量）")
    for model_name, v in summary["accuracies"]["frame_all"].items():
        lines.append(format_acc_line(model_name, v))
    lines.append("")

    lines.append("### 1.3 Clip 级准确率（四模型公共 clip 集）")
    for model_name, v in summary["accuracies"]["clip_common_4way"].items():
        lines.append(format_acc_line(model_name, v))
    lines.append("")

    lines.append("### 1.4 Frame 级准确率（四模型公共 frame 集）")
    for model_name, v in summary["accuracies"]["frame_common_4way"].items():
        lines.append(format_acc_line(model_name, v))
    lines.append("")

    lines.append("### 1.5 Frame 级准确率（base / opsd / sft 公共 frame 集，主结论依据）")
    for model_name, v in summary["accuracies"]["frame_common_3way_main"].items():
        lines.append(format_acc_line(model_name, v))
    lines.append("")

    lines.append("## 2. initial model：reference-free vs reference-based（frame级）\n")
    base_ref = summary["comparisons"]["base_va_vs_base_ref"]
    lines.append(f"- common frames: {base_ref['num_common']}")
    for k, v in base_ref["counts"].items():
        lines.append(f"- {k}: {v} ({base_ref['ratios_on_common'][k]:.6f})")
    lines.append("")

    ref_help = summary["reference_helpfulness_main_3way"]
    lines.append("## 3. 在主结论样本集上，reference 对 initial model 的帮助分布\n")
    lines.append(f"- main 3-way frames: {ref_help['num_main_3way_keys']}")
    lines.append(f"- base_ref covered frames: {ref_help['num_base_ref_covered_keys']} "
                 f"({ref_help['base_ref_coverage_ratio_on_main_3way']:.6f})")
    for bucket_name, count in ref_help["bucket_counts"].items():
        lines.append(f"- {bucket_name}: {count} ({ref_help['bucket_ratios'][bucket_name]:.6f})")
    lines.append("")

    key_bucket = "ref_helpful_base_wrong_ref_right"
    if key_bucket in ref_help["per_bucket"]:
        key_bucket_info = ref_help["per_bucket"][key_bucket]
        lines.append("### 3.1 reference-helpful 子集上的修复能力")
        for model_name, acc_obj in key_bucket_info["accuracies"].items():
            lines.append(format_acc_line(model_name, acc_obj))
        repair = key_bucket_info["repair_on_base_wrong_subset"]
        lines.append(f"- sft repair: {repair['sft_repaired']}/{repair['base_wrong_count']} "
                     f"({repair['sft_repair_rate']:.6f})")
        lines.append(f"- opsd repair: {repair['opsd_repaired']}/{repair['base_wrong_count']} "
                     f"({repair['opsd_repair_rate']:.6f})")
        lines.append("")

        lines.append("### 3.2 reference-helpful 子集上的统计检验")
        for stat in key_bucket_info["paired_stats"].values():
            lines.append(format_stat_test_line(stat))
            lines.append(f"  结论：{summarize_stat_conclusion(stat)}")
        lines.append("")

    lines.append("## 4. sft va 相较于 base va（frame级）\n")
    sft_cmp = summary["comparisons"]["sft_vs_base_va"]
    lines.append(f"- common frames (main 3-way set): {sft_cmp['num_common']}")
    for k, v in sft_cmp["counts"].items():
        lines.append(f"- {k}: {v} ({sft_cmp['ratios_on_common'][k]:.6f})")
    lines.append("")

    lines.append(f"- net_gain = improved - regressed: {sft_cmp['counts']['improved'] - sft_cmp['counts']['regressed']}")
    lines.append("")

    if "improved_by_reference_bucket" in sft_cmp["breakdown"]:
        lines.append("### 4.1 在 SFT 相比 initial model 新增修复的样本中，base_ref 属于哪一类")
        total = sum(sft_cmp["breakdown"]["improved_by_reference_bucket"].values())
        for k, v in sorted_items_desc(sft_cmp["breakdown"]["improved_by_reference_bucket"]):
            lines.append(format_reference_bucket_line(k, v, safe_div(v, total)))
        lines.append("")

    if "regressed_by_reference_bucket" in sft_cmp["breakdown"]:
        lines.append("### 4.2 在 SFT 相比 initial model 新增退化的样本中，base_ref 属于哪一类")
        total = sum(sft_cmp["breakdown"]["regressed_by_reference_bucket"].values())
        for k, v in sorted_items_desc(sft_cmp["breakdown"]["regressed_by_reference_bucket"]):
            lines.append(format_reference_bucket_line(k, v, safe_div(v, total)))
        lines.append("")

    lines.append("## 5. opsd va 相较于 base va（frame级）\n")
    opsd_cmp = summary["comparisons"]["opsd_vs_base_va"]
    lines.append(f"- common frames (main 3-way set): {opsd_cmp['num_common']}")
    for k, v in opsd_cmp["counts"].items():
        lines.append(f"- {k}: {v} ({opsd_cmp['ratios_on_common'][k]:.6f})")
    lines.append("")

    lines.append(f"- net_gain = improved - regressed: {opsd_cmp['counts']['improved'] - opsd_cmp['counts']['regressed']}")
    lines.append("")

    if "improved_by_reference_bucket" in opsd_cmp["breakdown"]:
        lines.append("### 5.1 在 OPSD 相比 initial model 新增修复的样本中，base_ref 属于哪一类")
        total = sum(opsd_cmp["breakdown"]["improved_by_reference_bucket"].values())
        for k, v in sorted_items_desc(opsd_cmp["breakdown"]["improved_by_reference_bucket"]):
            lines.append(format_reference_bucket_line(k, v, safe_div(v, total)))
        lines.append("")

    if "regressed_by_reference_bucket" in opsd_cmp["breakdown"]:
        lines.append("### 5.2 在 OPSD 相比 initial model 新增退化的样本中，base_ref 属于哪一类")
        total = sum(opsd_cmp["breakdown"]["regressed_by_reference_bucket"].values())
        for k, v in sorted_items_desc(opsd_cmp["breakdown"]["regressed_by_reference_bucket"]):
            lines.append(format_reference_bucket_line(k, v, safe_div(v, total)))
        lines.append("")

    lines.append("## 6. opsd va vs sft va（frame级）\n")
    opsd_sft = summary["comparisons"]["opsd_vs_sft"]
    lines.append(f"- common frames (main 3-way set): {opsd_sft['num_common']}")
    for k, v in opsd_sft["counts"].items():
        lines.append(f"- {k}: {v} ({opsd_sft['ratios_on_common'][k]:.6f})")
    lines.append("")

    lines.append("## 7. 主结论口径\n")
    lines.append("- 下述 base/sft/opsd 对比与结论仅基于 base_va、opsd_va、sft_va 三者公共 frame 集。")
    lines.append("- base_ref 只用于 reference 对比与分桶分析，不参与主结论样本集定义。")
    lines.append("- 如果要判断 OPSD 是否学到了 reference 带来的增益，应优先看 `ref_helpful_base_wrong_ref_right` 子集。")
    lines.append("")

    lines.append("## 8. 统计检验（主结论样本集）\n")
    for stat in summary["statistical_tests"]["main_3way"].values():
        lines.append(format_stat_test_line(stat))
        lines.append(f"  结论：{summarize_stat_conclusion(stat)}")
    lines.append("")

    lines.append("## 9. 对齐信息\n")
    for k, v in summary["alignment"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def resolve_arg(cli_value: Optional[str], config_value: Optional[str], name: str) -> str:
    val = cli_value if cli_value not in [None, ""] else config_value
    if val in [None, ""]:
        raise ValueError(
            f"参数缺失: {name}\n"
            f"请通过命令行 --{name.replace('_', '-')} 传入，或在脚本顶部 CONFIG['{name}'] 中配置。"
        )
    return val


def main():
    parser = argparse.ArgumentParser(description="比较不同训练方法得到模型的性能效果")
    parser.add_argument("--base-va", default=None, help="base model va 推理结果 json")
    parser.add_argument("--base-ref", default=None, help="base model reference-prompt 推理结果 json")
    parser.add_argument("--opsd-va", default=None, help="opsd model va 推理结果 json")
    parser.add_argument("--sft-va", default=None, help="sft va 推理结果 json")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    args = parser.parse_args()

    base_va = resolve_arg(args.base_va, CONFIG.get("base_va"), "base_va")
    base_ref = resolve_arg(args.base_ref, CONFIG.get("base_ref"), "base_ref")
    opsd_va = resolve_arg(args.opsd_va, CONFIG.get("opsd_va"), "opsd_va")
    sft_va = resolve_arg(args.sft_va, CONFIG.get("sft_va"), "sft_va")
    output_dir_str = resolve_arg(args.output_dir, CONFIG.get("output_dir"), "output_dir")

    output_dir = Path(output_dir_str)
    ensure_dir(output_dir)
    detail_dir = output_dir / "details"
    ensure_dir(detail_dir)

    model_files = {
        "base_va": base_va,
        "base_ref": base_ref,
        "opsd_va": opsd_va,
        "sft_va": sft_va,
    }

    print("========== 实际生效参数 ==========")
    for k, v in model_files.items():
        print(f"{k:10s}: {v}")
    print(f"{'output_dir':10s}: {output_dir}")
    print("=================================\n")

    print("========== 1) 读取输入文件 ==========")
    raw_data = {}
    for model_name, path in model_files.items():
        print(f"[LOAD] {model_name}: {path}")
        raw_data[model_name] = load_json(path)

    print("\n========== 2) 构建 clip / frame map ==========")
    clip_maps = {}
    frame_maps = {}
    for model_name, data in raw_data.items():
        clip_maps[model_name] = build_clip_map(data, model_name)
        frame_maps[model_name] = build_frame_map(data, model_name)
        print(f"[OK] {model_name}: clips={len(clip_maps[model_name])}, frames={len(frame_maps[model_name])}")

    clip_acc_all = {m: calc_clip_accuracy(clip_maps[m]) for m in model_files}
    frame_acc_all = {m: calc_frame_accuracy(frame_maps[m]) for m in model_files}

    common_clip_keys = sorted(set.intersection(*(set(clip_maps[m].keys()) for m in model_files)))
    common_frame_keys = sorted(set.intersection(*(set(frame_maps[m].keys()) for m in model_files)))
    common_frame_keys_3way_main = sorted(
        set(frame_maps["base_va"].keys()) & set(frame_maps["opsd_va"].keys()) & set(frame_maps["sft_va"].keys()))

    clip_acc_common = {m: subset_accuracy_from_keys(clip_maps[m], common_clip_keys) for m in model_files}
    frame_acc_common = {m: subset_accuracy_from_keys(frame_maps[m], common_frame_keys) for m in model_files}
    frame_acc_common_3way_main = {
        m: subset_accuracy_from_keys(frame_maps[m], common_frame_keys_3way_main)
        for m in ["base_va", "opsd_va", "sft_va"]
    }

    alignment_info = {
        "common_clip_count_4way": len(common_clip_keys),
        "common_frame_count_4way": len(common_frame_keys),
        "common_frame_count_3way_main_base_opsd_sft": len(common_frame_keys_3way_main),
    }
    for m in model_files:
        alignment_info[f"{m}_clip_total"] = len(clip_maps[m])
        alignment_info[f"{m}_frame_total"] = len(frame_maps[m])

    print("\n========== 3) base_va vs base_ref ==========")
    cmp_base_ref = compare_pairwise_four_buckets(
        frame_maps["base_va"], frame_maps["base_ref"], "base_va", "base_ref"
    )
    save_pairwise_bucket_details(
        detail_dir / "base_va_vs_base_ref",
        cmp_base_ref,
        frame_maps["base_va"],
        frame_maps["base_ref"],
        "base_va",
        "base_ref",
    )

    print("========== 4) sft_va vs base_va ==========")
    cmp_sft_base = compare_target_vs_base(
        base_map=frame_maps["base_va"],
        target_map=frame_maps["sft_va"],
        base_name="base_va",
        target_name="sft_va",
        ref_map=frame_maps["base_ref"],
        ref_name="base_ref",
        compare_keys=common_frame_keys_3way_main,
    )
    save_target_vs_base_details(
        detail_dir / "sft_va_vs_base_va",
        cmp_sft_base,
        frame_maps["base_va"],
        frame_maps["sft_va"],
        "base_va",
        "sft_va",
        ref_map=frame_maps["base_ref"],
        ref_name="base_ref",
    )

    print("========== 5) opsd_va vs base_va ==========")
    cmp_opsd_base = compare_target_vs_base(
        base_map=frame_maps["base_va"],
        target_map=frame_maps["opsd_va"],
        base_name="base_va",
        target_name="opsd_va",
        ref_map=frame_maps["base_ref"],
        ref_name="base_ref",
        compare_keys=common_frame_keys_3way_main,
    )
    save_target_vs_base_details(
        detail_dir / "opsd_va_vs_base_va",
        cmp_opsd_base,
        frame_maps["base_va"],
        frame_maps["opsd_va"],
        "base_va",
        "opsd_va",
        ref_map=frame_maps["base_ref"],
        ref_name="base_ref",
    )

    print("========== 6) opsd_va vs sft_va ==========")
    cmp_opsd_sft = compare_pairwise_four_buckets(
        frame_maps["opsd_va"], frame_maps["sft_va"], "opsd_va", "sft_va", compare_keys=common_frame_keys_3way_main
    )
    save_pairwise_bucket_details(
        detail_dir / "opsd_va_vs_sft_va",
        cmp_opsd_sft,
        frame_maps["opsd_va"],
        frame_maps["sft_va"],
        "opsd_va",
        "sft_va",
    )

    reference_helpfulness_main_3way = analyze_reference_helpfulness_on_main_set(
        base_va_map=frame_maps["base_va"],
        base_ref_map=frame_maps["base_ref"],
        sft_map=frame_maps["sft_va"],
        opsd_map=frame_maps["opsd_va"],
        compare_keys=common_frame_keys_3way_main,
    )

    statistical_tests = {
        "main_3way": {
            "sft_vs_base_va": compare_pairwise_stats(
                left_map=frame_maps["base_va"],
                right_map=frame_maps["sft_va"],
                left_name="base_va",
                right_name="sft_va",
                compare_keys=common_frame_keys_3way_main,
            ),
            "opsd_vs_base_va": compare_pairwise_stats(
                left_map=frame_maps["base_va"],
                right_map=frame_maps["opsd_va"],
                left_name="base_va",
                right_name="opsd_va",
                compare_keys=common_frame_keys_3way_main,
            ),
            "opsd_vs_sft": compare_pairwise_stats(
                left_map=frame_maps["sft_va"],
                right_map=frame_maps["opsd_va"],
                left_name="sft_va",
                right_name="opsd_va",
                compare_keys=common_frame_keys_3way_main,
            ),
        }
    }

    summary = {
        "inputs": model_files,
        "alignment": alignment_info,
        "accuracies": {
            "clip_all": clip_acc_all,
            "frame_all": frame_acc_all,
            "clip_common_4way": clip_acc_common,
            "frame_common_4way": frame_acc_common,
            "frame_common_3way_main": frame_acc_common_3way_main,
        },
        "comparisons": {
            "base_va_vs_base_ref": cmp_base_ref,
            "sft_vs_base_va": cmp_sft_base,
            "opsd_vs_base_va": cmp_opsd_base,
            "opsd_vs_sft": cmp_opsd_sft,
        },
        "statistical_tests": statistical_tests,
        "reference_helpfulness_main_3way": reference_helpfulness_main_3way,
    }

    write_json(output_dir / "summary.json", summary)
    write_report_md(output_dir / "report.md", summary)

    print("\n========== DONE ==========")
    print(f"summary.json  : {output_dir / 'summary.json'}")
    print(f"report.md     : {output_dir / 'report.md'}")
    print(f"details dir   : {detail_dir}")

    print("\n====== 各模型 Frame 级准确率（全量） ======")
    for m, v in frame_acc_all.items():
        print(f"{m:10s} | correct={v['correct']:8d} | total={v['total']:8d} | acc={v['accuracy']:.6f}")

    print("\n====== 各模型 Clip 级准确率（全量） ======")
    for m, v in clip_acc_all.items():
        print(f"{m:10s} | correct={v['correct']:8d} | total={v['total']:8d} | acc={v['accuracy']:.6f}")

    print("\n====== base_va vs base_ref ======")
    for k, v in cmp_base_ref["counts"].items():
        print(f"{k:30s}: {v}")

    print("\n====== sft_va vs base_va ======")
    for k, v in cmp_sft_base["counts"].items():
        print(f"{k:30s}: {v}")

    print("\n====== sft improved_by_reference_bucket ======")
    if "improved_by_reference_bucket" in cmp_sft_base["breakdown"]:
        for k, v in sorted_items_desc(cmp_sft_base["breakdown"]["improved_by_reference_bucket"]):
            print(f"{k:30s}: {v}")

    print("\n====== opsd_va vs base_va ======")
    for k, v in cmp_opsd_base["counts"].items():
        print(f"{k:30s}: {v}")

    print("\n====== opsd improved_by_reference_bucket ======")
    if "improved_by_reference_bucket" in cmp_opsd_base["breakdown"]:
        for k, v in sorted_items_desc(cmp_opsd_base["breakdown"]["improved_by_reference_bucket"]):
            print(f"{k:30s}: {v}")

    print("\n====== opsd_va vs sft_va ======")
    for k, v in cmp_opsd_sft["counts"].items():
        print(f"{k:30s}: {v}")

    print("\n====== statistical tests on main 3-way set ======")
    for stat in statistical_tests["main_3way"].values():
        print(format_stat_test_line(stat))
        print(f"  结论：{summarize_stat_conclusion(stat)}")

    print("\n====== reference_helpful subset on main 3-way set ======")
    ref_helpful_bucket = reference_helpfulness_main_3way["per_bucket"].get("ref_helpful_base_wrong_ref_right", {})
    repair = ref_helpful_bucket.get("repair_on_base_wrong_subset", {})
    if repair:
        print(f"{'base_wrong_count':30s}: {repair['base_wrong_count']}")
        print(f"{'sft_repaired':30s}: {repair['sft_repaired']} ({repair['sft_repair_rate']:.6f})")
        print(f"{'opsd_repaired':30s}: {repair['opsd_repaired']} ({repair['opsd_repair_rate']:.6f})")
    paired_stats = ref_helpful_bucket.get("paired_stats", {})
    if paired_stats:
        print("\n====== statistical tests on reference_helpful subset ======")
        for stat in paired_stats.values():
            print(format_stat_test_line(stat))
            print(f"  结论：{summarize_stat_conclusion(stat)}")


if __name__ == "__main__":
    main()
