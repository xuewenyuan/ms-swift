#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
import argparse
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


def write_report_md(path: Path, summary: Dict[str, Any]):
    lines = []

    lines.append("# 模型评测对比报告\n")

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

    lines.append("## 2. base model: va vs reference-prompt（frame级）\n")
    base_ref = summary["comparisons"]["base_va_vs_base_ref"]
    lines.append(f"- common frames: {base_ref['num_common']}")
    for k, v in base_ref["counts"].items():
        lines.append(f"- {k}: {v} ({base_ref['ratios_on_common'][k]:.6f})")
    lines.append("")

    lines.append("## 3. sft va 相较于 base va（frame级）\n")
    sft_cmp = summary["comparisons"]["sft_vs_base_va"]
    lines.append(f"- common frames (main 3-way set): {sft_cmp['num_common']}")
    for k, v in sft_cmp["counts"].items():
        lines.append(f"- {k}: {v} ({sft_cmp['ratios_on_common'][k]:.6f})")
    lines.append("")

    lines.append(f"- net_gain = improved - regressed: {sft_cmp['counts']['improved'] - sft_cmp['counts']['regressed']}")
    lines.append("")

    if "improved_by_reference_bucket" in sft_cmp["breakdown"]:
        lines.append("### 3.1 SFT 提升属于哪一部分（结合 base reference）")
        total = sum(sft_cmp["breakdown"]["improved_by_reference_bucket"].values())
        for k, v in sorted_items_desc(sft_cmp["breakdown"]["improved_by_reference_bucket"]):
            lines.append(f"- {k}: {v} ({safe_div(v, total):.6f})")
        lines.append("")

    if "regressed_by_reference_bucket" in sft_cmp["breakdown"]:
        lines.append("### 3.2 SFT 退化属于哪一部分（结合 base reference）")
        total = sum(sft_cmp["breakdown"]["regressed_by_reference_bucket"].values())
        for k, v in sorted_items_desc(sft_cmp["breakdown"]["regressed_by_reference_bucket"]):
            lines.append(f"- {k}: {v} ({safe_div(v, total):.6f})")
        lines.append("")

    lines.append("## 4. opsd va 相较于 base va（frame级）\n")
    opsd_cmp = summary["comparisons"]["opsd_vs_base_va"]
    lines.append(f"- common frames (main 3-way set): {opsd_cmp['num_common']}")
    for k, v in opsd_cmp["counts"].items():
        lines.append(f"- {k}: {v} ({opsd_cmp['ratios_on_common'][k]:.6f})")
    lines.append("")

    lines.append(f"- net_gain = improved - regressed: {opsd_cmp['counts']['improved'] - opsd_cmp['counts']['regressed']}")
    lines.append("")

    if "improved_by_reference_bucket" in opsd_cmp["breakdown"]:
        lines.append("### 4.1 OPSD 提升属于哪一部分（结合 base reference）")
        total = sum(opsd_cmp["breakdown"]["improved_by_reference_bucket"].values())
        for k, v in sorted_items_desc(opsd_cmp["breakdown"]["improved_by_reference_bucket"]):
            lines.append(f"- {k}: {v} ({safe_div(v, total):.6f})")
        lines.append("")

    if "regressed_by_reference_bucket" in opsd_cmp["breakdown"]:
        lines.append("### 4.2 OPSD 退化属于哪一部分（结合 base reference）")
        total = sum(opsd_cmp["breakdown"]["regressed_by_reference_bucket"].values())
        for k, v in sorted_items_desc(opsd_cmp["breakdown"]["regressed_by_reference_bucket"]):
            lines.append(f"- {k}: {v} ({safe_div(v, total):.6f})")
        lines.append("")

    lines.append("## 5. opsd va vs sft va（frame级）\n")
    opsd_sft = summary["comparisons"]["opsd_vs_sft"]
    lines.append(f"- common frames (main 3-way set): {opsd_sft['num_common']}")
    for k, v in opsd_sft["counts"].items():
        lines.append(f"- {k}: {v} ({opsd_sft['ratios_on_common'][k]:.6f})")
    lines.append("")

    lines.append("## 6. 主结论口径\n")
    lines.append("- 下述 base/sft/opsd 对比与结论仅基于 base_va、opsd_va、sft_va 三者公共 frame 集。")
    lines.append("- base_ref 只用于 reference 对比与分桶分析，不参与主结论样本集定义。")
    lines.append("")

    lines.append("## 7. 对齐信息\n")
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
        }
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


if __name__ == "__main__":
    main()
