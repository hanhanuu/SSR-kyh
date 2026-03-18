#!/usr/bin/env python3
import argparse
import csv
import os
import re
from typing import Dict, List


FIELDS = [
    "Method",
    "Dataset",
    "Checkpoint_Standard",
    "L2_1s",
    "L2_2s",
    "L2_3s",
    "L2_Avg",
    "ObjCol_1s",
    "ObjCol_2s",
    "ObjCol_3s",
    "ObjCol_Avg",
    "ObjCol_Avg_pct",
    "BoxCol_1s",
    "BoxCol_2s",
    "BoxCol_3s",
    "BoxCol_Avg",
    "BoxCol_Avg_pct",
    "EMA_Status",
    "Source",
    "Comment",
]

KEY_MAP = {
    "plan_L2_1s": "L2_1s",
    "plan_L2_2s": "L2_2s",
    "plan_L2_3s": "L2_3s",
    "plan_obj_col_1s": "ObjCol_1s",
    "plan_obj_col_2s": "ObjCol_2s",
    "plan_obj_col_3s": "ObjCol_3s",
    "plan_obj_box_col_1s": "BoxCol_1s",
    "plan_obj_box_col_2s": "BoxCol_2s",
    "plan_obj_box_col_3s": "BoxCol_3s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append planning metrics row to unified CSV.")
    parser.add_argument("--csv", required=True, help="Unified CSV path.")
    parser.add_argument("--metrics", required=True, help="metrics.txt path.")
    parser.add_argument("--method", required=True, help="Method display name.")
    parser.add_argument("--dataset", required=True, help="Dataset display name, e.g. nuScenes/fog3.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint standard name, e.g. epoch_3_ema.")
    parser.add_argument("--source", default=None, help="Source field; defaults to metrics path.")
    parser.add_argument("--ema-status", default="已确认EMA", help="EMA status field.")
    parser.add_argument("--comment", default="", help="Comment field.")
    return parser.parse_args()


def parse_metrics(metrics_path: str) -> Dict[str, float]:
    metric_pattern = re.compile(
        r"^\s*(plan_[A-Za-z0-9_]+)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
    )
    raw: Dict[str, float] = {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            match = metric_pattern.match(line.strip())
            if not match:
                continue
            key = match.group(1)
            if key in KEY_MAP:
                raw[key] = float(match.group(2))

    missing = [k for k in KEY_MAP if k not in raw]
    if missing:
        raise ValueError(f"Missing keys in metrics file {metrics_path}: {missing}")

    parsed = {KEY_MAP[k]: v for k, v in raw.items()}
    parsed["L2_Avg"] = (parsed["L2_1s"] + parsed["L2_2s"] + parsed["L2_3s"]) / 3.0
    parsed["ObjCol_Avg"] = (parsed["ObjCol_1s"] + parsed["ObjCol_2s"] + parsed["ObjCol_3s"]) / 3.0
    parsed["BoxCol_Avg"] = (parsed["BoxCol_1s"] + parsed["BoxCol_2s"] + parsed["BoxCol_3s"]) / 3.0
    parsed["ObjCol_Avg_pct"] = parsed["ObjCol_Avg"] * 100.0
    parsed["BoxCol_Avg_pct"] = parsed["BoxCol_Avg"] * 100.0
    return parsed


def format_float(v: float) -> str:
    return f"{v:.6f}"


def read_existing_rows(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_rows(csv_path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def main() -> int:
    args = parse_args()
    metrics = parse_metrics(args.metrics)
    source = args.source if args.source else args.metrics

    new_row = {
        "Method": args.method,
        "Dataset": args.dataset,
        "Checkpoint_Standard": args.checkpoint,
        "L2_1s": format_float(metrics["L2_1s"]),
        "L2_2s": format_float(metrics["L2_2s"]),
        "L2_3s": format_float(metrics["L2_3s"]),
        "L2_Avg": format_float(metrics["L2_Avg"]),
        "ObjCol_1s": format_float(metrics["ObjCol_1s"]),
        "ObjCol_2s": format_float(metrics["ObjCol_2s"]),
        "ObjCol_3s": format_float(metrics["ObjCol_3s"]),
        "ObjCol_Avg": format_float(metrics["ObjCol_Avg"]),
        "ObjCol_Avg_pct": format_float(metrics["ObjCol_Avg_pct"]),
        "BoxCol_1s": format_float(metrics["BoxCol_1s"]),
        "BoxCol_2s": format_float(metrics["BoxCol_2s"]),
        "BoxCol_3s": format_float(metrics["BoxCol_3s"]),
        "BoxCol_Avg": format_float(metrics["BoxCol_Avg"]),
        "BoxCol_Avg_pct": format_float(metrics["BoxCol_Avg_pct"]),
        "EMA_Status": args.ema_status,
        "Source": source,
        "Comment": args.comment,
    }

    rows = read_existing_rows(args.csv)
    key = (
        new_row["Method"],
        new_row["Dataset"],
        new_row["Checkpoint_Standard"],
        new_row["Source"],
    )

    replaced = False
    for idx, row in enumerate(rows):
        old_key = (
            row.get("Method", ""),
            row.get("Dataset", ""),
            row.get("Checkpoint_Standard", ""),
            row.get("Source", ""),
        )
        if old_key == key:
            rows[idx] = new_row
            replaced = True
            break

    if not replaced:
        rows.append(new_row)

    write_rows(args.csv, rows)
    action = "updated" if replaced else "appended"
    print(f"{action} row in {args.csv}: {new_row['Method']} / {new_row['Dataset']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
