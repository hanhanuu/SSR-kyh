#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


KEYS = (
    "plan_L2_1s",
    "plan_L2_2s",
    "plan_L2_3s",
    "plan_obj_col_1s",
    "plan_obj_col_2s",
    "plan_obj_col_3s",
    "plan_obj_box_col_1s",
    "plan_obj_box_col_2s",
    "plan_obj_box_col_3s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select best planning checkpoint by combined score on "
            "L2 / point collision / box collision."
        )
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Training work_dir. Used to scan metrics and infer checkpoint paths.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="eval_nuscenes_ep*ema*/metrics.txt,metrics_epoch*.txt",
        help=(
            "Glob pattern(s) under work-dir to find metrics files. "
            "Use comma to separate multiple patterns."
        ),
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="Explicit metrics.txt paths. If provided, work-dir/glob scan is skipped.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="0.4,0.3,0.3",
        help="Weights for L2,ObjCol,BoxCol in composite score, e.g. 0.4,0.3,0.3",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Print top-k candidates by combined score.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to dump ranking csv.",
    )
    return parser.parse_args()


def parse_weight_string(weights: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in weights.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("weights must have exactly 3 comma-separated numbers")
    vals = tuple(float(x) for x in parts)
    if any(v < 0 for v in vals):
        raise ValueError("weights must be non-negative")
    s = sum(vals)
    if s <= 0:
        raise ValueError("sum(weights) must be > 0")
    return tuple(v / s for v in vals)


def discover_metric_files(work_dir: Optional[str], pattern: str, explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        files = [os.path.abspath(p) for p in explicit]
    else:
        if not work_dir:
            raise ValueError("Either --metrics or --work-dir must be provided")
        files = []
        patterns = [p.strip() for p in pattern.split(",") if p.strip()]
        if not patterns:
            raise ValueError("No valid --glob pattern")
        for p in patterns:
            files.extend(glob.glob(os.path.join(work_dir, p)))
    files = [f for f in files if os.path.isfile(f)]
    return sorted(set(files))


def parse_metric_file(path: str) -> Dict[str, float]:
    data: Dict[str, float] = {}
    pattern = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            m = pattern.match(raw.strip())
            if not m:
                continue
            key = m.group(1)
            val = float(m.group(2))
            if key in KEYS:
                data[key] = val
    missing = [k for k in KEYS if k not in data]
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")
    data["L2_avg"] = (data["plan_L2_1s"] + data["plan_L2_2s"] + data["plan_L2_3s"]) / 3.0
    data["ObjCol_avg"] = (
        data["plan_obj_col_1s"] + data["plan_obj_col_2s"] + data["plan_obj_col_3s"]
    ) / 3.0
    data["BoxCol_avg"] = (
        data["plan_obj_box_col_1s"] + data["plan_obj_box_col_2s"] + data["plan_obj_box_col_3s"]
    ) / 3.0
    return data


def infer_epoch_and_ema(path: str) -> Tuple[Optional[int], Optional[bool]]:
    s = path.lower().replace("\\", "/")
    patterns = [
        (re.compile(r"ep(?:och)?[_-]?(\d+)_?ema"), True),
        (re.compile(r"epoch[_-]?(\d+)_?ema"), True),
        (re.compile(r"ep(?:och)?[_-]?(\d+)"), False),
        (re.compile(r"epoch[_-]?(\d+)"), False),
    ]
    for regex, is_ema in patterns:
        m = regex.search(s)
        if m:
            return int(m.group(1)), is_ema
    return None, None


def infer_checkpoint_path(work_dir: Optional[str], metric_path: str) -> Optional[str]:
    if not work_dir:
        return None
    epoch, is_ema = infer_epoch_and_ema(metric_path)
    if epoch is None:
        return None
    candidates = []
    if is_ema is True:
        candidates.extend(
            [
                os.path.join(work_dir, f"epoch_{epoch}_ema.pth"),
                os.path.join(work_dir, f"epoch_{epoch}.pth"),
            ]
        )
    else:
        candidates.extend(
            [
                os.path.join(work_dir, f"epoch_{epoch}.pth"),
                os.path.join(work_dir, f"epoch_{epoch}_ema.pth"),
            ]
        )
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def normalize_minmax(values: List[float]) -> List[float]:
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def pareto_front(items: List[Dict[str, float]]) -> List[bool]:
    """Return mask for non-dominated points (minimize all objectives)."""
    keys = ("L2_avg", "ObjCol_avg", "BoxCol_avg")
    n = len(items)
    on_front = [True] * n
    for i in range(n):
        if not on_front[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            no_worse = all(items[j][k] <= items[i][k] for k in keys)
            strictly_better = any(items[j][k] < items[i][k] for k in keys)
            if no_worse and strictly_better:
                on_front[i] = False
                break
    return on_front


def print_table(rows: List[Dict[str, object]], topk: int) -> None:
    show = rows[: max(topk, 1)]
    header = (
        f"{'Rank':<5} {'Score':<10} {'L2_avg':<10} {'ObjCol_avg':<12} "
        f"{'BoxCol_avg':<12} {'Pareto':<8} {'MetricFile':<50}"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(show, 1):
        metric_file = str(r["metric_file"])
        if len(metric_file) > 50:
            metric_file = "..." + metric_file[-47:]
        print(
            f"{i:<5} {r['score']:<10.6f} {r['L2_avg']:<10.6f} "
            f"{r['ObjCol_avg']:<12.6f} {r['BoxCol_avg']:<12.6f} "
            f"{str(r['pareto']):<8} {metric_file:<50}"
        )


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fields = [
        "rank",
        "score",
        "pareto",
        "L2_avg",
        "ObjCol_avg",
        "BoxCol_avg",
        "metric_file",
        "checkpoint",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            out = dict(r)
            out["rank"] = i
            w.writerow({k: out.get(k, "") for k in fields})


def main() -> int:
    args = parse_args()
    try:
        wl2, wobj, wbox = parse_weight_string(args.weights)
    except ValueError as e:
        print(f"Invalid --weights: {e}", file=sys.stderr)
        return 2

    try:
        files = discover_metric_files(args.work_dir, args.glob, args.metrics)
    except ValueError as e:
        print(f"Argument error: {e}", file=sys.stderr)
        return 2

    if not files:
        print("No metrics files found.", file=sys.stderr)
        return 1

    rows: List[Dict[str, object]] = []
    parse_errors = []
    work_dir_abs = os.path.abspath(args.work_dir) if args.work_dir else None
    for f in files:
        try:
            m = parse_metric_file(f)
        except Exception as e:  # pylint: disable=broad-except
            parse_errors.append((f, str(e)))
            continue
        rows.append(
            {
                "metric_file": os.path.abspath(f),
                "checkpoint": infer_checkpoint_path(work_dir_abs, f),
                "L2_avg": m["L2_avg"],
                "ObjCol_avg": m["ObjCol_avg"],
                "BoxCol_avg": m["BoxCol_avg"],
            }
        )

    if parse_errors:
        for fp, err in parse_errors:
            print(f"[WARN] skip {fp}: {err}", file=sys.stderr)

    if not rows:
        print("No valid metrics files after parsing.", file=sys.stderr)
        return 1

    l2_norm = normalize_minmax([float(r["L2_avg"]) for r in rows])
    obj_norm = normalize_minmax([float(r["ObjCol_avg"]) for r in rows])
    box_norm = normalize_minmax([float(r["BoxCol_avg"]) for r in rows])
    for i, r in enumerate(rows):
        r["score"] = wl2 * l2_norm[i] + wobj * obj_norm[i] + wbox * box_norm[i]

    pareto_mask = pareto_front([{"L2_avg": float(r["L2_avg"]), "ObjCol_avg": float(r["ObjCol_avg"]), "BoxCol_avg": float(r["BoxCol_avg"])} for r in rows])
    for i, r in enumerate(rows):
        r["pareto"] = pareto_mask[i]

    # Sort by score; then favor lower BoxCol and lower L2 as tie-breakers.
    rows.sort(
        key=lambda r: (
            float(r["score"]),
            float(r["BoxCol_avg"]),
            float(r["L2_avg"]),
            float(r["ObjCol_avg"]),
        )
    )

    print(
        f"weights(normalized): L2={wl2:.3f}, ObjCol={wobj:.3f}, BoxCol={wbox:.3f}\n"
        f"candidates: {len(rows)}"
    )
    print_table(rows, args.topk)

    best = rows[0]
    print("\nBest candidate:")
    print(f"  metric_file : {best['metric_file']}")
    print(f"  checkpoint  : {best['checkpoint'] or 'N/A (cannot infer from path)'}")
    print(f"  score       : {best['score']:.6f}")
    print(
        "  metrics     : "
        f"L2_avg={best['L2_avg']:.6f}, "
        f"ObjCol_avg={best['ObjCol_avg']:.6f}, "
        f"BoxCol_avg={best['BoxCol_avg']:.6f}, "
        f"Pareto={best['pareto']}"
    )

    if args.output_csv:
        out_path = os.path.abspath(args.output_csv)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_csv(out_path, rows)
        print(f"\nSaved ranking csv: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
