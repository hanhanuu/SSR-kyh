#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import re

import distutils

try:
    distutils.version
except Exception:
    try:
        distutils.version = importlib.import_module("setuptools._distutils.version")
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Write scalar metrics to TensorBoard.")
    p.add_argument("--log", help="Path to a log file containing lines like key:value")
    p.add_argument("--json", help="Path to a JSON file with {key: value} pairs")
    p.add_argument("--kv", nargs="+", help="Inline key=value pairs")
    p.add_argument("--outdir", required=True, help="TensorBoard output dir")
    p.add_argument("--step", type=int, default=0, help="Global step")
    p.add_argument("--tag-prefix", default="test/", help="Prefix for TB tags")
    return p.parse_args()


def parse_kv_list(kv_list):
    metrics = {}
    if not kv_list:
        return metrics
    for item in kv_list:
        if "=" not in item:
            raise ValueError(f"Invalid --kv item: {item} (expected key=value)")
        k, v = item.split("=", 1)
        metrics[k.strip()] = float(v.strip())
    return metrics


def parse_log(path):
    metrics = {}
    pattern = re.compile(
        r"^\s*([A-Za-z0-9_]+)\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue
            metrics[m.group(1)] = float(m.group(2))
    return metrics


def parse_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: float(v) for k, v in data.items()}


def main():
    args = parse_args()
    from torch.utils.tensorboard import SummaryWriter

    metrics = {}
    if args.log:
        metrics.update(parse_log(args.log))
    if args.json:
        metrics.update(parse_json(args.json))
    if args.kv:
        metrics.update(parse_kv_list(args.kv))

    if not metrics:
        raise SystemExit("No metrics found. Provide --log, --json, or --kv.")

    os.makedirs(args.outdir, exist_ok=True)
    writer = SummaryWriter(args.outdir)
    for k, v in metrics.items():
        writer.add_scalar(args.tag_prefix + k, v, args.step)
    writer.close()
    print(f"TB written to: {args.outdir} ({len(metrics)} metrics)")


if __name__ == "__main__":
    main()
