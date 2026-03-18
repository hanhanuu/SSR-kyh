#!/usr/bin/env python3
"""
Create nuScenes *info PKLs for corruption robustness evaluation.

This script rewrites sensor file paths inside an existing info PKL (typically
generated from clean nuScenes) so that dataloaders read corrupted sensor files,
while keeping tokens and annotations unchanged.

Two rewrite modes are supported:
  - anchor (default): replace the path prefix up to 'samples/' or 'sweeps/'.
  - prefix: replace an explicit old prefix with a new prefix.

Example (anchor mode):
  python tools/create_nuscenes_corruption_infos.py \\
    --src data/nuscenes/vad_nuscenes_infos_temporal_val.pkl \\
    --dst data/corruptions/vad_nuscenes_infos_temporal_val_fog3.pkl \\
    --mode anchor \\
    --new-root ./data/nuscenes_corruptions/fog/3 \\
    --check 50
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple


DEFAULT_ANCHORS = ("samples/", "sweeps/")


def _norm_root(root: str) -> str:
    root = root.replace("\\", "/")
    return root.rstrip("/") + "/"


def _rewrite_anchor(path: str, *, new_root: str, anchors: Tuple[str, ...]) -> str:
    if not isinstance(path, str):
        return path
    p = path.replace("\\", "/")
    for anchor in anchors:
        idx = p.find(anchor)
        if idx != -1:
            suffix = p[idx:]
            return new_root + suffix
    return path


def _candidate_prefixes(old_prefix: str) -> Tuple[str, ...]:
    old_prefix = old_prefix.replace("\\", "/")
    old_prefix = old_prefix.rstrip("/") + "/"
    if old_prefix.startswith("./"):
        return (old_prefix, old_prefix[2:])
    return (old_prefix, "./" + old_prefix)


def _rewrite_prefix(path: str, *, old_prefixes: Tuple[str, ...], new_prefix: str) -> str:
    if not isinstance(path, str):
        return path
    p = path.replace("\\", "/")
    for old in old_prefixes:
        if p.startswith(old):
            return new_prefix + p[len(old):]
    return path


def _rewrite_str_list(items: Any, rewrite_fn: Callable[[str], str]) -> int:
    if not isinstance(items, list):
        return 0
    changed = 0
    for i, v in enumerate(items):
        if not isinstance(v, str):
            continue
        new_v = rewrite_fn(v)
        if new_v != v:
            items[i] = new_v
            changed += 1
    return changed


def _extract_infos(payload: Any) -> Tuple[list, dict | None]:
    if isinstance(payload, dict) and "infos" in payload:
        infos = payload["infos"]
        if not isinstance(infos, list):
            raise TypeError(f"payload['infos'] must be a list, got {type(infos)}")
        return infos, payload
    if isinstance(payload, list):
        return payload, None
    raise TypeError(f"Unsupported PKL format: {type(payload)} (expect dict with 'infos' or list)")


def _collect_sensor_paths(sample: dict, rewrite_set: set[str]) -> Iterable[str]:
    if "lidar" in rewrite_set:
        lp = sample.get("lidar_path")
        if isinstance(lp, str):
            yield lp
        lp_adj = sample.get("lidar_path_adj")
        if isinstance(lp_adj, str):
            yield lp_adj
        pts = sample.get("pts_filename")
        if isinstance(pts, str):
            yield pts

    if "camera" in rewrite_set:
        cams = sample.get("cams")
        if isinstance(cams, dict):
            for cam_info in cams.values():
                if not isinstance(cam_info, dict):
                    continue
                dp = cam_info.get("data_path")
                if isinstance(dp, str):
                    yield dp
        img_filenames = sample.get("img_filename")
        if isinstance(img_filenames, list):
            for p in img_filenames:
                if isinstance(p, str):
                    yield p

    if "sweeps" in rewrite_set:
        sweeps = sample.get("sweeps")
        if isinstance(sweeps, list):
            for sw in sweeps:
                if not isinstance(sw, dict):
                    continue
                for v in sw.values():
                    if isinstance(v, str) and any(a in v for a in DEFAULT_ANCHORS):
                        yield v


def _exists(path: str) -> bool:
    return os.path.exists(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite nuScenes info PKL sensor paths for corruption evaluation."
    )
    parser.add_argument("--src", required=True, help="Source (clean) info PKL.")
    parser.add_argument("--dst", required=True, help="Destination info PKL.")
    parser.add_argument(
        "--mode",
        choices=("anchor", "prefix"),
        default="anchor",
        help="Rewrite mode: anchor (recommended) or prefix.",
    )
    parser.add_argument(
        "--new-root",
        required=True,
        help="New dataset root to prepend (should contain samples/ and/or sweeps/).",
    )
    parser.add_argument(
        "--anchors",
        nargs="+",
        default=list(DEFAULT_ANCHORS),
        help="Anchors used by anchor mode (default: samples/ sweeps/).",
    )
    parser.add_argument(
        "--rewrite",
        nargs="+",
        choices=("all", "lidar", "camera", "sweeps"),
        default=["all"],
        help="Which sensor paths to rewrite (default: all).",
    )
    parser.add_argument(
        "--old-prefix",
        default=None,
        help="Old prefix to replace (required when --mode prefix).",
    )
    parser.add_argument(
        "--check",
        type=int,
        default=0,
        help="Check existence of rewritten sensor files for N samples (0 disables).",
    )
    parser.add_argument(
        "--check-random",
        action="store_true",
        help="Randomly sample N items for --check instead of taking the first N.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --check-random.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not fail when some rewritten files are missing (still prints them).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write output PKL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    if args.mode == "prefix" and not args.old_prefix:
        raise SystemExit("--old-prefix is required when --mode prefix")

    new_root = _norm_root(args.new_root)
    anchors = tuple(a if a.endswith("/") else a + "/" for a in args.anchors)

    payload = pickle.load(open(src, "rb"))
    infos, payload_dict = _extract_infos(payload)

    rewrite_set = set(args.rewrite)
    if "all" in rewrite_set:
        rewrite_set.update({"lidar", "camera", "sweeps"})
        rewrite_set.remove("all")

    if args.mode == "anchor":
        rewrite_fn = lambda s: _rewrite_anchor(s, new_root=new_root, anchors=anchors)  # noqa: E731
        rewrite_desc = f"anchor({anchors}) -> {new_root}"
    else:
        old_prefixes = _candidate_prefixes(args.old_prefix)
        new_prefix = _norm_root(args.new_root)
        rewrite_fn = lambda s: _rewrite_prefix(s, old_prefixes=old_prefixes, new_prefix=new_prefix)  # noqa: E731
        rewrite_desc = f"prefix({old_prefixes}) -> {new_prefix}"

    total_changed = 0
    for i, sample in enumerate(infos):
        if not isinstance(sample, dict):
            raise TypeError(f"infos[{i}] must be dict, got {type(sample)}")

        changed = 0
        if "lidar" in rewrite_set:
            for k in ("lidar_path", "lidar_path_adj", "pts_filename"):
                v = sample.get(k)
                if not isinstance(v, str):
                    continue
                new_v = rewrite_fn(v)
                if new_v != v:
                    sample[k] = new_v
                    changed += 1

        if "camera" in rewrite_set:
            cams = sample.get("cams")
            if isinstance(cams, dict):
                for cam_info in cams.values():
                    if not isinstance(cam_info, dict):
                        continue
                    v = cam_info.get("data_path")
                    if not isinstance(v, str):
                        continue
                    new_v = rewrite_fn(v)
                    if new_v != v:
                        cam_info["data_path"] = new_v
                        changed += 1

            changed += _rewrite_str_list(sample.get("img_filename"), rewrite_fn)

        if "sweeps" in rewrite_set:
            sweeps = sample.get("sweeps")
            if isinstance(sweeps, list):
                for sw in sweeps:
                    if not isinstance(sw, dict):
                        continue
                    for k, v in list(sw.items()):
                        if not isinstance(v, str):
                            continue
                        new_v = rewrite_fn(v)
                        if new_v != v:
                            sw[k] = new_v
                            changed += 1

        total_changed += changed

    print(f"Loaded {len(infos)} samples from: {src}")
    print(f"Rewrite rule: {rewrite_desc}")
    print(f"Rewrite targets: {sorted(rewrite_set)}")
    print(f"Total rewritten string fields: {total_changed}")

    if args.check and args.check > 0:
        n = min(args.check, len(infos))
        indices = list(range(len(infos)))
        if args.check_random:
            random.seed(args.seed)
            indices = random.sample(indices, n)
        else:
            indices = indices[:n]

        missing = []
        for idx in indices:
            sample = infos[idx]
            for p in _collect_sensor_paths(sample, rewrite_set):
                if any(a in p for a in DEFAULT_ANCHORS) and not _exists(p):
                    missing.append((idx, p))

        if missing:
            print(f"[WARN] Missing files after rewrite: {len(missing)}")
            for idx, p in missing[:50]:
                print(f"  - sample[{idx}] {p}")
            if len(missing) > 50:
                print(f"  ... ({len(missing)-50} more)")
            if not args.allow_missing:
                print("Failing due to missing files. Use --allow-missing to ignore.")
                return 2
        else:
            print(f"[OK] Existence check passed for {n} samples.")

    if args.dry_run:
        print("Dry-run: not writing output PKL.")
        return 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    if payload_dict is not None:
        payload_dict["infos"] = infos
        out_payload = payload_dict
    else:
        out_payload = infos

    with open(dst, "wb") as f:
        pickle.dump(out_payload, f)
    print(f"Wrote: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
