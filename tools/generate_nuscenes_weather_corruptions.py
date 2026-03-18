#!/usr/bin/env python3
"""
Generate a nuScenes-like folder containing corrupted camera images (val split).

Why this script exists:
- Our VAD-style info PKLs store *absolute file paths* (e.g. ./data/nuscenes/samples/...).
- For robustness evaluation, it is convenient to materialize corrupted images
  under a new root (e.g. data/nuscenes_corruptions/fog/3/samples/...).
- Then use tools/create_nuscenes_corruption_infos.py to rewrite info PKL paths
  to this new root, without touching tokens/GT.

This script does NOT modify point clouds. It only writes corrupted images.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


NU_SCENES_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_infos(pkl_path: Path) -> List[dict]:
    obj = pickle.load(open(pkl_path, "rb"))
    if isinstance(obj, dict) and "infos" in obj:
        infos = obj["infos"]
    else:
        infos = obj
    if not isinstance(infos, list):
        raise TypeError(f"Unsupported info PKL format: {type(obj)}")
    return infos


def _norm_path(p: str) -> str:
    return p.replace("\\", "/")


def _anchor_suffix(path: str, anchor: str = "samples/") -> Optional[str]:
    path = _norm_path(path)
    idx = path.find(anchor)
    if idx == -1:
        return None
    return path[idx:]


def _rng_for(path: str, seed: int) -> np.random.Generator:
    h = hashlib.md5((str(seed) + "|" + path).encode("utf-8")).digest()
    # 64-bit seed
    s = int.from_bytes(h[:8], "little", signed=False)
    return np.random.default_rng(s)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _fog(img_bgr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    severity = int(severity)
    if severity < 1 or severity > 5:
        raise ValueError("severity must be in [1, 5]")

    alpha_base = {1: 0.18, 2: 0.28, 3: 0.38, 4: 0.48, 5: 0.58}[severity]
    contrast = {1: 0.95, 2: 0.90, 3: 0.85, 4: 0.80, 5: 0.75}[severity]
    blur_sigma = {1: 0.4, 2: 0.7, 3: 1.0, 4: 1.3, 5: 1.6}[severity]

    h, w = img_bgr.shape[:2]
    img = img_bgr.astype(np.float32) / 255.0

    # Low-res noise upsampled to create smooth fog patches.
    grid = {1: 64, 2: 64, 3: 48, 4: 32, 5: 24}[severity]
    nh = max(2, h // grid)
    nw = max(2, w // grid)
    small = rng.random((nh, nw), dtype=np.float32)
    noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=severity * 3.0, sigmaY=severity * 3.0)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)

    alpha = (alpha_base * noise).astype(np.float32)
    alpha = alpha[..., None]

    # Airlight (slightly gray-white).
    air = np.array([0.92, 0.92, 0.92], dtype=np.float32)
    out = img * (1.0 - alpha) + air * alpha

    mean = out.mean(axis=(0, 1), keepdims=True)
    out = mean + (out - mean) * contrast
    out = cv2.GaussianBlur(out, (0, 0), blur_sigma)
    out = (_clip01(out) * 255.0).astype(np.uint8)
    return out


def _rain(img_bgr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    severity = int(severity)
    if severity < 1 or severity > 5:
        raise ValueError("severity must be in [1, 5]")

    h, w = img_bgr.shape[:2]
    img = img_bgr.astype(np.float32)

    # Number of streaks roughly proportional to area.
    base = {1: 700, 2: 1200, 3: 1800, 4: 2600, 5: 3600}[severity]
    num = int(base * (h * w) / (1600 * 900))
    num = max(200, num)

    overlay = np.zeros((h, w), dtype=np.float32)
    # Slight slant angle (radians).
    angle = rng.uniform(-0.25, 0.25) + rng.choice([-1.0, 1.0]) * 0.10
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))

    min_len = max(12, h // 80)
    max_len = max(min_len + 1, h // 35)

    for _ in range(num):
        x1 = int(rng.integers(0, w))
        y1 = int(rng.integers(0, h))
        length = int(rng.integers(min_len, max_len))
        x2 = int(x1 + length * sin_a)
        y2 = int(y1 + length * cos_a)
        thickness = int(rng.integers(1, 2 + (severity >= 4)))
        cv2.line(overlay, (x1, y1), (x2, y2), color=1.0, thickness=thickness)

    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=0.7 + 0.2 * severity, sigmaY=0.7 + 0.2 * severity)
    overlay = np.clip(overlay, 0.0, 1.0)

    alpha = {1: 0.10, 2: 0.14, 3: 0.18, 4: 0.22, 5: 0.26}[severity]
    darken = {1: 0.98, 2: 0.95, 3: 0.92, 4: 0.90, 5: 0.88}[severity]

    out = img * darken + overlay[..., None] * (255.0 * alpha)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out


def _snow(img_bgr: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    severity = int(severity)
    if severity < 1 or severity > 5:
        raise ValueError("severity must be in [1, 5]")

    h, w = img_bgr.shape[:2]
    img = img_bgr.astype(np.float32) / 255.0

    base = {1: 600, 2: 1000, 3: 1500, 4: 2200, 5: 3200}[severity]
    num = int(base * (h * w) / (1600 * 900))
    num = max(200, num)

    overlay = np.zeros((h, w), dtype=np.float32)
    min_r = 1
    max_r = 1 + severity
    for _ in range(num):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(min_r, max_r + 1))
        cv2.circle(overlay, (x, y), r, color=1.0, thickness=-1)

    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=0.6 + 0.2 * severity, sigmaY=0.6 + 0.2 * severity)
    overlay = np.clip(overlay, 0.0, 1.0)

    alpha = {1: 0.18, 2: 0.24, 3: 0.30, 4: 0.36, 5: 0.42}[severity]
    out = img + overlay[..., None] * alpha
    out = _clip01(out)
    out = (out * 255.0).astype(np.uint8)
    return out


def _apply_corruption(img_bgr: np.ndarray, corruption: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    corruption = corruption.lower()
    if corruption == "fog":
        return _fog(img_bgr, severity, rng)
    if corruption == "rain":
        return _rain(img_bgr, severity, rng)
    if corruption == "snow":
        return _snow(img_bgr, severity, rng)
    raise ValueError(f"Unsupported corruption: {corruption} (choose from fog/rain/snow)")


@dataclass(frozen=True)
class Task:
    src: str
    dst: str


def _iter_image_tasks(
    infos: Sequence[dict],
    out_root: Path,
    cameras: Optional[Sequence[str]],
    overwrite: bool,
    max_images: int,
) -> List[Task]:
    allow_cams = set(cameras) if cameras else None
    seen: set[str] = set()
    tasks: List[Task] = []

    for info in infos:
        cams = info.get("cams", {})
        if not isinstance(cams, dict):
            continue
        for cam_name, cam_info in cams.items():
            if allow_cams is not None and cam_name not in allow_cams:
                continue
            if not isinstance(cam_info, dict):
                continue
            src = cam_info.get("data_path", None)
            if not isinstance(src, str):
                continue
            src = _norm_path(src)
            if src in seen:
                continue
            seen.add(src)

            suffix = _anchor_suffix(src, anchor="samples/")
            if suffix is None:
                continue
            dst = _norm_path(str(out_root / suffix))
            if (not overwrite) and os.path.exists(dst):
                continue
            tasks.append(Task(src=src, dst=dst))

            if max_images > 0 and len(tasks) >= max_images:
                return tasks

    return tasks


def _resolve_src(src: str) -> str:
    p = Path(src)
    if p.is_absolute():
        return str(p)
    root = _repo_root()
    return str((root / p).resolve())


def _resolve_dst(dst: str) -> str:
    p = Path(dst)
    if p.is_absolute():
        return str(p)
    root = _repo_root()
    return str((root / p).resolve())


def _process_one(args: Tuple[Task, str, int, int, bool]) -> Tuple[bool, str]:
    task, corruption, severity, seed, skip_missing = args
    src = _resolve_src(task.src)
    dst = _resolve_dst(task.dst)

    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        if skip_missing:
            return True, f"missing_src: {src}"
        return False, f"missing_src: {src}"

    rng = _rng_for(task.src, seed)
    out = _apply_corruption(img, corruption, severity, rng)

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(dst, out)
    if not ok:
        return False, f"write_fail: {dst}"
    return True, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate corrupted nuScenes camera images for evaluation.")
    parser.add_argument(
        "--info",
        required=True,
        help="Info PKL (e.g. data/nuscenes/vad_nuscenes_infos_temporal_val.pkl).",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Output root that will contain samples/CAM_*/... (e.g. data/nuscenes_corruptions/fog/3).",
    )
    parser.add_argument("--corruption", required=True, choices=["fog", "rain", "snow"])
    parser.add_argument("--severity", required=True, type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=list(NU_SCENES_CAMERAS),
        help="Camera names to process (default: all nuScenes cameras).",
    )
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers (I/O bound, default: 16).")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed (default: 0).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output images.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing src images instead of failing.")
    parser.add_argument("--max-images", type=int, default=0, help="Debug: only process first N images (0=all).")
    parser.add_argument("--dry-run", action="store_true", help="Only print task count; do not write images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    info_path = Path(args.info)
    out_root = Path(args.out_root)

    infos = _load_infos(info_path)
    tasks = _iter_image_tasks(
        infos=infos,
        out_root=out_root,
        cameras=args.cameras,
        overwrite=bool(args.overwrite),
        max_images=int(args.max_images),
    )

    print(f"Loaded infos: {info_path} (frames={len(infos)})")
    print(f"Corruption: {args.corruption} severity={args.severity}")
    print(f"Output root: {out_root}")
    print(f"Tasks (unique images to write): {len(tasks)}")
    if args.dry_run:
        return 0

    if len(tasks) == 0:
        print("No tasks to run (maybe outputs already exist and --overwrite not set).")
        return 0

    worker_args = [(t, args.corruption, args.severity, args.seed, bool(args.skip_missing)) for t in tasks]

    ok_count = 0
    errors: List[str] = []

    # NOTE: Some environments restrict POSIX semaphores (multiprocessing Pool may fail).
    # Thread pool is usually good enough here because this job is I/O bound.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        for ok, msg in tqdm(ex.map(_process_one, worker_args), total=len(worker_args), ncols=80):
            if ok:
                ok_count += 1
            if msg:
                errors.append(msg)

    print(f"Done. ok={ok_count}/{len(tasks)}")
    if errors:
        print(f"Warnings/Errors: {len(errors)} (showing first 30)")
        for e in errors[:30]:
            print("  -", e)
        if any(e.startswith("missing_src:") or e.startswith("write_fail:") for e in errors):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
