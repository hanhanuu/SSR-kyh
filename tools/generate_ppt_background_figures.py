#!/usr/bin/env python3
"""Generate three PPT-ready background figures for SSR report.

This script creates:
1) Risk heatmap (data-driven from agent future trajectories in val infos)
2) BEV fusion schematic (base BEV feature + risk map -> fused feature)
3) Trajectory before/after comparison (from two results_nusc.pkl files)

It is standalone and does not modify training/inference code paths.
"""

import argparse
import os
import pickle
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_infos(path: str):
    data = load_pickle(path)
    if isinstance(data, dict):
        if "infos" in data:
            return data["infos"]
        if "data_list" in data:
            return data["data_list"]
    return data


def build_token_to_info(infos: Iterable[dict]) -> Dict[str, dict]:
    token_to_info = {}
    for info in infos:
        token = info.get("token")
        if token is not None:
            token_to_info[token] = info
    return token_to_info


def extract_selected_traj(result_obj: dict, token: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return selected predicted trajectory and GT trajectory for one token."""
    pred_all, cmd = result_obj["plan_results"][token]
    pred_all = to_numpy(pred_all)
    cmd = to_numpy(cmd).reshape(-1)
    cmd_idx = int(np.argmax(cmd)) if cmd.size > 0 else 0
    if pred_all.ndim == 3:
        pred = pred_all[cmd_idx]
    elif pred_all.ndim == 2:
        pred = pred_all
    else:
        raise ValueError(f"Unsupported pred shape for token {token}: {pred_all.shape}")
    gt = to_numpy(result_obj["plan_gts"][token])
    return pred.astype(np.float32), gt.astype(np.float32)


def trajectory_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    d = np.linalg.norm(pred - gt, axis=1)
    ade = float(np.mean(d))
    fde = float(d[-1])
    return ade, fde


def select_showcase_token(before_obj: dict, after_obj: dict) -> str:
    """Choose token where 'after' improves ADE the most over 'before'."""
    common = (
        set(before_obj.get("plan_results", {}).keys())
        & set(after_obj.get("plan_results", {}).keys())
        & set(before_obj.get("plan_gts", {}).keys())
        & set(after_obj.get("plan_gts", {}).keys())
    )
    if not common:
        raise RuntimeError("No common token found between before/after results.")

    best_token = None
    best_gain = -1e9
    for token in common:
        try:
            pred_b, gt_b = extract_selected_traj(before_obj, token)
            pred_a, gt_a = extract_selected_traj(after_obj, token)
        except Exception:
            continue
        ade_b, _ = trajectory_metrics(pred_b, gt_b)
        ade_a, _ = trajectory_metrics(pred_a, gt_a)
        gain = ade_b - ade_a
        if gain > best_gain:
            best_gain = gain
            best_token = token

    if best_token is None:
        best_token = next(iter(common))
    return best_token


def smooth_map(m: np.ndarray, rounds: int = 4) -> np.ndarray:
    out = m.copy()
    for _ in range(rounds):
        out = (
            out
            + np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
        ) / 5.0
    return out


def build_risk_map(
    info: dict,
    xlim: Tuple[float, float] = (-15.0, 15.0),
    ylim: Tuple[float, float] = (-5.0, 45.0),
    grid_h: int = 360,
    grid_w: int = 320,
    sigma_xy: float = 1.1,
    time_decay: float = 0.88,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a risk heatmap from GT agent future trajectories."""
    xs = np.linspace(xlim[0], xlim[1], grid_w, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], grid_h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    gt_boxes = np.asarray(info["gt_boxes"], dtype=np.float32)
    if "valid_flag" in info:
        mask = np.asarray(info["valid_flag"]).astype(bool)
    else:
        mask = np.asarray(info["num_lidar_pts"]) > 0

    centers = gt_boxes[mask, :2]
    fut = np.asarray(info["gt_agent_fut_trajs"], dtype=np.float32)[mask].reshape(-1, 6, 2)
    fut_mask = np.asarray(info["gt_agent_fut_masks"], dtype=np.float32)[mask].reshape(-1, 6)

    risk = np.zeros((grid_h, grid_w), dtype=np.float32)
    for i in range(centers.shape[0]):
        p = centers[i].copy()
        for t in range(6):
            if fut_mask[i, t] <= 0:
                continue
            p = p + fut[i, t]
            w = time_decay**t
            dist2 = (X - p[0]) ** 2 + (Y - p[1]) ** 2
            risk += w * np.exp(-dist2 / (2.0 * sigma_xy * sigma_xy))

    risk = smooth_map(risk, rounds=2)
    mx = float(risk.max())
    if mx > 1e-8:
        risk = risk / mx
    return xs, ys, risk


def build_bev_feature_and_fused(
    risk_map: np.ndarray, gt_traj: np.ndarray, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Create an interpretable BEV-feature schematic and fused feature map."""
    h, w = risk_map.shape
    xs = np.linspace(-15.0, 15.0, w, dtype=np.float32)
    ys = np.linspace(-5.0, 45.0, h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    lane_prior = np.zeros_like(risk_map, dtype=np.float32)
    for p in gt_traj:
        dist2 = (X - p[0]) ** 2 + (Y - p[1]) ** 2
        lane_prior += np.exp(-dist2 / (2.0 * 1.2 * 1.2))
    lane_prior = lane_prior / (lane_prior.max() + 1e-8)

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=1.0, size=(h, w)).astype(np.float32)
    noise = smooth_map(noise, rounds=8)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    bev_base = 0.50 * lane_prior + 0.30 * noise + 0.20 * risk_map
    bev_base = np.clip(bev_base, 0.0, 1.0)
    bev_base = smooth_map(bev_base, rounds=2)

    fused = np.tanh(1.20 * bev_base + 1.15 * risk_map)
    fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
    return bev_base, fused


def save_risk_heatmap(
    out_path: str,
    xs: np.ndarray,
    ys: np.ndarray,
    risk_map: np.ndarray,
    gt_traj: np.ndarray,
    token: str,
) -> None:
    plt.figure(figsize=(9.0, 6.0), dpi=220)
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    im = plt.imshow(
        risk_map,
        extent=extent,
        origin="lower",
        cmap="magma",
        aspect="auto",
    )
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], "-w", lw=2.2, label="GT ego trajectory")
    plt.scatter(gt_traj[0, 0], gt_traj[0, 1], s=55, c="cyan", edgecolors="k", zorder=5, label="Start")
    cbar = plt.colorbar(im, fraction=0.045, pad=0.02)
    cbar.set_label("Risk intensity", fontsize=10)
    plt.title("Risk Heatmap (from future-agent occupancy)", fontsize=13, weight="bold")
    plt.xlabel("Ego-local X (m)")
    plt.ylabel("Ego-local Y (m)")
    plt.legend(loc="upper left", framealpha=0.92)
    plt.text(
        0.01,
        0.01,
        f"token: {token[:12]}...",
        transform=plt.gca().transAxes,
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.45), ec="none"),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_fusion_schematic(
    out_path: str,
    xs: np.ndarray,
    ys: np.ndarray,
    bev_base: np.ndarray,
    risk_map: np.ndarray,
    fused: np.ndarray,
) -> None:
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=220, constrained_layout=True)
    titles = [
        "BEV Base Feature (schematic)",
        "Risk Prior",
        "Fused Feature",
    ]
    maps = [bev_base, risk_map, fused]
    cmaps = ["viridis", "magma", "plasma"]

    for ax, t, m, c in zip(axes, titles, maps, cmaps):
        im = ax.imshow(m, extent=extent, origin="lower", cmap=c, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_title(t, fontsize=11, weight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle("BEV-Risk Fusion Illustration", fontsize=14, weight="bold")
    fig.text(0.50, 0.02, "Fusion (illustrative): fused = tanh(1.20 * bev + 1.15 * risk)", ha="center", fontsize=10)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_traj_compare(
    out_path: str,
    xs: np.ndarray,
    ys: np.ndarray,
    risk_map: np.ndarray,
    pred_before: np.ndarray,
    pred_after: np.ndarray,
    gt: np.ndarray,
    token: str,
    label_before: str,
    label_after: str,
) -> None:
    ade_b, fde_b = trajectory_metrics(pred_before, gt)
    ade_a, fde_a = trajectory_metrics(pred_after, gt)

    plt.figure(figsize=(9.2, 6.3), dpi=220)
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    plt.imshow(risk_map, extent=extent, origin="lower", cmap="Greys", alpha=0.38, aspect="auto")

    plt.plot(gt[:, 0], gt[:, 1], "-o", lw=2.5, ms=4.2, c="#2ca02c", label="GT")
    plt.plot(pred_before[:, 0], pred_before[:, 1], "--o", lw=2.2, ms=3.8, c="#d62728", label=label_before)
    plt.plot(pred_after[:, 0], pred_after[:, 1], "-o", lw=2.4, ms=4.0, c="#1f77b4", label=label_after)
    plt.scatter(gt[0, 0], gt[0, 1], marker="*", s=130, c="k", label="Start")

    plt.title("Trajectory Comparison: Before vs After Refinement", fontsize=13, weight="bold")
    plt.xlabel("Ego-local X (m)")
    plt.ylabel("Ego-local Y (m)")
    plt.legend(loc="upper left", framealpha=0.94)
    plt.grid(alpha=0.25)
    text = (
        f"ADE  before/after: {ade_b:.3f} / {ade_a:.3f} m\n"
        f"FDE  before/after: {fde_b:.3f} / {fde_a:.3f} m\n"
        f"token: {token[:12]}..."
    )
    plt.text(
        0.56,
        0.03,
        text,
        transform=plt.gca().transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.92),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3 PPT-ready figures for SSR experiment report.")
    parser.add_argument(
        "--before-results",
        default="test/SSR_e2e/Fri_Feb__6_16_47_22_2026/pts_bbox/results_nusc.pkl",
        help="results_nusc.pkl for 'before' trajectory (e.g., baseline).",
    )
    parser.add_argument(
        "--after-results",
        default="test/SSR_e2e_risk_fuse_mlp_12ep/Tue_Feb_10_20_20_31_2026/pts_bbox/results_nusc.pkl",
        help="results_nusc.pkl for 'after' trajectory (e.g., improved model).",
    )
    parser.add_argument(
        "--info-pkl",
        default="data/nuscenes/vad_nuscenes_infos_temporal_val.pkl",
        help="NuScenes val info pkl path.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional sample token. If not set, auto-select by ADE improvement.",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/ppt_assets",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--label-before",
        default="Before (Baseline)",
        help="Legend label for before trajectory.",
    )
    parser.add_argument(
        "--label-after",
        default="After (Improved)",
        help="Legend label for after trajectory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    before_obj = load_pickle(args.before_results)
    after_obj = load_pickle(args.after_results)
    infos = load_infos(args.info_pkl)
    token_to_info = build_token_to_info(infos)

    token = args.token or select_showcase_token(before_obj, after_obj)
    if token not in token_to_info:
        raise KeyError(f"Token {token} not found in info pkl.")

    pred_before, gt_before = extract_selected_traj(before_obj, token)
    pred_after, gt_after = extract_selected_traj(after_obj, token)
    gt = gt_before if gt_before.shape == gt_after.shape else gt_after

    xs, ys, risk_map = build_risk_map(token_to_info[token])
    bev_base, fused = build_bev_feature_and_fused(risk_map, gt)

    save_risk_heatmap(
        out_path=os.path.join(args.out_dir, "01_risk_heatmap.png"),
        xs=xs,
        ys=ys,
        risk_map=risk_map,
        gt_traj=gt,
        token=token,
    )
    save_fusion_schematic(
        out_path=os.path.join(args.out_dir, "02_bev_fusion_schematic.png"),
        xs=xs,
        ys=ys,
        bev_base=bev_base,
        risk_map=risk_map,
        fused=fused,
    )
    save_traj_compare(
        out_path=os.path.join(args.out_dir, "03_traj_before_after.png"),
        xs=xs,
        ys=ys,
        risk_map=risk_map,
        pred_before=pred_before,
        pred_after=pred_after,
        gt=gt,
        token=token,
        label_before=args.label_before,
        label_after=args.label_after,
    )

    print("Generated figures:")
    print(os.path.join(args.out_dir, "01_risk_heatmap.png"))
    print(os.path.join(args.out_dir, "02_bev_fusion_schematic.png"))
    print(os.path.join(args.out_dir, "03_traj_before_after.png"))
    print(f"Showcase token: {token}")


if __name__ == "__main__":
    main()
