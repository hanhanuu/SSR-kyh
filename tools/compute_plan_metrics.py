#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import torch

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.bbox import Box3DMode
from projects.mmdet3d_plugin.SSR.planner.metric_stp3 import PlanningMetric


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute planning metrics from results_nusc.pkl')
    parser.add_argument(
        '--results',
        required=True,
        help='Path to results_nusc.pkl')
    parser.add_argument(
        '--info',
        default='data/nuscenes/vad_nuscenes_infos_temporal_val.pkl',
        help='Path to val infos pkl')
    return parser.parse_args()


def load_infos(path):
    info = pickle.load(open(path, 'rb'))
    if isinstance(info, dict):
        if 'infos' in info:
            return info['infos']
        if 'data_list' in info:  # mmdet3d 1.x export style
            return info['data_list']
    return info


def main():
    args = parse_args()
    res = pickle.load(open(args.results, 'rb'))

    # unwrap list results (common in mmdet test outputs)
    if isinstance(res, list):
        res = {'bbox_results': res}
    # Fast path: averaged metric_results already saved per sample
    if 'bbox_results' in res and isinstance(res['bbox_results'], list) and 'plan_results' not in res:
        metrics = {}
        for item in res['bbox_results']:
            mr = item.get('metric_results', {}) if isinstance(item, dict) else {}
            for k, v in mr.items():
                metrics.setdefault(k, []).append(v)
        if metrics:
            print('Averaged metric_results over', len(res['bbox_results']), 'samples:')
            for k, lst in metrics.items():
                arr = np.array(lst)
                print(f'{k}: {arr.mean():.6f}')
                if k.startswith('plan_obj_col') or k.startswith('plan_obj_box_col'):
                    print(f'  ({k} percent): {arr.mean()*100:.4f}%')
            return

    infos = load_infos(args.info)
    token2info = {i['token']: i for i in infos}

    pm = PlanningMetric()
    metrics = {k: 0.0 for k in [
        'plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s',
        'plan_obj_col_1s', 'plan_obj_col_2s', 'plan_obj_col_3s',
        'plan_obj_box_col_1s', 'plan_obj_box_col_2s', 'plan_obj_box_col_3s',
    ]}

    count = 0
    for token, lst in res['plan_results'].items():
        if token not in token2info:
            continue
        pred, cmd = lst[0], lst[1]
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred)
        if not isinstance(cmd, torch.Tensor):
            cmd = torch.tensor(cmd)
        cmd = cmd.reshape(-1)
        cmd_idx = int(torch.nonzero(cmd)[0, 0].item()) if cmd.sum() > 0 else 0
        if pred.ndim == 3:
            pred = pred[cmd_idx]
        pred = pred.cumsum(dim=0)

        info = token2info[token]
        gt = torch.tensor(info['gt_ego_fut_trajs'], dtype=pred.dtype).cumsum(dim=0)
        if pred.shape[0] < 6 or gt.shape[0] < 6:
            continue

        mask = info.get('valid_flag', info['num_lidar_pts'] > 0)
        gt_boxes = info['gt_boxes'][mask]
        gt_fut_trajs = info['gt_agent_fut_trajs'][mask]
        gt_fut_masks = info['gt_agent_fut_masks'][mask]
        gt_fut_goal = info['gt_agent_fut_goal'][mask]
        gt_lcf_feat = info['gt_agent_lcf_feat'][mask]
        gt_fut_yaw = info['gt_agent_fut_yaw'][mask]
        attr_np = np.concatenate(
            [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None],
             gt_lcf_feat, gt_fut_yaw],
            axis=-1
        ).astype(np.float32)
        attr = torch.tensor(attr_np)[None, ...]

        gt_bboxes = LiDARInstance3DBoxes(
            gt_boxes,
            box_dim=gt_boxes.shape[-1],
            origin=(0.5, 0.5, 0.5)
        ).convert_to(Box3DMode.LIDAR)
        seg_np, ped_np = pm.get_birds_eye_view_label(gt_bboxes, attr)
        occupancy = torch.logical_or(
            torch.from_numpy(seg_np).long(),
            torch.from_numpy(ped_np).long()
        )

        for i in range(3):
            cur = (i + 1) * 2
            l2 = pm.compute_L2(pred[:cur].detach(), gt[:cur])
            obj_coll, obj_box_coll = pm.evaluate_coll(
                pred[None, :cur].detach(),
                gt[None, :cur],
                occupancy[None, :cur]
            )
            metrics[f'plan_L2_{i+1}s'] += l2
            metrics[f'plan_obj_col_{i+1}s'] += obj_coll.mean().item()
            metrics[f'plan_obj_box_col_{i+1}s'] += obj_box_coll.mean().item()

        count += 1

    print('count', count)
    for k in metrics:
        print(k, metrics[k] / max(count, 1))


if __name__ == '__main__':
    main()
