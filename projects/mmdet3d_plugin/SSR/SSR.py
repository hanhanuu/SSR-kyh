"""
 Copyright (c) Zhijia Technology. All rights reserved.
 
 Author: Peidong Li (lipeidong@smartxtruck.com / peidongl@outlook.com)
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import time
import copy
import os

import torch
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmcv.runner import force_fp32, auto_fp16
from scipy.optimize import linear_sum_assignment
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet3d.models.builder import build_loss
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.SSR.planner.metric_stp3 import PlanningMetric
from .tokenlearner import TokenFuser
import torch.nn.functional as F
import torch.nn as nn
# models/model_ssr.py

@DETECTORS.register_module()
class SSR(MVXTwoStageDetector):
    """SSR model.
    """
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 latent_world_model=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 fut_ts=6,
                 fut_mode=6,
                 loss_bev=None,
                 ann_file=None,
                 # World-model BEV loss switch (default keeps original behavior)
                 wm_loss_mode='mse',
                 wm_voxel_size=0.15,
                 wm_fog_alpha_range=(0.0, 0.06),
                 wm_logvar_clamp=(-10.0, 5.0),
                 wm_use_fog_prior=True,
                 wm_latent_noise_weights=(1.0, 0.2, 0.01),
                 infer_safe_reorder=False,
                 infer_safe_reorder_use_box=True,
                 infer_safe_reorder_risk_weight=1.0,
                 infer_safe_reorder_cmd_penalty=0.08,
                 infer_safe_reorder_dev_weight=0.05,
                 infer_safe_reorder_box_kernel_m=(1.8, 4.2),
    ):
        self.ann_file = ann_file
        if self.ann_file:
            if not os.path.exists(self.ann_file):
                raise ValueError(f"Annotation file {self.ann_file} does not exist.")
        super(SSR,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = pts_bbox_head['valid_fut_ts']
        self.bev_h = int(pts_bbox_head.get('bev_h', 0)) if isinstance(pts_bbox_head, dict) else 0
        self.bev_w = int(pts_bbox_head.get('bev_w', 0)) if isinstance(pts_bbox_head, dict) else 0
        self.ann_file = ann_file


        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        self.planning_metric = None
        self.embed_dims = 256
        self.latent_world_model = latent_world_model
        self.tokenfuser = TokenFuser(16, 256)
        self.wm_loss_mode = wm_loss_mode
        self.wm_voxel_size = float(wm_voxel_size)
        self.wm_fog_alpha_range = tuple(float(x) for x in wm_fog_alpha_range)
        self.wm_logvar_clamp = tuple(float(x) for x in wm_logvar_clamp)
        self.wm_use_fog_prior = bool(wm_use_fog_prior)
        self.wm_latent_noise_weights = tuple(float(x) for x in wm_latent_noise_weights)
        if len(self.wm_latent_noise_weights) != 3:
            raise ValueError('wm_latent_noise_weights must contain 3 values: [struct, noise, kl].')
        self.infer_safe_reorder = bool(infer_safe_reorder)
        self.infer_safe_reorder_use_box = bool(infer_safe_reorder_use_box)
        self.infer_safe_reorder_risk_weight = max(float(infer_safe_reorder_risk_weight), 0.0)
        self.infer_safe_reorder_cmd_penalty = max(float(infer_safe_reorder_cmd_penalty), 0.0)
        self.infer_safe_reorder_dev_weight = max(float(infer_safe_reorder_dev_weight), 0.0)
        self.infer_safe_reorder_box_kernel_m = tuple(float(x) for x in infer_safe_reorder_box_kernel_m)
        if len(self.infer_safe_reorder_box_kernel_m) != 2:
            raise ValueError('infer_safe_reorder_box_kernel_m must contain 2 values: (w, l).')

        if self.latent_world_model is not None:
            self.latent_world_model = build_transformer_layer_sequence(self.latent_world_model)
            for p in self.latent_world_model.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
            self.loss_bev = build_loss(loss_bev)

            if self.wm_loss_mode == 'gaussian_nll_fog_prior':
                self.wm_logvar_head = nn.Conv2d(self.embed_dims, 1, kernel_size=1)
                nn.init.constant_(self.wm_logvar_head.bias, 0.0)
            elif self.wm_loss_mode == 'latent_noise_fog':
                self.wm_struct_head = nn.Linear(self.embed_dims, self.embed_dims)
                self.wm_noise_head = nn.Linear(self.embed_dims, self.embed_dims)
                nn.init.xavier_uniform_(self.wm_struct_head.weight)
                nn.init.constant_(self.wm_struct_head.bias, 0.0)
                nn.init.xavier_uniform_(self.wm_noise_head.weight)
                nn.init.constant_(self.wm_noise_head.bias, 0.0)

    @staticmethod
    def _wm_meshgrid(y, x):
        """Compat meshgrid for torch versions without indexing arg."""
        try:
            return torch.meshgrid(y, x, indexing='ij')
        except TypeError:
            return torch.meshgrid(y, x)

    def _wm_fog_prior_logvar(self, feat):
        """Distance-based fog prior on log-variance (higher uncertainty further away)."""
        if not self.wm_use_fog_prior:
            return None
        B, _, H, W = feat.shape
        device = feat.device
        dtype = feat.dtype
        ys, xs = self._wm_meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
        )
        dist = torch.sqrt((xs - (W - 1) / 2) ** 2 + (ys - (H - 1) / 2) ** 2) * self.wm_voxel_size
        dist = dist.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        alpha_min, alpha_max = self.wm_fog_alpha_range
        if alpha_max <= alpha_min:
            alpha = torch.full((B, 1, 1, 1), alpha_min, device=device, dtype=dtype)
        else:
            alpha = torch.rand((B, 1, 1, 1), device=device, dtype=dtype) * (alpha_max - alpha_min) + alpha_min
        return alpha * dist

    def _wm_gaussian_nll(self, mean, logvar, target):
        logvar_min, logvar_max = self.wm_logvar_clamp
        logvar = logvar.clamp(min=logvar_min, max=logvar_max)
        var = torch.exp(logvar)
        loss = (target - mean) ** 2 / (var + 1e-6) + logvar
        return loss.mean()

    def _wm_as_bev_2d(self, feat):
        """Convert BEV feature to [B, C, H, W] for spatial losses."""
        if feat.dim() == 4:
            return feat
        if feat.dim() != 3:
            raise ValueError(f'Unsupported BEV feature dims: {feat.shape}')
        B, HW, C = feat.shape
        H, W = self.bev_h, self.bev_w
        if H <= 0 or W <= 0:
            side = int(HW ** 0.5)
            H = side
            W = HW // max(side, 1)
        if H * W != HW:
            raise ValueError(f'Cannot reshape BEV [B,HW,C]={feat.shape} into [B,C,H,W] with H={H}, W={W}.')
        return feat.permute(0, 2, 1).contiguous().view(B, C, H, W)

    def _wm_compute_noise_mask(self, prev_bev, next_bev):
        """Build fog-aware noisy-region mask on BEV."""
        prev_bev = self._wm_as_bev_2d(prev_bev)
        next_bev = self._wm_as_bev_2d(next_bev)
        device = prev_bev.device
        dtype = prev_bev.dtype
        B, _, H, W = prev_bev.shape
        diff = torch.abs(next_bev - prev_bev).mean(dim=1, keepdim=True)
        diff = diff / (diff.max().detach() + 1e-6)

        alpha_min, alpha_max = self.wm_fog_alpha_range
        if alpha_max <= alpha_min:
            alpha = torch.full((B, 1, 1, 1), alpha_min, device=device, dtype=dtype)
        else:
            alpha = torch.rand((B, 1, 1, 1), device=device, dtype=dtype) * (alpha_max - alpha_min) + alpha_min

        ys, xs = self._wm_meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
        )
        dist = torch.sqrt((xs - (W - 1) / 2) ** 2 + (ys - (H - 1) / 2) ** 2) * self.wm_voxel_size
        dist = dist.unsqueeze(0).unsqueeze(0)
        fog_threshold = torch.exp(-alpha * dist)
        return (diff > fog_threshold).to(dtype=dtype)

    def _wm_fog_weight_map(self, bev_feat):
        """Distance-aware fog weight (larger in far range)."""
        bev_feat = self._wm_as_bev_2d(bev_feat)
        B, _, H, W = bev_feat.shape
        device = bev_feat.device
        dtype = bev_feat.dtype
        y = torch.linspace(-H / 2, H / 2, H, device=device, dtype=dtype)
        x = torch.linspace(-W / 2, W / 2, W, device=device, dtype=dtype)
        yy, xx = self._wm_meshgrid(y, x)
        dist = torch.sqrt(xx ** 2 + yy ** 2).unsqueeze(0).unsqueeze(0)

        alpha_min, alpha_max = self.wm_fog_alpha_range
        if alpha_max <= alpha_min:
            alpha = torch.full((B, 1, 1, 1), alpha_min, device=device, dtype=dtype)
        else:
            alpha = torch.rand((B, 1, 1, 1), device=device, dtype=dtype) * (alpha_max - alpha_min) + alpha_min
        return 1 - torch.exp(-alpha * dist)

    @staticmethod
    def _wm_masked_l1(pred, target, mask, eps=1e-6):
        """L1 loss averaged on masked elements only."""
        if mask is None:
            return F.l1_loss(pred, target)
        mask_expand = mask.expand_as(pred)
        diff = torch.abs(pred - target) * mask_expand
        denom = mask_expand.sum().clamp_min(eps)
        return diff.sum() / denom

    @staticmethod
    def _sample_map_at_traj(traj_xy, risk_map, pc_range):
        """Sample map values at trajectories with bilinear interpolation."""
        if traj_xy.dim() != 3:
            return None
        if risk_map is None:
            return None
        if risk_map.dim() == 2:
            risk_in = risk_map[None, None, ...]
        elif risk_map.dim() == 3:
            risk_in = risk_map[None, ...]
        elif risk_map.dim() == 4:
            risk_in = risk_map
        else:
            return None
        if risk_in.size(1) != 1:
            risk_in = risk_in[:, :1, ...]

        x_min, y_min, _, x_max, y_max, _ = [float(x) for x in pc_range]
        m, t, _ = traj_xy.shape
        risk_rep = risk_in.repeat(m, 1, 1, 1)
        x = (traj_xy[..., 0] - x_min) / max(x_max - x_min, 1e-6)
        y = (traj_xy[..., 1] - y_min) / max(y_max - y_min, 1e-6)
        x = x * 2.0 - 1.0
        y = y * 2.0 - 1.0
        grid = torch.stack([x, y], dim=-1).unsqueeze(2)  # (M, T, 1, 2)
        sampled = F.grid_sample(risk_rep, grid, align_corners=True)  # (M,1,T,1)
        return sampled.squeeze(1).squeeze(-1)  # (M, T)

    def _sample_box_map_at_traj(self, traj_xy, risk_map, pc_range):
        """Approximate box risk by max-pooling map around ego footprint."""
        if risk_map is None:
            return None
        if risk_map.dim() == 2:
            risk_in = risk_map[None, None, ...]
        elif risk_map.dim() == 3:
            risk_in = risk_map[None, ...]
        elif risk_map.dim() == 4:
            risk_in = risk_map
        else:
            return None

        bev_w = max(int(getattr(self.pts_bbox_head, 'bev_w', 0)), 1)
        bev_h = max(int(getattr(self.pts_bbox_head, 'bev_h', 0)), 1)
        real_w = max(float(getattr(self.pts_bbox_head, 'real_w', 1.0)), 1e-6)
        real_h = max(float(getattr(self.pts_bbox_head, 'real_h', 1.0)), 1e-6)
        if bev_w <= 1 or bev_h <= 1:
            return self._sample_map_at_traj(traj_xy, risk_in, pc_range)

        dx = real_w / max(float(bev_w - 1), 1.0)
        dy = real_h / max(float(bev_h - 1), 1.0)
        box_w, box_l = self.infer_safe_reorder_box_kernel_m
        kx = max(int(round(box_w / max(dx, 1e-6))), 1)
        ky = max(int(round(box_l / max(dy, 1e-6))), 1)
        if kx % 2 == 0:
            kx += 1
        if ky % 2 == 0:
            ky += 1
        risk_box = F.max_pool2d(
            risk_in, kernel_size=(ky, kx), stride=1, padding=(ky // 2, kx // 2))
        return self._sample_map_at_traj(traj_xy, risk_box, pc_range)

    def _select_safe_mode_idx(self, ego_fut_preds, ego_fut_cmd, risk_map):
        """Test-time safety reordering over trajectory modes."""
        if ego_fut_cmd is None:
            return 0, 0
        nonzero = torch.nonzero(ego_fut_cmd > 0.5, as_tuple=False)
        if nonzero.numel() > 0:
            cmd_idx = int(nonzero[0, 0].item())
        else:
            cmd_idx = int(torch.argmax(ego_fut_cmd).item())

        if (not self.infer_safe_reorder) or risk_map is None:
            return cmd_idx, cmd_idx
        if ego_fut_preds.dim() != 3 or ego_fut_preds.size(0) <= 1:
            return cmd_idx, cmd_idx

        pc_range = getattr(self.pts_bbox_head, 'pc_range', None)
        if pc_range is None or len(pc_range) < 6:
            return cmd_idx, cmd_idx

        traj_abs = ego_fut_preds.cumsum(dim=-2)
        if self.infer_safe_reorder_use_box:
            risk_vals = self._sample_box_map_at_traj(traj_abs, risk_map, pc_range)
        else:
            risk_vals = self._sample_map_at_traj(traj_abs, risk_map, pc_range)
        if risk_vals is None:
            return cmd_idx, cmd_idx

        risk_mean = risk_vals.mean(dim=1)
        risk_max = risk_vals.max(dim=1)[0]
        risk_score = 0.7 * risk_mean + 0.3 * risk_max

        cmd_penalty = torch.zeros_like(risk_score)
        if cmd_idx < cmd_penalty.numel():
            cmd_penalty[:] = self.infer_safe_reorder_cmd_penalty
            cmd_penalty[cmd_idx] = 0.0

        end_pts = traj_abs[:, -1, :]
        cmd_end = end_pts[cmd_idx]
        end_dev = torch.norm(end_pts - cmd_end[None, :], dim=-1)

        total = (
            self.infer_safe_reorder_risk_weight * risk_score
            + cmd_penalty
            + self.infer_safe_reorder_dev_weight * end_dev
        )
        best_idx = int(torch.argmin(total).item())
        return best_idx, cmd_idx

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,                          
                          img_metas,
                          gt_bboxes_ignore=None,
                          map_gt_bboxes_ignore=None,
                          prev_bev=None,
                          next_bev=None,
                          ego_his_trajs=None,
                          ego_fut_trajs=None,
                          ego_fut_masks=None,
                          ego_fut_cmd=None,
                          ego_lcf_feat=None,
                          gt_attr_labels=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat, cmd=ego_fut_cmd)
        loss_inputs = [
            gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d,
            outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd, gt_attr_labels
        ]
        
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        if self.latent_world_model is not None:
            act_query = outs['act_query']
            # act_pos = outs['act_pos']
            bev_embed = outs['bev_embed']

            pred_latent = self.latent_world_model(
                    query=act_query,
                    key=act_query,
                    value=act_query)
            
            pred_bev = self.tokenfuser(pred_latent.permute(1, 0, 2), bev_embed)
            if self.wm_loss_mode == 'gaussian_nll_fog_prior':
                pred_bev_2d = self._wm_as_bev_2d(pred_bev)
                target_bev_2d = self._wm_as_bev_2d(next_bev.detach())
                logvar = self.wm_logvar_head(pred_bev_2d)
                prior = self._wm_fog_prior_logvar(pred_bev_2d)
                if prior is not None:
                    logvar = logvar + prior
                loss_bev = self._wm_gaussian_nll(pred_bev_2d, logvar, target_bev_2d)
                # keep the same scaling convention as build_loss(loss_bev)
                if hasattr(self.loss_bev, 'loss_weight'):
                    loss_bev = loss_bev * float(getattr(self.loss_bev, 'loss_weight', 1.0))
            elif self.wm_loss_mode == 'latent_noise_fog':
                struct_latent = self.wm_struct_head(pred_latent)
                noise_latent = self.wm_noise_head(pred_latent)
                fog_weight = self._wm_fog_weight_map(bev_embed)
                # Use per-sample fog gate instead of a global scalar to stabilize multi-GPU training.
                fog_gate = fog_weight.mean(dim=(2, 3), keepdim=False).view(1, -1, 1)
                combined_latent = struct_latent + noise_latent * fog_gate

                pred_bev = self.tokenfuser(combined_latent.permute(1, 0, 2), bev_embed)
                pred_bev_2d = self._wm_as_bev_2d(pred_bev)
                target_bev_2d = self._wm_as_bev_2d(next_bev.detach())
                noise_mask = self._wm_compute_noise_mask(bev_embed.detach(), next_bev.detach())
                real_mask = 1.0 - noise_mask

                loss_struct = self._wm_masked_l1(pred_bev_2d, target_bev_2d, real_mask)
                pred_noise = pred_bev_2d * noise_mask
                gt_noise = target_bev_2d * noise_mask
                loss_noise_global = F.l1_loss(
                    pred_noise.mean(dim=(2, 3)),
                    gt_noise.mean(dim=(2, 3)),
                )
                loss_noise_local = self._wm_masked_l1(pred_bev_2d, target_bev_2d, noise_mask)
                loss_noise = 0.5 * loss_noise_global + 0.5 * loss_noise_local
                loss_kl = torch.mean(noise_latent ** 2)
                ws, wn, wk = self.wm_latent_noise_weights
                loss_bev = ws * loss_struct + wn * loss_noise + wk * loss_kl
                if hasattr(self.loss_bev, 'loss_weight'):
                    loss_bev = loss_bev * float(getattr(self.loss_bev, 'loss_weight', 1.0))
            else:
                loss_bev = self.loss_bev(pred_bev, next_bev.detach())
            losses.update(loss_bev=loss_bev)

        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list, prev_cmd):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                cmd = prev_cmd[:, i, ...]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev=prev_bev, only_bev=True, cmd=cmd)
            self.train()
            return prev_bev
    
    def obtain_next_bev(self, img, img_metas):
        """Obtain future BEV features.
        """
        self.eval()
        with torch.no_grad():
            img_feats = self.extract_feat(img=img, img_metas=img_metas)
            next_bev = self.pts_bbox_head(
                    img_feats, img_metas, only_bev=True)
            self.train()
            return next_bev

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        len_queue = img.size(1)
        prev_img = img[:, :-2, ...]
        next_img = img[:, -1, ...]
        img = img[:, -2, ...]
        prev_cmd = ego_fut_cmd[:, :-2, ...]
        next_cmd = ego_fut_cmd[:, -1, ...]
        ego_fut_cmd = ego_fut_cmd[:, -2, ...]

        prev_trajs = ego_fut_trajs[:, :-2, ...]
        next_trajs = ego_fut_trajs[:, -1, ...]
        ego_fut_trajs = ego_fut_trajs[:, -2, ...]

        prev_masks = ego_fut_masks[:, :-2, ...]
        next_masks = ego_fut_masks[:, -1, ...]
        ego_fut_masks = ego_fut_masks[:, -2, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        # next_img_metas = copy.deepcopy(img_metas)
        next_img_metas = [each[len_queue-1] for each in img_metas]

        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas, prev_cmd) if len_queue > 1 else None
        next_bev = self.obtain_next_bev(next_img, next_img_metas)

        img_metas = [each[len_queue-2] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d,
                                            map_gt_bboxes_3d, map_gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, map_gt_bboxes_ignore, prev_bev, next_bev,
                                            ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                            ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                            ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels)
        losses.update(losses_pts)
        return losses

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        map_gt_bboxes_3d,
        map_gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas=img_metas[0],
            img=img[0],
            prev_bev=self.prev_frame_info['prev_bev'],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            map_gt_bboxes_3d=map_gt_bboxes_3d,
            map_gt_labels_3d=map_gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            **kwargs
        )
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_bev'] = new_prev_bev
        self.prev_frame_info['prev_angle'] = tmp_angle

        return bbox_results

    def simple_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        map_gt_bboxes_3d,
        map_gt_labels_3d,
        img=None,
        prev_bev=None,
        points=None,
        fut_valid_flag=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts, metric_dict = self.simple_test_pts(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            map_gt_bboxes_3d,
            map_gt_labels_3d,
            prev_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            start=None,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict

        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        x,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        map_gt_bboxes_3d,
        map_gt_labels_3d,
        prev_bev=None,
        fut_valid_flag=None,
        rescale=False,
        start=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
    ):
        """Test function"""
        mapped_class_names = [
            'car', 'truck', 'construction_vehicle', 'bus',
            'trailer', 'barrier', 'motorcycle', 'bicycle', 
            'pedestrian', 'traffic_cone'
        ]

        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, cmd=ego_fut_cmd,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)

        bbox_results = []
        for i in range(len(outs['ego_fut_preds'])):
            bbox_result=dict()
            pred_i = outs['ego_fut_preds'][i]
            risk_i = outs['risk_map'][i] if ('risk_map' in outs and outs['risk_map'] is not None) else None
            cmd_i = ego_fut_cmd[i, 0, 0] if ego_fut_cmd is not None else None
            select_idx, cmd_idx = self._select_safe_mode_idx(pred_i, cmd_i, risk_i)
            bbox_result['ego_fut_preds'] = pred_i.cpu()
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_result['ego_fut_select_idx'] = int(select_idx)
            bbox_result['ego_fut_cmd_idx'] = int(cmd_idx)
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, 'only support batch_size=1 now'
        # score_threshold = 0.6
        with torch.no_grad():
            gt_bbox = gt_bboxes_3d[0][0]
            gt_map_bbox = map_gt_bboxes_3d[0]
            gt_label = gt_labels_3d[0][0].to('cpu')
            gt_map_label = map_gt_labels_3d[0].to('cpu')
            gt_attr_label = gt_attr_labels[0][0].to('cpu')
            fut_valid_flag = bool(fut_valid_flag[0][0])
      
            metric_dict={}
            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = bbox_result['ego_fut_preds']
            ego_fut_trajs = ego_fut_trajs[0, 0]

            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            if 'ego_fut_select_idx' in bbox_result:
                ego_fut_cmd_idx = int(bbox_result['ego_fut_select_idx'])
            else:
                ego_fut_cmd_idx = int(torch.nonzero(ego_fut_cmd)[0, 0].item())

            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_pred[None],
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                gt_map_boxes = gt_map_bbox,
                gt_map_labels = gt_map_label,
                fut_valid_flag = fut_valid_flag
            )
            metric_dict.update(metric_dict_planner_stp3)

        return outs['bev_embed'], bbox_results, metric_dict

    def map_pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            map_boxes_3d=bboxes.to('cpu'),
            map_scores_3d=scores.cpu(),
            map_labels_3d=labels.cpu(),
            map_pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['map_attrs_3d'] = attrs.cpu()

        return result_dict

    ### same planning metric as stp3
    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        gt_map_boxes,
        gt_map_labels,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
            # 'plan_obj_col_plus_1s':0,
            # 'plan_obj_col_plus_2s':0,
            # 'plan_obj_col_plus_3s':0,
            # 'plan_obj_box_col_plus_1s':0,
            # 'plan_obj_box_col_plus_2s':0,
            # 'plan_obj_box_col_plus_3s':0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian, segmentation_plus = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats, gt_map_boxes, gt_map_labels)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                traj_L2_stp3 = self.planning_metric.compute_L2_stp3(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                # obj_coll_plus, obj_box_coll_plus = self.planning_metric.evaluate_coll(
                #     pred_ego_fut_trajs[:, :cur_time].detach(),
                #     gt_ego_fut_trajs[:, :cur_time],
                #     segmentation_plus)
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i + 1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i + 1)] = obj_box_coll.mean().item()
                # metric_dict['plan_obj_col_plus_{}s'.format(i + 1)] = obj_coll_plus.mean().item()
                # metric_dict['plan_obj_box_col_plus_{}s'.format(i + 1)] = obj_box_coll_plus.mean().item()
                metric_dict['plan_L2_stp3_{}s'.format(i+1)] = traj_L2_stp3
                metric_dict['plan_obj_col_stp3_{}s'.format(i + 1)] = obj_coll[-1].item()
                metric_dict['plan_obj_box_col_stp3_{}s'.format(i + 1)] = obj_box_coll[-1].item()
                # metric_dict['plan_obj_col_stp3_plus_{}s'.format(i + 1)] = obj_coll_plus[-1].item()
                # metric_dict['plan_obj_box_col_stp3_plus_{}s'.format(i + 1)] = obj_box_coll_plus[-1].item()
                # if (i == 0):
                #     metric_dict['plan_1'] = obj_box_coll[0].item()
                #     metric_dict['plan_2'] = obj_box_coll[1].item()
                # if (i == 1):
                #     metric_dict['plan_3'] = obj_box_coll[2].item()
                #     metric_dict['plan_4'] = obj_box_coll[3].item()
                # if (i == 2):
                #     metric_dict['plan_5'] = obj_box_coll[4].item()
                #     metric_dict['plan_6'] = obj_box_coll[5].item()
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_L2_stp3_{}s'.format(i + 1)] = 0.0
            
        return metric_dict

    def set_epoch(self, epoch): 
        self.pts_bbox_head.epoch = epoch
