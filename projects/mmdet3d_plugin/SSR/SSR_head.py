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
import copy
from math import pi, cos, sin
import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss 
from mmdet.models.dense_heads import DETRHead
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.SSR.utils.map_utils import (
    normalize_2d_pts, normalize_2d_bbox, denormalize_2d_pts, denormalize_2d_bbox
)
from .tokenlearner import *
from mmdet.models.utils import LearnedPositionalEncoding
from .risk_refiner import RiskHead, ASPPRiskHead, RiskFusion, TrajRefiner, TrajMLPRefiner

class MLN(nn.Module):
    ''' 
    from "https://github.com/exiawsh/StreamPETR"
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out

class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.mlp_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.mlp_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.mlp_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.mlp_expand(x_se)
        return x * self.gate(x_se)
    
@HEADS.register_module()
class SSRHead(DETRHead):
    """Head of SSR model.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """
    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 loss_traj=dict(type='L1Loss', loss_weight=0.25),
                 loss_traj_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.8),
                 map_bbox_coder=None,
                 map_num_query=900,
                 map_num_classes=3,
                 map_num_vec=20,
                 map_num_pts_per_vec=2,
                 map_num_pts_per_gt_vec=2,
                 map_query_embed_type='all_pts',
                 map_transform_method='minmax',
                 map_gt_shift_pts_pattern='v0',
                 map_dir_interval=1,
                 map_code_size=None,
                 map_code_weights=None,
                 loss_map_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_map_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_map_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_map_pts=dict(
                    type='ChamferDistance',loss_src_weight=1.0,loss_dst_weight=1.0
                 ),
                 loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 num_scenes=16,
                 latent_decoder=None,
                 way_decoder=None,
                 ego_fut_mode=3,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 ego_lcf_feat_idx=None,
                 valid_fut_ts=6,
                 use_risk_head=False,
                 risk_head_type='conv',
                 risk_out_act='softplus',
                 risk_supervision='pseudo',
                 risk_loss_weight=1.0,
                 loss_risk=dict(type='MSELoss', loss_weight=1.0),
                 risk_target_dilate=0,
                 use_traj_refiner=False,
                 refine_steps=0,
                 refine_alpha=0.1,
                 refiner_patch_size=5,
                 refiner_use_bev=True,
                 refiner_use_risk=True,
                 refiner_num_layers=1,
                 refiner_num_heads=4,
                 refiner_ffn_dim=None,
                 traj_refiner_type='mlp',
                 refiner_mlp_hidden=None,
                 traj0_aux_weight=0.0,
                 traj_risk_weight=0.0,
                 risk_loss_warmup_epochs=0.0,
                 risk_loss_warmup_start=0.0,
                 traj_risk_warmup_epochs=0.0,
                 traj_risk_warmup_start=0.0,
                 risk_fusion_mode='none',
                 use_dynamic_metric_loss=False,
                 metric_dyn_priors=(0.4, 0.3, 0.3),
                 metric_dyn_gamma=1.5,
                 metric_dyn_ema_momentum=0.98,
                 metric_dyn_min_weight=0.15,
                 metric_dyn_max_weight=0.70,
                 metric_dyn_warmup_epochs=2.0,
                 metric_dyn_warmup_start=0.0,
                 metric_box_kernel_m=(1.8, 4.2),
                 metric_col_scale=1.0,
                 use_plan_box_col_loss=False,
                 plan_box_col_loss_weight=0.5,
                 plan_box_col_warmup_epochs=2.0,
                 plan_box_col_warmup_start=0.0,
                 plan_box_col_use_future_occ=True,
                 use_plan_pcgrad=False,
                 pcgrad_eps=1e-8,
                 pcgrad_min_scale=0.25,
                 pcgrad_max_scale=2.0,
                 **kwargs):
        def _strip_keys(node, keys):
            if isinstance(node, dict):
                return {
                    k: _strip_keys(v, keys)
                    for k, v in node.items()
                    if k not in keys
                }
            if isinstance(node, list):
                return [_strip_keys(v, keys) for v in node]
            if isinstance(node, tuple):
                return tuple(_strip_keys(v, keys) for v in node)
            return node

        if 'train_cfg' in kwargs and kwargs['train_cfg'] is not None:
            kwargs['train_cfg'] = _strip_keys(
                kwargs['train_cfg'], {'ann_file', 'map_ann_file'})
        if 'test_cfg' in kwargs and kwargs['test_cfg'] is not None:
            kwargs['test_cfg'] = _strip_keys(
                kwargs['test_cfg'], {'ann_file', 'map_ann_file'})

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode

        self.latent_decoder = latent_decoder
        self.way_decoder = way_decoder

        self.ego_fut_mode = ego_fut_mode
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        self.num_scenes = num_scenes
        self.use_risk_head = use_risk_head
        self.risk_head_type = risk_head_type
        self.risk_out_act = risk_out_act
        self.risk_supervision = risk_supervision
        self.risk_loss_weight = risk_loss_weight
        self.risk_target_dilate = risk_target_dilate
        self.use_traj_refiner = use_traj_refiner
        self.traj_refiner_type = traj_refiner_type
        self.refine_steps = refine_steps
        self.refine_alpha = refine_alpha
        self.refiner_patch_size = refiner_patch_size
        self.refiner_use_bev = refiner_use_bev
        self.refiner_use_risk = refiner_use_risk
        self.refiner_num_layers = refiner_num_layers
        self.refiner_num_heads = refiner_num_heads
        self.refiner_ffn_dim = refiner_ffn_dim
        self.refiner_mlp_hidden = refiner_mlp_hidden
        self.traj0_aux_weight = traj0_aux_weight
        self.traj_risk_weight = traj_risk_weight
        self.risk_loss_warmup_epochs = max(float(risk_loss_warmup_epochs), 0.0)
        self.risk_loss_warmup_start = float(risk_loss_warmup_start)
        self.traj_risk_warmup_epochs = max(float(traj_risk_warmup_epochs), 0.0)
        self.traj_risk_warmup_start = float(traj_risk_warmup_start)
        self.risk_fusion_mode = risk_fusion_mode
        self.use_dynamic_metric_loss = bool(use_dynamic_metric_loss)
        self.metric_dyn_priors = tuple(float(v) for v in metric_dyn_priors)
        self.metric_dyn_gamma = float(metric_dyn_gamma)
        self.metric_dyn_ema_momentum = min(max(float(metric_dyn_ema_momentum), 0.0), 0.9999)
        self.metric_dyn_min_weight = max(float(metric_dyn_min_weight), 0.0)
        self.metric_dyn_max_weight = min(max(float(metric_dyn_max_weight), 0.0), 1.0)
        if self.metric_dyn_max_weight < self.metric_dyn_min_weight:
            self.metric_dyn_max_weight = self.metric_dyn_min_weight
        self.metric_dyn_warmup_epochs = max(float(metric_dyn_warmup_epochs), 0.0)
        self.metric_dyn_warmup_start = float(metric_dyn_warmup_start)
        self.metric_box_kernel_m = tuple(float(v) for v in metric_box_kernel_m)
        self.metric_col_scale = max(float(metric_col_scale), 0.0)
        self.use_plan_box_col_loss = bool(use_plan_box_col_loss)
        self.plan_box_col_loss_weight = max(float(plan_box_col_loss_weight), 0.0)
        self.plan_box_col_warmup_epochs = max(float(plan_box_col_warmup_epochs), 0.0)
        self.plan_box_col_warmup_start = float(plan_box_col_warmup_start)
        self.plan_box_col_use_future_occ = bool(plan_box_col_use_future_occ)
        self.use_plan_pcgrad = bool(use_plan_pcgrad)
        self.pcgrad_eps = max(float(pcgrad_eps), 1e-12)
        self.pcgrad_min_scale = max(float(pcgrad_min_scale), 0.0)
        self.pcgrad_max_scale = max(float(pcgrad_max_scale), self.pcgrad_min_scale)

        if loss_traj_cls['use_sigmoid'] == True:
            self.traj_num_cls = 1
        else:
          self.traj_num_cls = 2

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        if map_code_size is not None:
            self.map_code_size = map_code_size
        else:
            self.map_code_size = 10
        if map_code_weights is not None:
            self.map_code_weights = map_code_weights
        else:
            self.map_code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.map_query_embed_type = map_query_embed_type
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec

        if loss_map_cls['use_sigmoid'] == True:
            self.map_cls_out_channels = map_num_classes
        else:
            self.map_cls_out_channels = map_num_classes + 1

        super(SSRHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.map_code_weights = nn.Parameter(torch.tensor(
            self.map_code_weights, requires_grad=False), requires_grad=False)

        self.loss_plan_reg = build_loss(loss_plan_reg)
        priors = torch.tensor(self.metric_dyn_priors, dtype=torch.float32)
        if priors.numel() != 3:
            raise ValueError('metric_dyn_priors must contain exactly 3 values for L2/ObjCol/BoxCol.')
        priors = priors / priors.sum().clamp(min=1e-6)
        self.register_buffer('metric_dyn_priors_tensor', priors, persistent=False)
        self.register_buffer('metric_proxy_ema', torch.ones(3, dtype=torch.float32), persistent=False)
        self.register_buffer('metric_proxy_ema_ready', torch.zeros(1, dtype=torch.float32), persistent=False)

        self.loss_risk = None
        if self.use_risk_head and loss_risk is not None:
            self.loss_risk = build_loss(loss_risk)

        self.risk_head = None
        if self.use_risk_head:
            if self.risk_head_type == 'conv':
                self.risk_head = RiskHead(
                    in_channels=self.embed_dims,
                    hidden_channels=self.embed_dims // 2,
                    out_act=self.risk_out_act,
                )
            elif self.risk_head_type == 'aspp':
                self.risk_head = ASPPRiskHead(
                    in_channels=self.embed_dims,
                    hidden_channels=self.embed_dims,
                    out_act=self.risk_out_act,
                )
            else:
                raise ValueError(f'Unsupported risk_head_type: {self.risk_head_type}')
        self.risk_fusion = None
        if self.use_risk_head and self.risk_fusion_mode != 'none':
            self.risk_fusion = RiskFusion(
                channels=self.embed_dims, mode=self.risk_fusion_mode)

        self.traj_refiner = None
        if self.use_traj_refiner:
            if self.traj_refiner_type == 'attn':
                self.traj_refiner = TrajRefiner(
                    embed_dims=self.embed_dims,
                    num_layers=self.refiner_num_layers,
                    num_heads=self.refiner_num_heads,
                    ffn_dim=self.refiner_ffn_dim or self.embed_dims * 2,
                    patch_size=self.refiner_patch_size,
                    use_bev=self.refiner_use_bev,
                    alpha=self.refine_alpha,
                    max_fut_ts=self.fut_ts,
                    pc_range=self.pc_range,
                )
            elif self.traj_refiner_type == 'mlp':
                self.traj_refiner = TrajMLPRefiner(
                    embed_dims=self.embed_dims,
                    hidden_dim=self.refiner_mlp_hidden,
                    use_bev=self.refiner_use_bev,
                    use_risk=self.refiner_use_risk,
                    alpha=self.refine_alpha,
                    max_fut_ts=self.fut_ts,
                    pc_range=self.pc_range,
                )
            else:
                raise ValueError(f'Unsupported traj_refiner_type: {self.traj_refiner_type}')

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        traj_branch = []
        for _ in range(self.num_reg_fcs):
            traj_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_branch.append(nn.ReLU())
        traj_branch.append(Linear(self.embed_dims*2, self.fut_ts*2))
        traj_branch = nn.Sequential(*traj_branch)

        traj_cls_branch = []
        for _ in range(self.num_reg_fcs):
            traj_cls_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims*2))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(Linear(self.embed_dims*2, self.traj_num_cls))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        map_cls_branch = []
        for _ in range(self.num_reg_fcs):
            map_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_cls_branch.append(nn.LayerNorm(self.embed_dims))
            map_cls_branch.append(nn.ReLU(inplace=True))
        map_cls_branch.append(Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch = nn.Sequential(*map_cls_branch)

        map_reg_branch = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.ReLU())
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)


        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_decoder_layers = 1
        num_map_decoder_layers = 1
        if self.transformer.decoder is not None:
            num_decoder_layers = self.transformer.decoder.num_layers
        if self.transformer.map_decoder is not None:
            num_map_decoder_layers = self.transformer.map_decoder.num_layers
        num_motion_decoder_layers = 1
        num_pred = (num_decoder_layers + 1) if \
            self.as_two_stage else num_decoder_layers
        motion_num_pred = (num_motion_decoder_layers + 1) if \
            self.as_two_stage else num_motion_decoder_layers
        map_num_pred = (num_map_decoder_layers + 1) if \
            self.as_two_stage else num_map_decoder_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(cls_branch, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)
            self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
            self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.traj_branches = nn.ModuleList(
                [traj_branch for _ in range(motion_num_pred)])
            self.traj_cls_branches = nn.ModuleList(
                [traj_cls_branch for _ in range(motion_num_pred)])
            self.map_cls_branches = nn.ModuleList(
                [map_cls_branch for _ in range(map_num_pred)])
            self.map_reg_branches = nn.ModuleList(
                [map_reg_branch for _ in range(map_num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
            if self.map_query_embed_type == 'all_pts':
                self.map_query_embedding = nn.Embedding(self.map_num_query,
                                                    self.embed_dims * 2)
            elif self.map_query_embed_type == 'instance_pts':
                self.map_query_embedding = None
                self.map_instance_embedding = nn.Embedding(self.map_num_vec, self.embed_dims * 2)
                self.map_pts_embedding = nn.Embedding(self.map_num_pts_per_vec, self.embed_dims * 2)
        
        self.ego_query = nn.Embedding(1, self.embed_dims)	

        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims + len(self.ego_lcf_feat_idx) \
            if self.ego_lcf_feat_idx is not None else self.embed_dims
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)
        self.navi_embedding = nn.Embedding(3, self.embed_dims)
        self.navi_se = SELayer(self.embed_dims)

        self.way_point = nn.Embedding(self.ego_fut_mode*self.fut_ts, self.embed_dims * 2)
        self.tokenlearner = TokenLearnerV11(self.num_scenes, self.embed_dims * 2)

        self.latent_decoder = build_transformer_layer_sequence(self.latent_decoder)
        self.way_decoder = build_transformer_layer_sequence(self.way_decoder)

        self.action_mln = MLN(self.fut_ts*2)
        self.pos_mln = MLN(self.fut_ts*2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.latent_decoder is not None:
            for p in self.latent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p) 
        if self.way_decoder is not None:
            for p in self.way_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                cmd=None
            ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype


        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        bev_embed = self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        if only_bev:
            return bev_embed

        pos_embd = bev_pos.flatten(2).permute(0, 2, 1)
        cmd = cmd[0, 0, 0]
        cmd_idx = torch.nonzero(cmd)[0, 0]

        navi_embed = self.navi_embedding.weight[cmd_idx][None, None]
        bev_navi_embed = self.navi_se(bev_embed, navi_embed)

        bev_query = torch.cat((bev_navi_embed, pos_embd), -1)

        learned_latent_query, selected = self.tokenlearner(bev_query)

        learned_latent_query=learned_latent_query.permute(1, 0, 2)
        latent_query, latent_pos = torch.split(
            learned_latent_query, self.embed_dims, dim=2)

        latent_query = self.latent_decoder(
                query=latent_query,
                key=latent_query,
                value=latent_query,
                query_pos=latent_pos,
                key_pos=latent_pos)

        way_point = self.way_point.weight.to(dtype)
        wp_pos, way_point = torch.split(
            way_point, self.embed_dims, dim=1)

        wp_pos = wp_pos.unsqueeze(0).expand(bs, -1, -1)
        way_point = way_point.unsqueeze(0).expand(bs, -1, -1)
        wp_pos = wp_pos.permute(1, 0, 2)
        way_point = way_point.permute(1, 0, 2)

        way_point = self.way_decoder(
                query=way_point,
                key=latent_query,
                value=latent_query,
                query_pos=wp_pos,
                key_pos=latent_pos)

        outputs_ego_trajs = self.ego_fut_decoder(way_point)
        outputs_ego_trajs = outputs_ego_trajs.permute(1, 0, 2). view(bs, 
                                                      self.ego_fut_mode, self.fut_ts, 2)
        outputs_ego_trajs_fut=outputs_ego_trajs[:,cmd_idx,...]
        wp_vector = outputs_ego_trajs_fut.reshape(-1)

        wp_vector = wp_vector.unsqueeze(0).unsqueeze(0)

        act_query = self.action_mln(latent_query, wp_vector)
        # act_pos = self.pos_mln(latent_pos[:self.num_scenes, ...], wp_vector)

        risk_map = None
        bev_feat = bev_embed.permute(0, 2, 1).reshape(
            bs, self.embed_dims, self.bev_h, self.bev_w)
        if self.use_risk_head:
            risk_map = self.risk_head(bev_feat)
        bev_feat_for_refine = bev_feat
        if self.risk_fusion is not None and risk_map is not None:
            bev_feat_for_refine = self.risk_fusion(bev_feat, risk_map)

        ego_fut_preds = outputs_ego_trajs
        if self.use_traj_refiner and self.traj_refiner is not None and self.refine_steps > 0:
            if self.traj_refiner_type == 'attn':
                if risk_map is not None:
                    ego_fut_preds = self.traj_refiner(
                        outputs_ego_trajs,
                        risk_map,
                        bev_feat=bev_feat_for_refine,
                        refine_steps=self.refine_steps,
                    )
            else:  # mlp refiner supports None risk_map / bev_feat depending on flags
                ego_fut_preds = self.traj_refiner(
                    outputs_ego_trajs,
                    risk_map if self.refiner_use_risk else None,
                    bev_feat=bev_feat_for_refine if self.refiner_use_bev else None,
                    refine_steps=self.refine_steps,
                )

        outs = {
            'bev_embed': bev_embed,
            'scene_query': latent_query,
            'act_query': act_query,
            # 'act_pos': act_pos,
            'ego_fut_preds': ego_fut_preds,
        }
        if self.use_risk_head:
            outs['risk_map'] = risk_map
        if self.use_traj_refiner:
            outs['ego_fut_preds0'] = outputs_ego_trajs

        return outs

    def _build_risk_target(self, gt_bboxes_list, device, dtype):
        """Build a pseudo risk target from GT 3D boxes (BEV AABB)."""
        B = len(gt_bboxes_list)
        target = torch.zeros(
            (B, 1, self.bev_h, self.bev_w), device=device, dtype=dtype)
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        for b, boxes in enumerate(gt_bboxes_list):
            if boxes is None or len(boxes) == 0:
                continue
            if hasattr(boxes, 'tensor'):
                t = boxes.tensor
            else:
                t = boxes
            if t.numel() == 0:
                continue
            x = t[:, 0]
            y = t[:, 1]
            w = t[:, 3]
            l = t[:, 4]
            x1 = (x - w / 2).clamp(min=x_min, max=x_max)
            x2 = (x + w / 2).clamp(min=x_min, max=x_max)
            y1 = (y - l / 2).clamp(min=y_min, max=y_max)
            y2 = (y + l / 2).clamp(min=y_min, max=y_max)

            ix1 = ((x1 - x_min) / max(self.real_w, 1e-6) * (self.bev_w - 1)).floor().long()
            ix2 = ((x2 - x_min) / max(self.real_w, 1e-6) * (self.bev_w - 1)).ceil().long()
            iy1 = ((y1 - y_min) / max(self.real_h, 1e-6) * (self.bev_h - 1)).floor().long()
            iy2 = ((y2 - y_min) / max(self.real_h, 1e-6) * (self.bev_h - 1)).ceil().long()

            ix1 = ix1.clamp(0, self.bev_w - 1)
            ix2 = ix2.clamp(0, self.bev_w - 1)
            iy1 = iy1.clamp(0, self.bev_h - 1)
            iy2 = iy2.clamp(0, self.bev_h - 1)

            for i in range(ix1.size(0)):
                target[b, 0, iy1[i]:iy2[i] + 1, ix1[i]:ix2[i] + 1] = 1.0
        return target

    def _build_risk_target_future_occ(self, gt_bboxes_list, gt_attr_labels, device, dtype):
        """Build a risk target from GT future occupancy (AABB, aggregated over time).

        This aligns better with STP3-style planning collision metrics which use
        future agent boxes (from gt_attr_labels) rather than only the current box.
        """
        B = len(gt_bboxes_list)
        target = torch.zeros(
            (B, 1, self.bev_h, self.bev_w), device=device, dtype=dtype)

        if gt_attr_labels is None:
            return target

        # Make gt_attr_labels indexable per batch.
        if isinstance(gt_attr_labels, (list, tuple)):
            attr_list = list(gt_attr_labels)
        elif torch.is_tensor(gt_attr_labels) and gt_attr_labels.dim() >= 2:
            # (B, N, C) -> list of (N, C)
            attr_list = [gt_attr_labels[i] for i in range(min(B, gt_attr_labels.size(0)))]
        else:
            attr_list = [None] * B

        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        T = int(min(self.valid_fut_ts, self.fut_ts, 6))
        # NuScenes VAD attr layout uses: fut_traj(6*2) + fut_mask(6) + goal(1) + lcf_feat(9) + fut_yaw(6)
        type_idx = 6 * 3 + 9  # 27 when T=6
        veh_human_set = set([2, 3, 4, 5, 6, 7, 8] + list(range(14, 24)))

        for b, boxes in enumerate(gt_bboxes_list):
            if boxes is None or len(boxes) == 0:
                continue
            if hasattr(boxes, 'tensor'):
                box_t = boxes.tensor
            else:
                box_t = boxes
            if box_t.numel() == 0:
                continue
            if torch.is_tensor(box_t):
                box_t = box_t.detach().to(device)

            attr = attr_list[b] if b < len(attr_list) else None
            if attr is None:
                continue
            if isinstance(attr, (list, tuple)) and len(attr) == 1:
                attr = attr[0]
            if torch.is_tensor(attr) and attr.dim() == 3 and attr.size(0) == 1:
                attr = attr.squeeze(0)
            if not torch.is_tensor(attr) or attr.numel() == 0:
                continue
            # gt_attr_labels is scattered to GPU while gt_bboxes_3d stays on CPU (cpu_only=True).
            # Move boxes to GPU so risk supervision stays on GPU (avoid CPU-heavy work).
            attr = attr.detach().to(device)

            n = min(box_t.size(0), attr.size(0))
            box_t = box_t[:n]
            attr = attr[:n]

            # Extract future deltas and masks (always 6 steps in data; we clamp by T).
            fut_traj = attr[:, :6 * 2].reshape(n, 6, 2)[:, :T]
            fut_mask = attr[:, 6 * 2:6 * 3].reshape(n, 6)[:, :T]
            # Optional type filtering; if nothing passes, fall back to include all.
            included = 0
            if attr.size(1) > type_idx:
                types = attr[:, type_idx].long()
                type_ok = torch.tensor(
                    [int(t.item()) in veh_human_set for t in types],
                    device=attr.device,
                    dtype=torch.bool,
                )
            else:
                type_ok = torch.ones((n,), device=attr.device, dtype=torch.bool)

            pos0 = box_t[:, 0:2]
            fut_pos = fut_traj.cumsum(dim=1) + pos0[:, None, :]  # (n, T, 2)

            # Rasterize AABB boxes on our BEV grid.
            for i in range(n):
                if not bool(type_ok[i].item()):
                    continue
                w = box_t[i, 3]
                l = box_t[i, 4]
                for t in range(T):
                    if float(fut_mask[i, t].item()) < 0.5:
                        continue
                    included += 1
                    x = fut_pos[i, t, 0]
                    y = fut_pos[i, t, 1]
                    x1 = (x - w / 2).clamp(min=x_min, max=x_max)
                    x2 = (x + w / 2).clamp(min=x_min, max=x_max)
                    y1 = (y - l / 2).clamp(min=y_min, max=y_max)
                    y2 = (y + l / 2).clamp(min=y_min, max=y_max)

                    ix1 = ((x1 - x_min) / max(self.real_w, 1e-6) * (self.bev_w - 1)).floor().long()
                    ix2 = ((x2 - x_min) / max(self.real_w, 1e-6) * (self.bev_w - 1)).ceil().long()
                    iy1 = ((y1 - y_min) / max(self.real_h, 1e-6) * (self.bev_h - 1)).floor().long()
                    iy2 = ((y2 - y_min) / max(self.real_h, 1e-6) * (self.bev_h - 1)).ceil().long()

                    ix1 = ix1.clamp(0, self.bev_w - 1)
                    ix2 = ix2.clamp(0, self.bev_w - 1)
                    iy1 = iy1.clamp(0, self.bev_h - 1)
                    iy2 = iy2.clamp(0, self.bev_h - 1)

                    target[b, 0, iy1:iy2 + 1, ix1:ix2 + 1] = 1.0

            # If type filtering excluded everything, try again without filtering.
            if included == 0 and attr.size(1) > type_idx:
                for i in range(n):
                    w = box_t[i, 3]
                    l = box_t[i, 4]
                    for t in range(T):
                        if float(fut_mask[i, t].item()) < 0.5:
                            continue
                        x = fut_pos[i, t, 0]
                        y = fut_pos[i, t, 1]
                        x1 = (x - w / 2).clamp(min=x_min, max=x_max)
                        x2 = (x + w / 2).clamp(min=x_min, max=x_max)
                        y1 = (y - l / 2).clamp(min=y_min, max=y_max)
                        y2 = (y + l / 2).clamp(min=y_min, max=y_max)
                        ix1 = ((x1 - x_min) / max(self.real_w, 1e-6) * (self.bev_w - 1)).floor().long()
                        ix2 = ((x2 - x_min) / max(self.real_w, 1e-6) * (self.bev_w - 1)).ceil().long()
                        iy1 = ((y1 - y_min) / max(self.real_h, 1e-6) * (self.bev_h - 1)).floor().long()
                        iy2 = ((y2 - y_min) / max(self.real_h, 1e-6) * (self.bev_h - 1)).ceil().long()
                        ix1 = ix1.clamp(0, self.bev_w - 1)
                        ix2 = ix2.clamp(0, self.bev_w - 1)
                        iy1 = iy1.clamp(0, self.bev_h - 1)
                        iy2 = iy2.clamp(0, self.bev_h - 1)
                        target[b, 0, iy1:iy2 + 1, ix1:ix2 + 1] = 1.0

        if self.risk_target_dilate and int(self.risk_target_dilate) > 0:
            k = int(self.risk_target_dilate) * 2 + 1
            target = F.max_pool2d(target, kernel_size=k, stride=1, padding=k // 2)
        return target

    def _sample_risk_at_traj(self, traj, risk_map):
        """Sample risk map at trajectory points (bilinear)."""
        risk_in = risk_map
        if traj.dim() == 4:
            B, M, T, _ = traj.shape
            traj = traj.view(B * M, T, 2)
            risk_in = risk_map.repeat_interleave(M, dim=0)
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        x = (traj[..., 0] - x_min) / max(x_max - x_min, 1e-6)
        y = (traj[..., 1] - y_min) / max(y_max - y_min, 1e-6)
        x = x * 2.0 - 1.0
        y = y * 2.0 - 1.0
        grid = torch.stack([x, y], dim=-1).unsqueeze(2)  # (BM, T, 1, 2)
        risk = F.grid_sample(risk_in, grid, align_corners=True)
        return risk.squeeze(1).squeeze(-1)

    def _sample_risk_box_at_traj(self, traj, risk_map):
        """Approximate box-collision risk by sampling max-pooled footprint risk."""
        if self.bev_w <= 1 or self.bev_h <= 1:
            return self._sample_risk_at_traj(traj, risk_map)
        dx = self.real_w / max(float(self.bev_w - 1), 1.0)
        dy = self.real_h / max(float(self.bev_h - 1), 1.0)
        box_w, box_l = self.metric_box_kernel_m
        kx = max(int(round(box_w / max(dx, 1e-6))), 1)
        ky = max(int(round(box_l / max(dy, 1e-6))), 1)
        if kx % 2 == 0:
            kx += 1
        if ky % 2 == 0:
            ky += 1
        risk_box = F.max_pool2d(risk_map, kernel_size=(ky, kx), stride=1, padding=(ky // 2, kx // 2))
        return self._sample_risk_at_traj(traj, risk_box)

    def _compute_dynamic_metric_weights(self, proxy_l2, proxy_obj_col, proxy_box_col):
        """Compute dynamic weights for L2 / point-collision / box-collision proxies."""
        eps = 1e-6
        cur = torch.stack([proxy_l2, proxy_obj_col, proxy_box_col]).detach().float().clamp(min=eps)
        if float(self.metric_proxy_ema_ready.item()) < 0.5:
            self.metric_proxy_ema.copy_(cur)
            self.metric_proxy_ema_ready.fill_(1.0)
        else:
            m = self.metric_dyn_ema_momentum
            self.metric_proxy_ema.mul_(m).add_(cur * (1.0 - m))

        ema = self.metric_proxy_ema.detach().clamp(min=eps)
        ratios = (cur / ema).clamp(min=0.25, max=4.0)
        priors = self.metric_dyn_priors_tensor.to(device=cur.device, dtype=cur.dtype)

        raw = priors * torch.pow(ratios, self.metric_dyn_gamma)
        weights = raw / raw.sum().clamp(min=eps)

        weights = weights.clamp(min=self.metric_dyn_min_weight, max=self.metric_dyn_max_weight)
        weights = weights / weights.sum().clamp(min=eps)

        warm = float(self._get_epoch_warmup_scale(self.metric_dyn_warmup_epochs, self.metric_dyn_warmup_start))
        warm = max(0.0, min(1.0, warm))
        weights = (1.0 - warm) * priors + warm * weights
        weights = weights / weights.sum().clamp(min=eps)
        return weights.to(dtype=proxy_l2.dtype)

    def _get_epoch_warmup_scale(self, warmup_epochs, start):
        """Linear warmup factor based on runner epoch."""
        if warmup_epochs <= 0:
            return 1.0
        cur_epoch = max(float(getattr(self, 'epoch', 0)), 0.0)
        progress = min(cur_epoch / float(warmup_epochs), 1.0)
        start = min(max(float(start), 0.0), 1.0)
        return start + (1.0 - start) * progress

    def _compute_plan_pcgrad_scales(self, loss_main, loss_aux, traj_tensor):
        """PCGrad-style conflict handling for planning L2 vs box-collision losses."""
        one = loss_main.detach().new_tensor(1.0)
        zero = loss_main.detach().new_tensor(0.0)
        if (not self.use_plan_pcgrad) or traj_tensor is None or (not traj_tensor.requires_grad):
            return one, one, zero

        g_main = torch.autograd.grad(
            loss_main, traj_tensor, retain_graph=True, allow_unused=True)[0]
        g_aux = torch.autograd.grad(
            loss_aux, traj_tensor, retain_graph=True, allow_unused=True)[0]
        if g_main is None or g_aux is None:
            return one, one, zero

        g_main = g_main.reshape(-1)
        g_aux = g_aux.reshape(-1)
        dot = (g_main * g_aux).sum()
        norm_main = (g_main * g_main).sum().clamp(min=self.pcgrad_eps)
        norm_aux = (g_aux * g_aux).sum().clamp(min=self.pcgrad_eps)
        cos = (dot / (torch.sqrt(norm_main * norm_aux).clamp(min=self.pcgrad_eps))).detach()

        if float(dot.detach().item()) >= 0.0:
            return one, one, cos

        main_scale = (1.0 - dot / norm_main).detach().clamp(
            min=self.pcgrad_min_scale, max=self.pcgrad_max_scale)
        aux_scale = (1.0 - dot / norm_aux).detach().clamp(
            min=self.pcgrad_min_scale, max=self.pcgrad_max_scale)
        return main_scale, aux_scale, cos

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        ego_fut_preds = preds_dicts['ego_fut_preds']
        ego_fut_preds0 = preds_dicts.get('ego_fut_preds0', None)
        risk_map = preds_dicts.get('risk_map', None)
        risk_warmup_scale = self._get_epoch_warmup_scale(
            self.risk_loss_warmup_epochs, self.risk_loss_warmup_start)
        traj_risk_warmup_scale = self._get_epoch_warmup_scale(
            self.traj_risk_warmup_epochs, self.traj_risk_warmup_start)

        loss_dict = dict()

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )
        mode_time_weight = ego_fut_cmd[..., None] * ego_fut_masks[:, None, :]
        if ego_fut_preds.dim() == 4:
            B, M, T, _ = ego_fut_preds.shape
            mode_time_weight_flat = mode_time_weight.reshape(B * M, T)
        else:
            mode_time_weight_flat = ego_fut_masks

        if self.use_plan_box_col_loss and not self.use_dynamic_metric_loss:
            if self.plan_box_col_use_future_occ:
                box_col_target = self._build_risk_target_future_occ(
                    gt_bboxes_list,
                    gt_attr_labels,
                    device=ego_fut_preds.device,
                    dtype=ego_fut_preds.dtype,
                )
            else:
                box_col_target = self._build_risk_target(
                    gt_bboxes_list,
                    device=ego_fut_preds.device,
                    dtype=ego_fut_preds.dtype,
                )
            proxy_box_col = self._sample_risk_box_at_traj(ego_fut_preds, box_col_target)
            norm = mode_time_weight_flat.sum().clamp(min=1.0)
            loss_plan_box_col = (proxy_box_col * mode_time_weight_flat).sum() / norm
            box_col_warmup = self._get_epoch_warmup_scale(
                self.plan_box_col_warmup_epochs, self.plan_box_col_warmup_start)
            loss_plan_box_col = (
                loss_plan_box_col * self.plan_box_col_loss_weight * box_col_warmup)
            if self.use_plan_pcgrad:
                main_scale, box_scale, pcgrad_cos = self._compute_plan_pcgrad_scales(
                    loss_plan_l1, loss_plan_box_col, ego_fut_preds)
                loss_plan_l1 = loss_plan_l1 * main_scale
                loss_plan_box_col = loss_plan_box_col * box_scale
                loss_dict['stat_pcgrad_cos'] = pcgrad_cos
                loss_dict['stat_pcgrad_l2_scale'] = main_scale
                loss_dict['stat_pcgrad_box_scale'] = box_scale
            loss_dict['loss_plan_col_box'] = loss_plan_box_col

        dynamic_metric_applied = False
        if self.use_dynamic_metric_loss and risk_map is not None:
            risk_point = self._sample_risk_at_traj(ego_fut_preds, risk_map)
            risk_box = self._sample_risk_box_at_traj(ego_fut_preds, risk_map)
            norm = mode_time_weight_flat.sum().clamp(min=1.0)
            proxy_obj_col = (risk_point * mode_time_weight_flat).sum() / norm
            proxy_box_col = (risk_box * mode_time_weight_flat).sum() / norm

            dyn_w = self._compute_dynamic_metric_weights(
                loss_plan_l1, proxy_obj_col, proxy_box_col)
            loss_dict['loss_plan_reg'] = loss_plan_l1 * dyn_w[0]
            loss_dict['loss_plan_col_obj'] = proxy_obj_col * dyn_w[1] * self.metric_col_scale
            loss_dict['loss_plan_col_box'] = proxy_box_col * dyn_w[2] * self.metric_col_scale
            loss_dict['stat_dyn_w_l2'] = dyn_w[0].detach()
            loss_dict['stat_dyn_w_obj'] = dyn_w[1].detach()
            loss_dict['stat_dyn_w_box'] = dyn_w[2].detach()
            loss_dict['stat_proxy_obj_col'] = proxy_obj_col.detach()
            loss_dict['stat_proxy_box_col'] = proxy_box_col.detach()
            dynamic_metric_applied = True

        if not dynamic_metric_applied:
            loss_dict['loss_plan_reg'] = loss_plan_l1

        if self.traj0_aux_weight > 0.0 and ego_fut_preds0 is not None:
            loss_plan_aux = self.loss_plan_reg(
                ego_fut_preds0,
                ego_fut_gt,
                loss_plan_l1_weight
            )
            loss_dict['loss_plan_reg_aux'] = loss_plan_aux * self.traj0_aux_weight

        if self.use_risk_head and risk_map is not None and self.loss_risk is not None:
            if self.risk_supervision == 'pseudo':
                risk_target = self._build_risk_target(
                    gt_bboxes_list, device=risk_map.device, dtype=risk_map.dtype)
            elif self.risk_supervision == 'future_occ':
                risk_target = self._build_risk_target_future_occ(
                    gt_bboxes_list, gt_attr_labels, device=risk_map.device, dtype=risk_map.dtype)
            else:
                raise ValueError(f'Unsupported risk_supervision: {self.risk_supervision}')
            loss_risk = self.loss_risk(risk_map, risk_target)
            loss_dict['loss_risk'] = loss_risk * self.risk_loss_weight * risk_warmup_scale

        if self.traj_risk_weight > 0.0 and risk_map is not None:
            risk_vals = self._sample_risk_at_traj(ego_fut_preds, risk_map)
            loss_traj_risk = risk_vals.mean()
            loss_dict['loss_traj_risk'] = (
                loss_traj_risk * self.traj_risk_weight * traj_risk_warmup_scale)

        return loss_dict
