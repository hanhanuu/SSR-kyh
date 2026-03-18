import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskHead(nn.Module):
    """Lightweight conv head to predict BEV risk/cost map."""

    def __init__(self, in_channels=256, hidden_channels=None, out_act='softplus'):
        super().__init__()
        hidden_channels = hidden_channels or max(8, in_channels // 2)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.out_act = out_act

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        if self.out_act == 'softplus':
            x = F.softplus(x)
        elif self.out_act == 'sigmoid':
            x = torch.sigmoid(x)
        return x


class ASPPRiskHead(nn.Module):
    """Stronger multi-scale risk head with dilated conv + SE."""

    def __init__(self, in_channels=256, hidden_channels=None, out_act='softplus'):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        mid = hidden_channels
        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, mid),
            nn.ReLU(inplace=True),
        )
        # multi-dilation branches
        dilations = (1, 3, 5)
        branch_ch = mid // 2
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid, branch_ch, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(16, branch_ch),
                nn.ReLU(inplace=True),
            ) for d in dilations
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * len(dilations), mid, kernel_size=1, bias=False),
            nn.GroupNorm(32, mid),
            nn.ReLU(inplace=True),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, mid // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid // 4, mid, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(mid, 1, kernel_size=3, padding=1)
        self.out_act = out_act

    def forward(self, x):
        x = self.stem(x)
        feats = [b(x) for b in self.branches]
        x = self.fuse(torch.cat(feats, dim=1))
        # squeeze-excitation
        x = x * self.se(x) + x
        x = self.out(x)
        if self.out_act == 'softplus':
            x = F.softplus(x)
        elif self.out_act == 'sigmoid':
            x = torch.sigmoid(x)
        return x


class RiskFusion(nn.Module):
    """Fuse risk map into BEV features.

    Modes:
        - concat: [bev, risk] -> 1x1 conv back to C
        - gated: bev * (1 + sigmoid(conv3x3(risk)))
        - dual: gated first, then concat with risk and 1x1 fuse (stronger)
        - dual_res: dual with learnable residual scale (safe init)
    """

    def __init__(self, channels=256, mode='concat'):
        super().__init__()
        self.mode = mode
        if mode == 'concat':
            self.fuse = nn.Conv2d(channels + 1, channels, kernel_size=1)
        elif mode == 'gated':
            self.fuse = nn.Conv2d(1, channels, kernel_size=3, padding=1)
            self.act = nn.Sigmoid()
        elif mode == 'dual':
            self.gate = nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
            self.fuse = nn.Conv2d(channels + 1, channels, kernel_size=1)
        elif mode == 'dual_res':
            self.gate = nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
            self.fuse = nn.Conv2d(channels + 1, channels, kernel_size=1)
            # Start close to identity to reduce "negative transfer" risk.
            self._gamma = nn.Parameter(torch.tensor(-6.0))
        else:
            raise ValueError(f'Unsupported risk fusion mode: {mode}')

    def forward(self, bev_feat, risk_map):
        if risk_map is None:
            return bev_feat
        if self.mode == 'concat':
            x = torch.cat([bev_feat, risk_map], dim=1)
            return self.fuse(x)
        if self.mode == 'gated':
            gate = self.act(self.fuse(risk_map))
            return bev_feat * (1.0 + gate)
        # dual: gated residual then concat fuse
        gate = self.gate(risk_map)
        bev_gated = bev_feat * (1.0 + gate)
        fused = self.fuse(torch.cat([bev_gated, risk_map], dim=1))
        if self.mode == 'dual_res':
            scale = torch.sigmoid(self._gamma)  # in (0,1), init ~0.0025
            return bev_feat + scale * (fused - bev_feat)
        return fused


class TrajMLPRefiner(nn.Module):
    """Simple per-point MLP refiner using sampled BEV / risk features."""

    def __init__(
        self,
        embed_dims=256,
        hidden_dim=None,
        use_bev=True,
        use_risk=True,
        alpha=0.1,
        max_fut_ts=6,
        pc_range=None,
    ):
        super().__init__()
        self.use_bev = use_bev
        self.use_risk = use_risk
        self.alpha = alpha
        self.max_fut_ts = max_fut_ts
        self.pc_range = pc_range

        hidden_dim = hidden_dim or embed_dims * 2
        # input: [xy(2) + risk(1?) + bev(C?)]
        self.in_dim = 2 + (1 if use_risk else 0) + (embed_dims if use_bev else 0)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.time_emb = nn.Embedding(max_fut_ts, 2)

    def _normalize_xy(self, xy):
        if self.pc_range is None:
            return xy
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        x = (xy[..., 0] - x_min) / max(x_max - x_min, 1e-6)
        y = (xy[..., 1] - y_min) / max(y_max - y_min, 1e-6)
        x = x * 2.0 - 1.0
        y = y * 2.0 - 1.0
        return torch.stack([x, y], dim=-1)

    def _sample_point(self, feat, coords):
        """Sample single point per trajectory step.
        feat: (B, C, H, W), coords: (B, T, 2) normalized [-1,1]
        return: (B, T, C)
        """
        grid = coords.unsqueeze(2)  # (B, T, 1, 2)
        sampled = F.grid_sample(feat, grid, align_corners=True)  # (B, C, T, 1)
        return sampled.squeeze(-1).permute(0, 2, 1)

    def forward(self, traj, risk_map=None, bev_feat=None, refine_steps=1):
        if traj.dim() == 3:
            traj = traj.unsqueeze(1)
        B, M, T, _ = traj.shape
        traj_flat = traj.reshape(B * M, T, 2)

        risk_rep = None
        bev_rep = None
        if self.use_risk:
            if risk_map is None:
                if bev_feat is not None:
                    risk_map = bev_feat.new_zeros(B, 1, bev_feat.shape[2], bev_feat.shape[3])
                else:
                    # no signal available, return original traj
                    return traj
            risk_rep = risk_map.repeat_interleave(M, dim=0)
        if self.use_bev and bev_feat is not None:
            bev_rep = bev_feat.repeat_interleave(M, dim=0)

        coords = self._normalize_xy(traj_flat)
        xy_feat = coords
        time_idx = torch.arange(T, device=traj.device)
        xy_feat = xy_feat + self.time_emb(time_idx)[None, :, :]

        feat_list = [xy_feat]
        if self.use_risk and risk_rep is not None:
            risk_feat = self._sample_point(risk_rep, coords)  # (BM, T, 1)
            feat_list.append(risk_feat)
        if self.use_bev and bev_rep is not None:
            bev_feat_sampled = self._sample_point(bev_rep, coords)  # (BM, T, C)
            feat_list.append(bev_feat_sampled)

        features = torch.cat(feat_list, dim=-1)

        out = traj_flat
        for _ in range(max(refine_steps, 1)):
            delta = self.mlp(features)
            out = out + self.alpha * delta
            # re-normalize coords for next step
            coords = self._normalize_xy(out)
            feat_list = [coords + self.time_emb(time_idx)[None, :, :]]
            if self.use_risk and risk_rep is not None:
                feat_list.append(self._sample_point(risk_rep, coords))
            if self.use_bev and bev_rep is not None:
                feat_list.append(self._sample_point(bev_rep, coords))
            features = torch.cat(feat_list, dim=-1)

        traj_out = out.view(B, M, T, 2)
        return traj_out


def _build_patch_offsets(patch_size, H, W, device, dtype):
    if patch_size <= 1:
        return None
    step_x = 2.0 / max(W - 1, 1)
    step_y = 2.0 / max(H - 1, 1)
    half = patch_size // 2
    offs_x = torch.linspace(-half * step_x, half * step_x, patch_size, device=device, dtype=dtype)
    offs_y = torch.linspace(-half * step_y, half * step_y, patch_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(offs_y, offs_x, indexing='ij')
    return torch.stack([grid_x, grid_y], dim=-1)  # (k, k, 2)


def _sample_patches(feat, coords, patch_size):
    """Sample local patches around coords from feat using grid_sample.

    Args:
        feat: (B, C, H, W)
        coords: (B, T, 2) normalized to [-1, 1]
    Returns:
        patches: (B, T, K, C) where K = patch_size * patch_size
    """
    B, C, H, W = feat.shape
    device, dtype = feat.device, feat.dtype
    if patch_size <= 1:
        grid = coords.unsqueeze(2)  # (B, T, 1, 2)
        sampled = F.grid_sample(feat, grid, align_corners=True)  # (B, C, T, 1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, T, C)
        return sampled.unsqueeze(2)  # (B, T, 1, C)

    offsets = _build_patch_offsets(patch_size, H, W, device, dtype)
    grid = coords[:, :, None, None, :] + offsets[None, None, :, :, :]  # (B, T, k, k, 2)
    grid = grid.view(B * coords.size(1), patch_size, patch_size, 2)
    feat_rep = feat.repeat_interleave(coords.size(1), dim=0)  # (B*T, C, H, W)
    sampled = F.grid_sample(feat_rep, grid, align_corners=True)  # (B*T, C, k, k)
    sampled = sampled.view(B, coords.size(1), C, patch_size, patch_size)
    sampled = sampled.flatten(3).permute(0, 1, 3, 2)  # (B, T, K, C)
    return sampled


class TrajRefiner(nn.Module):
    """Trajectory refinement transformer using local BEV/risk patches."""

    def __init__(
        self,
        embed_dims=256,
        num_layers=1,
        num_heads=4,
        ffn_dim=None,
        patch_size=5,
        use_bev=True,
        alpha=0.1,
        max_fut_ts=6,
        pc_range=None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim or embed_dims * 2
        self.patch_size = patch_size
        self.use_bev = use_bev
        self.alpha = alpha
        self.max_fut_ts = max_fut_ts
        self.pc_range = pc_range

        self.query_xy = nn.Linear(2, embed_dims)
        self.time_emb = nn.Embedding(max_fut_ts, embed_dims)

        kv_in = 1 + (embed_dims if use_bev else 0)
        self.kv_proj = nn.Linear(kv_in, embed_dims)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    dict(
                        attn=nn.MultiheadAttention(embed_dims, num_heads, batch_first=True),
                        norm1=nn.LayerNorm(embed_dims),
                        ffn=nn.Sequential(
                            nn.Linear(embed_dims, self.ffn_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.ffn_dim, embed_dims),
                        ),
                        norm2=nn.LayerNorm(embed_dims),
                    )
                )
            )
        self.delta = nn.Linear(embed_dims, 2)

    def _normalize_xy(self, xy):
        if self.pc_range is None:
            return xy
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        x = (xy[..., 0] - x_min) / max(x_max - x_min, 1e-6)
        y = (xy[..., 1] - y_min) / max(y_max - y_min, 1e-6)
        x = x * 2.0 - 1.0
        y = y * 2.0 - 1.0
        return torch.stack([x, y], dim=-1)

    def _build_kv(self, risk_map, bev_feat, coords):
        risk_patch = _sample_patches(risk_map, coords, self.patch_size)  # (B, T, K, 1)
        if self.use_bev and bev_feat is not None:
            bev_patch = _sample_patches(bev_feat, coords, self.patch_size)  # (B, T, K, C)
            kv = torch.cat([risk_patch, bev_patch], dim=-1)
        else:
            kv = risk_patch
        kv = self.kv_proj(kv)
        B, T, K, C = kv.shape
        return kv.view(B, T * K, C)

    def forward(self, traj, risk_map, bev_feat=None, refine_steps=1):
        """Refine trajectory.

        Args:
            traj: (B, M, T, 2) or (B, T, 2)
            risk_map: (B, 1, H, W)
            bev_feat: (B, C, H, W) or None
        """
        if traj.dim() == 3:
            traj = traj.unsqueeze(1)
        B, M, T, _ = traj.shape
        traj_flat = traj.reshape(B * M, T, 2)
        risk_rep = risk_map.repeat_interleave(M, dim=0)
        bev_rep = bev_feat.repeat_interleave(M, dim=0) if (self.use_bev and bev_feat is not None) else None

        for _ in range(max(refine_steps, 1)):
            coords = self._normalize_xy(traj_flat)
            kv_tokens = self._build_kv(risk_rep, bev_rep, coords)

            q = self.query_xy(coords)
            t_idx = torch.arange(T, device=q.device)
            q = q + self.time_emb(t_idx)[None, :, :]

            x = q
            for layer in self.layers:
                attn_out, _ = layer['attn'](x, kv_tokens, kv_tokens)
                x = layer['norm1'](x + attn_out)
                ffn_out = layer['ffn'](x)
                x = layer['norm2'](x + ffn_out)

            delta = self.delta(x)
            traj_flat = traj_flat + self.alpha * delta

        traj_out = traj_flat.view(B, M, T, 2)
        return traj_out

    def sample_risk(self, traj, risk_map):
        """Sample risk_map at trajectory points (bilinear)."""
        risk_in = risk_map
        if traj.dim() == 4:
            B, M, T, _ = traj.shape
            traj = traj.reshape(B * M, T, 2)
            risk_in = risk_map.repeat_interleave(M, dim=0)
        coords = self._normalize_xy(traj).unsqueeze(2)  # (BM, T, 1, 2)
        risk = F.grid_sample(risk_in, coords, align_corners=True)  # (BM, 1, T, 1)
        risk = risk.squeeze(1).squeeze(-1)  # (BM, T)
        return risk
