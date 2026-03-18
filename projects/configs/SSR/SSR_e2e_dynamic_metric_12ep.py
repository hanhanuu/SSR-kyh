_base_ = ['./SSR_e2e_risk_fuse_mlp_aspp_futureocc_riskw06_12ep.py']

# Loss ablation: directly optimize a dynamic combination of
# L2 proxy + point-collision proxy + box-collision proxy.
model = dict(
    pts_bbox_head=dict(
        use_dynamic_metric_loss=True,
        metric_dyn_priors=(0.34, 0.33, 0.33),
        metric_dyn_gamma=1.8,
        metric_dyn_ema_momentum=0.985,
        metric_dyn_min_weight=0.20,
        metric_dyn_max_weight=0.55,
        metric_dyn_warmup_epochs=2.0,
        metric_dyn_warmup_start=0.20,
        metric_box_kernel_m=(1.8, 4.2),
        metric_col_scale=1.4,
        traj_risk_weight=0.0,
    )
)
