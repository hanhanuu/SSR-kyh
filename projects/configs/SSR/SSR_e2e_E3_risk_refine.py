_base_ = ['./SSR_e2e.py']

# E3: SSR + Risk Head + Trajectory Refiner
model = dict(
    pts_bbox_head=dict(
        use_risk_head=True,
        risk_head_type='conv',
        risk_out_act='softplus',
        risk_supervision='pseudo',
        risk_loss_weight=1.0,
        loss_risk=dict(type='MSELoss', loss_weight=1.0),
        use_traj_refiner=True,
        refine_steps=3,
        refine_alpha=0.1,
        refiner_patch_size=5,
        refiner_use_bev=True,
        refiner_num_layers=1,
        refiner_num_heads=4,
        refiner_ffn_dim=512,
        traj0_aux_weight=0.2,
        traj_risk_weight=0.0,
    )
)
