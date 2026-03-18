_base_ = ['./SSR_e2e.py']

# 12-epoch SSR with ASPP risk head supervised by future occupancy (no BEV fusion) + MLP refiner
total_epochs = 12
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

model = dict(
    pts_bbox_head=dict(
        use_risk_head=True,
        risk_head_type='aspp',
        risk_supervision='future_occ',
        risk_target_dilate=1,
        risk_out_act='sigmoid',
        risk_loss_weight=1.0,
        loss_risk=dict(type='MSELoss', loss_weight=1.0),
        # no fusion with BEV features (ablation Ours-1)
        risk_fusion_mode='none',
        # refiner (MLP)
        use_traj_refiner=True,
        traj_refiner_type='mlp',
        refiner_use_risk=True,
        refiner_use_bev=True,
        refiner_mlp_hidden=512,
        refine_steps=3,
        refine_alpha=0.1,
        traj0_aux_weight=0.3,
        traj_risk_weight=0.0,
    )
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
