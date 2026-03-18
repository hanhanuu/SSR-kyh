_base_ = ['./SSR_e2e.py']

# 12-epoch SSR with ASPP risk head supervised by future occupancy + dual fusion + MLP refiner
total_epochs = 12
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

model = dict(
    pts_bbox_head=dict(
        use_risk_head=True,
        risk_head_type='aspp',
        # Use future occupancy (from gt_attr_labels) instead of only current GT boxes.
        risk_supervision='future_occ',
        # Optional dilation (in BEV cells) to soften class-imbalance / add safety margin.
        risk_target_dilate=1,
        risk_out_act='sigmoid',
        risk_loss_weight=1.0,
        loss_risk=dict(type='MSELoss', loss_weight=1.0),
        risk_fusion_mode='dual_res',
        # refiner (MLP)
        use_traj_refiner=True,
        traj_refiner_type='mlp',
        refiner_use_risk=True,
        refiner_use_bev=True,
        refiner_mlp_hidden=512,
        refine_steps=3,
        refine_alpha=0.1,
        traj0_aux_weight=0.3,
        # If you want stronger safety, try >0.0, but it may trade off L2.
        traj_risk_weight=0.0,
    )
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
