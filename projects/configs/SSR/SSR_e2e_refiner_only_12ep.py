_base_ = ['./SSR_e2e.py']

# Full training: 12 epochs, MLP refiner only (no risk head)
total_epochs = 12
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

model = dict(
    pts_bbox_head=dict(
        # risk head disabled
        use_risk_head=False,
        risk_fusion_mode='none',
        # refiner (MLP) uses BEV only
        use_traj_refiner=True,
        traj_refiner_type='mlp',
        refiner_use_risk=False,
        refiner_use_bev=True,
        refiner_mlp_hidden=512,
        refine_steps=3,
        refine_alpha=0.1,
        traj0_aux_weight=0.3,
    )
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
