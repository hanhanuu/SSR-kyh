_base_ = ['./SSR_e2e_1gpu_1ep.py']

# Set epochs here to keep runner.max_epochs == total_epochs
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# Main ablation: Risk head + BEV concat fusion + MLP refiner
model = dict(
    pts_bbox_head=dict(
        # risk head
        use_risk_head=True,
        risk_head_type='conv',
        risk_out_act='softplus',
        risk_supervision='pseudo',
        risk_loss_weight=1.0,
        loss_risk=dict(type='MSELoss', loss_weight=1.0),
        risk_fusion_mode='concat',  # 'none' | 'concat' | 'gated'
        # refiner (MLP)
        use_traj_refiner=True,
        traj_refiner_type='mlp',  # new option
        refiner_use_risk=True,
        refiner_use_bev=True,
        refiner_mlp_hidden=512,
        refine_steps=3,
        refine_alpha=0.1,
        traj0_aux_weight=0.3,  # supervise traj0 lightly
        traj_risk_weight=0.0,
    )
)

# Tighter logging for quick sanity run
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
