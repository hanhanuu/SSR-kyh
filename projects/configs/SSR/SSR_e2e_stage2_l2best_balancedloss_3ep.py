_base_ = ['./SSR_e2e_risk_refiner_aspp_futureocc_nofuse_12ep.py']

# Stage-2 balanced loss tuning:
# 1) start from best stage-2 EMA checkpoint
# 2) lower lr for stable refinement
# 3) add light trajectory risk regularization to reduce collision metrics
total_epochs = 3
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

load_from = 'work_dirs/ssr_stage2_l2best_safety_warmup_3ep/epoch_3_ema.pth'
resume_from = None

optimizer = dict(lr=8e-6)
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)

model = dict(
    pts_bbox_head=dict(
        # keep risk supervision but slightly lower direct risk-map loss
        risk_loss_weight=0.55,
        risk_loss_warmup_epochs=1.5,
        risk_loss_warmup_start=0.3,
        # add light trajectory-level safety regularization
        traj_risk_weight=0.08,
        traj_risk_warmup_epochs=2.0,
        traj_risk_warmup_start=0.0,
        # slightly lower aux weight to focus final refined trajectory
        traj0_aux_weight=0.25,
    ))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
