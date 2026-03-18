_base_ = ['./SSR_e2e_risk_refiner_aspp_futureocc_nofuse_12ep.py']

# Stage-2 fine-tuning:
# 1) start from best L2 checkpoint
# 2) smaller lr
# 3) short schedule with linear warmup on safety losses
total_epochs = 3
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

load_from = 'work_dirs/ssr_refiner_only_g123_12ep/epoch_12_ema.pth'
resume_from = None

optimizer = dict(lr=1e-5)
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)

model = dict(
    pts_bbox_head=dict(
        # target safety weight
        risk_loss_weight=0.6,
        # linear warmup by epoch:
        # epoch 0 -> 20%, epoch 1 -> 60%, epoch >=2 -> 100%
        risk_loss_warmup_epochs=2.0,
        risk_loss_warmup_start=0.2,
        # disabled for now; keep explicit for clarity
        traj_risk_weight=0.0,
        traj_risk_warmup_epochs=0.0,
        traj_risk_warmup_start=0.0,
    ))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
