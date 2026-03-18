_base_ = ['./SSR_e2e_risk_fuse_mlp_12ep.py']

# Wrapper config: identical to risk_fuse_mlp_12ep, kept for distinct work_dir naming
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
