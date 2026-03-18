_base_ = ['./SSR_e2e_risk_fuse_mlp_aspp_12ep.py']

# Wrapper config for ASPP risk head run on GPUs 1/2/3 naming only
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
