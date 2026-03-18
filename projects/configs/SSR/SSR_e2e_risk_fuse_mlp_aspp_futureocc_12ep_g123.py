_base_ = ['./SSR_e2e_risk_fuse_mlp_aspp_futureocc_12ep.py']

# Wrapper config for naming only
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
