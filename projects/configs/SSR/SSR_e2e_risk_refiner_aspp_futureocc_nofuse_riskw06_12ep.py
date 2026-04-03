_base_ = ['./SSR_e2e_risk_refiner_aspp_futureocc_nofuse_12ep.py']

# 12-epoch no-fusion risk+MLP variant with lower risk supervision weight.
model = dict(
    pts_bbox_head=dict(
        risk_loss_weight=0.6,
    ),
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
