_base_ = ['./SSR_e2e_risk_refiner_aspp_futureocc_nofuse_riskw06_12ep.py']

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
