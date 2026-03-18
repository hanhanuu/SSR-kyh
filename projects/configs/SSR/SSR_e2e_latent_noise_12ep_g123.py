_base_ = ['./SSR_e2e_latent_noise_12ep.py']

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
