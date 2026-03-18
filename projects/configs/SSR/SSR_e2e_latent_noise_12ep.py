_base_ = ['./SSR_e2e.py']

# Loss ablation-1: latent decomposition + fog-aware noise mask consistency.
model = dict(
    wm_loss_mode='latent_noise_fog',
    wm_fog_alpha_range=(0.0, 0.06),
    wm_voxel_size=0.15,
)
