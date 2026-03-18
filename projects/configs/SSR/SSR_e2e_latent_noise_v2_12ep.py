_base_ = ['./SSR_e2e.py']

# Loss experiment group: latent decomposition + fog-aware noise mask.
# This replaces baseline world-model MSE loss by latent-noise BEV objective.
model = dict(
    wm_loss_mode='latent_noise_fog',
    wm_fog_alpha_range=(0.0, 0.06),
    wm_voxel_size=0.15,
    wm_latent_noise_weights=(1.0, 0.2, 0.01),
)
