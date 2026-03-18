_base_ = ['./SSR_e2e_risk_fuse_mlp_aspp_futureocc_12ep.py']

# World-model BEV consistency loss: uncertainty-weighted Gaussian NLL (+ optional fog prior).
model = dict(
    wm_loss_mode='gaussian_nll_fog_prior',
    wm_use_fog_prior=True,
    wm_fog_alpha_range=(0.0, 0.02),
    wm_voxel_size=0.15,
    wm_logvar_clamp=(-6.0, 2.0),
)

