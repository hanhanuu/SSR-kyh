# SSR Paper 12-Epoch Model Index

This index maps the useful 12-epoch models in `work_dirs/paper_metrics_all_ema_unified.csv`
to versioned config/training files in the repository.

## Baseline and Risk-Field Module

| CSV Method | Work Dir | Config |
| --- | --- | --- |
| 原始SSR (Baseline-0) | `work_dirs/SSR_e2e_full_2gpu_lr5e-5` | `projects/configs/SSR/SSR_e2e.py` |
| Baseline-1 (无risk 有MLP) | `work_dirs/ssr_refiner_only_g123_12ep` | `projects/configs/SSR/SSR_e2e_refiner_only_12ep.py` |
| 改进risk二版+MLP (无fuse) | `work_dirs/ssr_ours1_risk_mlp_nofuse_futureocc_g123_12ep` | `projects/configs/SSR/SSR_e2e_risk_refiner_aspp_futureocc_nofuse_12ep_g123.py` |
| 改进risk二版+融合+MLP (riskw0.6) | `work_dirs/ssr_risk_fuse_mlp_aspp_futureocc_riskw06_g123_12ep` | `projects/configs/SSR/SSR_e2e_risk_fuse_mlp_aspp_futureocc_riskw06_12ep_g123.py` |
| 改进risk二版+MLP (无fuse riskw0.6) | `work_dirs/ssr_ours1_risk_mlp_nofuse_futureocc_riskw06_g123_12ep` | `projects/configs/SSR/SSR_e2e_risk_refiner_aspp_futureocc_nofuse_riskw06_12ep_g123.py` |

## Loss-Optimization Module (L2 + Collision)

| CSV Method | Work Dir | Config |
| --- | --- | --- |
| Loss改进-1 (latent_noise_v2) | `work_dirs/ssr_loss1_latent_noise_v2_g123_12ep` | `projects/configs/SSR/SSR_e2e_latent_noise_v2_12ep_g123.py` |
| Baseline + 可微BoxCol损失 + warmup | `work_dirs/ssr_exp2_g1_boxcol_warmup_g123_12ep` | `projects/configs/SSR/SSR_e2e_boxcol_warmup_12ep_g123.py` |
| 组1 + 梯度冲突处理(PCGrad) | `work_dirs/ssr_exp2_g2_boxcol_pcgrad_g123_12ep` | `projects/configs/SSR/SSR_e2e_boxcol_warmup_pcgrad_12ep_g123.py` |

## Key Architecture Files

- `projects/mmdet3d_plugin/SSR/SSR.py`
- `projects/mmdet3d_plugin/SSR/SSR_head.py`

## Key Training Scripts

- `tools/run_loss1_latent_noise_v2_train_12ep.sh`
- `tools/run_exp2_group1_boxcol_warmup_12ep.sh`
- `tools/run_exp2_group1_boxcol_pcgrad_12ep.sh`
- `tools/run_risk_nofuse_riskw06_12ep.sh`
