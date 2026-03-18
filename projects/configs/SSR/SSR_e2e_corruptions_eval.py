import os
from pathlib import Path

# Generic wrapper for corruption robustness evaluation.
#
# Usage:
#   export SSR_BASE_CFG=projects/configs/SSR/SSR_e2e_risk_fuse_mlp_12ep_g123.py
#   export NUSC_CORRUPT_INFO=data/corruptions/vad_nuscenes_infos_temporal_val_fog3.pkl
#   ./tools/dist_test.sh projects/configs/SSR/SSR_e2e_corruptions_eval.py <ckpt>.pth 4 --out <out>.pkl
#
# If SSR_BASE_CFG is not set, it defaults to the main (Ours-2) config.
_base_cfg = os.environ.get('SSR_BASE_CFG', './SSR_e2e_risk_fuse_mlp_12ep_g123.py')
if os.path.isabs(_base_cfg):
    _base_path = Path(_base_cfg)
elif _base_cfg.replace("\\", "/").startswith("projects/"):
    _base_path = Path(os.getcwd()) / _base_cfg
else:
    _base_path = Path(__file__).resolve().parent / _base_cfg
_base_ = [str(_base_path.resolve())]

corrupt_info = os.environ.get('NUSC_CORRUPT_INFO', None)
if corrupt_info is None:
    raise ValueError(
        'Set env var NUSC_CORRUPT_INFO to the rewritten info PKL path, e.g. '
        'data/corruptions/vad_nuscenes_infos_temporal_val_fog3.pkl'
    )

data = dict(
    val=dict(ann_file=corrupt_info),
    test=dict(ann_file=corrupt_info),
)

# Cleanup to avoid mmcv deepcopy hitting module objects
del os, Path
