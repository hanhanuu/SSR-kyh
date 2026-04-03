#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

WORK_DIR="${WORK_DIR:-work_dirs/ssr_ours1_risk_mlp_nofuse_futureocc_riskw06_g123_12ep}"
CFG="${CFG:-projects/configs/SSR/SSR_e2e_risk_refiner_aspp_futureocc_nofuse_riskw06_12ep_g123.py}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
PORT="${PORT:-29544}"

export PATH="/home/hfut/miniconda3/envs/ssr_legacy/bin:$PATH"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/mmdetection3d:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

mkdir -p "$WORK_DIR"
echo "[`date '+%F %T'`] train start: cfg=$CFG work_dir=$WORK_DIR gpus=$GPU_IDS"
PORT="$PORT" ./tools/dist_train.sh "$CFG" "$NUM_GPUS" --work-dir "$WORK_DIR" 2>&1 | tee "$WORK_DIR/train.log"
