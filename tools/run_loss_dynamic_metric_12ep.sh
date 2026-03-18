#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

WORK_DIR="${WORK_DIR:-work_dirs/ssr_loss_dynamic_metric_g123_12ep}"
CFG="${CFG:-projects/configs/SSR/SSR_e2e_dynamic_metric_12ep_g123.py}"
EVAL_CFG="${EVAL_CFG:-projects/configs/SSR/SSR_e2e_corruptions_eval.py}"
CSV_PATH="${CSV_PATH:-work_dirs/paper_metrics_all_ema_unified.csv}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
TARGET_EPOCH="${TARGET_EPOCH:-12}"

NUSC_INFO="${NUSC_INFO:-data/nuscenes/vad_nuscenes_infos_temporal_val.pkl}"
FOG3_INFO="${FOG3_INFO:-data/corruptions/vad_nuscenes_infos_temporal_val_fog3_cam.pkl}"

METHOD_NAME="${METHOD_NAME:-Loss改进-动态加权(L2+点碰撞+盒碰撞)}"
COMMENT_NOTE="${COMMENT_NOTE:-自动追加（GPU0,1,2,3；dynamic_metric 12ep）}"

export PATH="/home/hfut/miniconda3/envs/ssr_legacy/bin:$PATH"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/mmdetection3d:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

mkdir -p "$WORK_DIR"

echo "[`date '+%F %T'`] Training start: $WORK_DIR"
./tools/dist_train.sh "$CFG" "$NUM_GPUS" --work-dir "$WORK_DIR" 2>&1 | tee "$WORK_DIR/train.log"

if [[ -f "$WORK_DIR/epoch_${TARGET_EPOCH}_ema.pth" ]]; then
  CKPT="$WORK_DIR/epoch_${TARGET_EPOCH}_ema.pth"
  CKPT_STD="epoch_${TARGET_EPOCH}_ema"
  TAG="ep${TARGET_EPOCH}ema"
else
  CKPT="$WORK_DIR/epoch_${TARGET_EPOCH}.pth"
  CKPT_STD="epoch_${TARGET_EPOCH}"
  TAG="ep${TARGET_EPOCH}"
fi

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: target checkpoint not found: $CKPT" >&2
  exit 1
fi

BASE_CFG="$REPO_DIR/$WORK_DIR/$(basename "$CFG")"
if [[ ! -f "$BASE_CFG" ]]; then
  BASE_CFG="$REPO_DIR/$CFG"
fi

run_eval() {
  local split_name="$1"
  local corr_info="$2"
  local eval_dir="$3"

  mkdir -p "$eval_dir"
  echo "[`date '+%F %T'`] Eval start: ${split_name}"
  SSR_BASE_CFG="$BASE_CFG" \
  NUSC_CORRUPT_INFO="$corr_info" \
    ./tools/dist_test.sh "$EVAL_CFG" "$CKPT" "$NUM_GPUS" --out "$eval_dir/results.pkl" \
    2>&1 | tee "$eval_dir/test.log"

  python tools/compute_plan_metrics.py \
    --results "$eval_dir/results.pkl" \
    --info "$NUSC_INFO" \
    | tee "$eval_dir/metrics.txt"
}

EVAL_NUSC="$WORK_DIR/eval_nuscenes_${TAG}_${NUM_GPUS}gpu"
EVAL_FOG3="$WORK_DIR/eval_fog3_${TAG}_${NUM_GPUS}gpu"

run_eval "nuScenes" "$NUSC_INFO" "$EVAL_NUSC"
run_eval "fog3" "$FOG3_INFO" "$EVAL_FOG3"

python tools/append_plan_metrics_to_unified_csv.py \
  --csv "$CSV_PATH" \
  --metrics "$EVAL_NUSC/metrics.txt" \
  --method "$METHOD_NAME" \
  --dataset "nuScenes" \
  --checkpoint "$CKPT_STD" \
  --source "$EVAL_NUSC/metrics.txt" \
  --comment "$COMMENT_NOTE"

python tools/append_plan_metrics_to_unified_csv.py \
  --csv "$CSV_PATH" \
  --metrics "$EVAL_FOG3/metrics.txt" \
  --method "$METHOD_NAME" \
  --dataset "fog3" \
  --checkpoint "$CKPT_STD" \
  --source "$EVAL_FOG3/metrics.txt" \
  --comment "$COMMENT_NOTE"

echo "[`date '+%F %T'`] Pipeline done. CKPT=${CKPT}"
