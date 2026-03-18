#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${1:-work_dirs/ssr_stage2_l2best_safety_warmup_3ep}"
CFG="${2:-projects/configs/SSR/SSR_e2e_stage2_l2best_safety_warmup_3ep.py}"
GPU_IDS="${3:-0,1,2,3}"
NUM_GPUS="${4:-4}"
MAX_EPOCH="${5:-3}"
POLL_SECS="${6:-60}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

WORK_DIR_ABS="$REPO_DIR/$WORK_DIR"
TAKEOVER_LOG="$WORK_DIR_ABS/tmux_takeover.log"
RESUME_LOG="$WORK_DIR_ABS/train_tmux_resume.log"

mkdir -p "$WORK_DIR_ABS"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$TAKEOVER_LOG"
}

is_training_running() {
  ps -ef | grep -F "tools/train.py" | grep -F -- "--work-dir ${WORK_DIR}" | grep -v grep >/dev/null 2>&1
}

is_finished() {
  [[ -f "$WORK_DIR_ABS/epoch_${MAX_EPOCH}_ema.pth" || -f "$WORK_DIR_ABS/epoch_${MAX_EPOCH}.pth" ]]
}

log "tmux takeover started. work_dir=${WORK_DIR}"

if is_finished; then
  log "Target epoch checkpoint already exists. Nothing to do."
  exit 0
fi

while is_training_running; do
  log "Detected active training process. Waiting ${POLL_SECS}s..."
  sleep "$POLL_SECS"
done

if is_finished; then
  log "Training already finished while waiting."
  exit 0
fi

if [[ ! -f "$WORK_DIR_ABS/latest.pth" ]]; then
  log "No latest.pth found. Cannot resume safely."
  exit 1
fi

log "No active training detected. Resuming from latest.pth in tmux."
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PATH="/home/hfut/miniconda3/envs/ssr_legacy/bin:$PATH"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/mmdetection3d:${PYTHONPATH:-}"

./tools/dist_train.sh "$CFG" "$NUM_GPUS" \
  --work-dir "$WORK_DIR" \
  --resume-from "$WORK_DIR/latest.pth" \
  2>&1 | tee -a "$RESUME_LOG"

log "tmux takeover resume command finished."
