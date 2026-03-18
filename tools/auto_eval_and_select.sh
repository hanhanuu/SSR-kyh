#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=""
BASE_CFG=""
GPU_IDS="0,1,2,3"
NUM_GPUS=4
MAX_EPOCH=3
INFO_PKL="data/nuscenes/vad_nuscenes_infos_temporal_val.pkl"
EVAL_CFG="projects/configs/SSR/SSR_e2e_corruptions_eval.py"
WEIGHTS="0.4,0.3,0.3"
POLL_SECS=60
WAIT_FOR_TRAIN=1

usage() {
  cat <<'EOF'
Usage:
  tools/auto_eval_and_select.sh \
    --work-dir <work_dir> \
    --base-cfg <base_cfg> \
    [--gpu-ids 0,1,2,3] \
    [--num-gpus 4] \
    [--max-epoch 3] \
    [--info-pkl data/nuscenes/vad_nuscenes_infos_temporal_val.pkl] \
    [--weights 0.4,0.3,0.3] \
    [--poll-secs 60] \
    [--no-wait-train]

Description:
  1) Waits until training for the given work_dir is finished (default).
  2) Evaluates each epoch checkpoint (prefer epoch_X_ema.pth).
  3) Computes planning metrics.txt for each epoch.
  4) Runs tools/select_best_plan_checkpoint.py to select final best checkpoint.
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

abs_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s\n' "$PWD/$p"
  fi
}

train_running() {
  ps -ef | grep -F "tools/train.py" | grep -F -- "--work-dir ${WORK_DIR}" | grep -v grep >/dev/null 2>&1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --base-cfg) BASE_CFG="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --max-epoch) MAX_EPOCH="$2"; shift 2 ;;
    --info-pkl) INFO_PKL="$2"; shift 2 ;;
    --eval-cfg) EVAL_CFG="$2"; shift 2 ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --poll-secs) POLL_SECS="$2"; shift 2 ;;
    --no-wait-train) WAIT_FOR_TRAIN=0; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$WORK_DIR" || -z "$BASE_CFG" ]]; then
  echo "ERROR: --work-dir and --base-cfg are required" >&2
  usage
  exit 2
fi

WORK_DIR="$(abs_path "$WORK_DIR")"
BASE_CFG="$(abs_path "$BASE_CFG")"
INFO_PKL="$(abs_path "$INFO_PKL")"
EVAL_CFG="$(abs_path "$EVAL_CFG")"

if [[ ! -d "$WORK_DIR" ]]; then
  echo "ERROR: work-dir not found: $WORK_DIR" >&2
  exit 2
fi
if [[ ! -f "$BASE_CFG" ]]; then
  echo "ERROR: base cfg not found: $BASE_CFG" >&2
  exit 2
fi
if [[ ! -f "$INFO_PKL" ]]; then
  echo "ERROR: info pkl not found: $INFO_PKL" >&2
  exit 2
fi
if [[ ! -f "$EVAL_CFG" ]]; then
  echo "ERROR: eval cfg not found: $EVAL_CFG" >&2
  exit 2
fi

export PATH="/home/hfut/miniconda3/envs/ssr_legacy/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/mmdetection3d:${PYTHONPATH:-}"

if [[ "$WAIT_FOR_TRAIN" -eq 1 ]]; then
  log "Waiting for training to finish for work-dir: $WORK_DIR"
  while true; do
    done_ckpt="$WORK_DIR/epoch_${MAX_EPOCH}_ema.pth"
    if [[ ! -f "$done_ckpt" ]]; then
      done_ckpt="$WORK_DIR/epoch_${MAX_EPOCH}.pth"
    fi
    if [[ -f "$done_ckpt" ]] && ! train_running; then
      log "Detected training complete and checkpoint ready: $done_ckpt"
      break
    fi
    sleep "$POLL_SECS"
  done
fi

for epoch in $(seq 1 "$MAX_EPOCH"); do
  ckpt="$WORK_DIR/epoch_${epoch}_ema.pth"
  tag="ep${epoch}ema"
  if [[ ! -f "$ckpt" ]]; then
    ckpt="$WORK_DIR/epoch_${epoch}.pth"
    tag="ep${epoch}"
  fi
  if [[ ! -f "$ckpt" ]]; then
    log "Skip epoch ${epoch}: checkpoint not found."
    continue
  fi

  eval_dir="$WORK_DIR/eval_nuscenes_${tag}_${NUM_GPUS}gpu"
  mkdir -p "$eval_dir"
  metric_file="$eval_dir/metrics.txt"
  if [[ -s "$metric_file" ]]; then
    log "Skip epoch ${epoch}: metrics already exists -> $metric_file"
    continue
  fi

  log "Evaluating checkpoint: $ckpt"
  (
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    export SSR_BASE_CFG="$BASE_CFG"
    export NUSC_CORRUPT_INFO="$INFO_PKL"
    ./tools/dist_test.sh "$EVAL_CFG" "$ckpt" "$NUM_GPUS" --out "$eval_dir/results.pkl" \
      2>&1 | tee "$eval_dir/test.log"
  )

  log "Computing planning metrics: $eval_dir/results.pkl"
  python tools/compute_plan_metrics.py \
    --results "$eval_dir/results.pkl" \
    --info "$INFO_PKL" \
    | tee "$metric_file"
done

ranking_csv="$WORK_DIR/checkpoint_ranking.csv"
best_log="$WORK_DIR/select_best.log"
log "Selecting best checkpoint with weights: $WEIGHTS"
python tools/select_best_plan_checkpoint.py \
  --work-dir "$WORK_DIR" \
  --glob 'eval_nuscenes_ep*ema_*gpu/metrics.txt,eval_nuscenes_ep*_*gpu/metrics.txt,metrics_epoch*.txt' \
  --weights "$WEIGHTS" \
  --output-csv "$ranking_csv" \
  | tee "$best_log"

python - "$ranking_csv" "$WORK_DIR/best_checkpoint.txt" <<'PY'
import csv
import sys

csv_path = sys.argv[1]
out_path = sys.argv[2]
best = None
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        best = row
        break

with open(out_path, 'w', encoding='utf-8') as f:
    if best is None:
        f.write('N/A\n')
    else:
        ckpt = best.get('checkpoint', '').strip() or 'N/A'
        f.write(f'{ckpt}\n')
print(f'saved best checkpoint path to: {out_path}')
PY

log "Auto eval + selection finished."
