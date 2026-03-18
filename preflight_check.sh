#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_LINK="${ROOT_DIR}/data/nuscenes"
CAN_BUS_DIR="${ROOT_DIR}/data/can_bus"

WATCH_MODE=0
WATCH_INTERVAL="${WATCH_INTERVAL:-60}"
LOG_FILE=""
ALERT_ON_FAIL="${ALERT_ON_FAIL:-1}"
ALERT_BELL="${ALERT_BELL:-1}"
ALERT_CMD="${ALERT_CMD:-}"

for arg in "$@"; do
  case "$arg" in
    --watch)
      WATCH_MODE=1
      ;;
    --interval=*)
      WATCH_INTERVAL="${arg#*=}"
      ;;
    --log=*)
      LOG_FILE="${arg#*=}"
      ;;
  esac
done

run_checks() {
missing=0

echo "=== SSR Preflight Check ==="
echo "Repo: ${ROOT_DIR}"
echo "Time: $(date '+%F %T')"
echo

# Resolve data path
if [ -L "${DATA_LINK}" ]; then
  DATA_TARGET="$(readlink -f "${DATA_LINK}")"
  echo "Data link: ${DATA_LINK} -> ${DATA_TARGET}"
elif [ -d "${DATA_LINK}" ]; then
  DATA_TARGET="${DATA_LINK}"
  echo "Data dir: ${DATA_TARGET}"
else
  echo "FAIL: data/nuscenes does not exist."
  missing=$((missing+1))
  DATA_TARGET=""
fi

echo
if [ -n "${DATA_TARGET}" ]; then
  for d in "v1.0-trainval" "samples" "sweeps" "maps"; do
    if [ -d "${DATA_TARGET}/${d}" ]; then
      echo "PASS: ${d} exists."
    else
      echo "FAIL: ${d} missing in ${DATA_TARGET}."
      missing=$((missing+1))
    fi
  done
fi

echo
if [ -d "${CAN_BUS_DIR}" ]; then
  echo "PASS: can_bus directory exists."
else
  echo "WARN: can_bus directory missing at ${CAN_BUS_DIR}."
fi

echo
if [ -n "${DATA_TARGET}" ]; then
  for f in "vad_nuscenes_infos_temporal_train.pkl" "vad_nuscenes_infos_temporal_val.pkl"; do
    if [ -s "${DATA_TARGET}/${f}" ]; then
      echo "PASS: ${f} exists."
    else
      echo "FAIL: ${f} missing."
      missing=$((missing+1))
    fi
  done
fi

echo
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU:"
  nvidia-smi -L || true
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  fi
else
  echo "WARN: nvidia-smi not found."
fi

echo
echo "Training process:"
TRAIN_PIDS="$(pgrep -f "tools/train.py" || true)"
if [ -n "${TRAIN_PIDS}" ]; then
  echo "PASS: train.py running (PIDs: ${TRAIN_PIDS})"
else
  echo "WARN: train.py not running."
fi

echo
echo "GPU usage check (auto-stop on single-GPU degradation):"
EXPECTED_GPUS="${EXPECTED_GPUS:-}"
if [ -z "${EXPECTED_GPUS}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  EXPECTED_GPUS="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
fi
AUTO_STOP_SINGLE_GPU="${AUTO_STOP_SINGLE_GPU:-1}"
ACTIVE_GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  ACTIVE_GPU_COUNT="$(nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader 2>/dev/null \
    | awk -F',' '$2 ~ /python/ {print $1}' | sort -u | wc -l | tr -d ' ')"
fi
if [ -n "${EXPECTED_GPUS}" ]; then
  echo "Expected GPUs: ${EXPECTED_GPUS}, Active GPUs: ${ACTIVE_GPU_COUNT}"
  if [ -n "${TRAIN_PIDS}" ] && [ "${EXPECTED_GPUS}" -ge 2 ] && [ "${ACTIVE_GPU_COUNT}" -gt 0 ] && [ "${ACTIVE_GPU_COUNT}" -lt "${EXPECTED_GPUS}" ]; then
    echo "FAIL: GPU count degraded (active ${ACTIVE_GPU_COUNT} < expected ${EXPECTED_GPUS})."
    if [ "${AUTO_STOP_SINGLE_GPU}" -eq 1 ]; then
      echo "AUTO-STOP: killing training processes."
      pkill -f "tools/train.py" || true
    else
      echo "AUTO-STOP disabled (set AUTO_STOP_SINGLE_GPU=1 to enable)."
    fi
  else
    echo "PASS: GPU count OK or training not running."
  fi
else
  echo "INFO: EXPECTED_GPUS not set and CUDA_VISIBLE_DEVICES not provided; skipping degradation check."
fi

echo
echo "Latest training log:"
LOG_DIR="${ROOT_DIR}/work_dirs/SSR_e2e_full_2gpu_lr5e-5"
if [ -d "${LOG_DIR}" ]; then
  LATEST_LOG="$(ls -t "${LOG_DIR}"/*.log 2>/dev/null | head -n 1 || true)"
  if [ -n "${LATEST_LOG}" ]; then
    echo "Latest log: ${LATEST_LOG}"
    LAST_EPOCH_LINE="$(grep -a "Epoch \\[" "${LATEST_LOG}" | tail -n 1 || true)"
    if [ -n "${LAST_EPOCH_LINE}" ]; then
      echo "Last Epoch line:"
      echo "${LAST_EPOCH_LINE}"
    else
      echo "WARN: no Epoch lines found in latest log."
    fi
  else
    echo "WARN: no .log files found in ${LOG_DIR}"
  fi
else
  echo "WARN: log dir not found: ${LOG_DIR}"
fi

echo
if [ "${missing}" -eq 0 ]; then
  echo "Preflight: PASS"
  return 0
else
  echo "Preflight: FAIL (missing ${missing} items)"
  return 1
fi
}

run_and_report() {
  if [ -n "${LOG_FILE}" ]; then
    run_checks 2>&1 | tee -a "${LOG_FILE}"
    STATUS=${PIPESTATUS[0]}
  else
    run_checks
    STATUS=$?
  fi

  if [ "${STATUS}" -ne 0 ] && [ "${ALERT_ON_FAIL}" -eq 1 ]; then
    echo "ALERT: preflight check failed."
    if [ "${ALERT_BELL}" -eq 1 ]; then
      printf '\a'
    fi
    if [ -n "${ALERT_CMD}" ]; then
      eval "${ALERT_CMD}" || true
    fi
  fi
  return "${STATUS}"
}

if [ "${WATCH_MODE}" -eq 1 ]; then
  if [ -z "${LOG_FILE}" ]; then
    LOG_FILE="${ROOT_DIR}/preflight_watch.log"
  fi
  while true; do
    run_and_report || true
    echo
    echo "Watch: sleeping ${WATCH_INTERVAL}s..."
    sleep "${WATCH_INTERVAL}"
    echo
  done
else
  run_and_report
fi
