#!/usr/bin/env bash
# Shared helpers for eval scripts. Source from scripts/eval/*.sh

ensure_bash() {
    if [ -z "${BASH_VERSION:-}" ]; then
        exec bash "$0" "$@"
    fi
}

setup_eval_env() {
    local caller="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
    SCRIPT_DIR=$(cd "$(dirname "$caller")" && pwd)
    REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
    cd "$REPO_ROOT"
    export PYTHONPATH=./
    detect_python
}

detect_python() {
    if [ -n "${PYTHON_BIN:-}" ] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        return
    fi
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        echo "Error: python/python3 not found. Activate your environment first." >&2
        exit 1
    fi
}

# Single-GPU queue: gpu_queue_init <queue_log> <gpu_id> ...
gpu_queue_init() {
    GPU_QUEUE_LOG="$1"
    shift
    GPU_QUEUE_FAILED=0
    GPU_QUEUE_AVAILABLE=("$@")
    GPU_QUEUE_PIDS=()
    GPU_QUEUE_GPUS=()
    GPU_QUEUE_LABELS=()
}

gpu_queue_reap_one() {
    local finished_pid wait_status=0
    wait -n -p finished_pid "${GPU_QUEUE_PIDS[@]}" || wait_status=$?
    [ "$wait_status" -ne 0 ] && GPU_QUEUE_FAILED=1
    local idx
    for idx in "${!GPU_QUEUE_PIDS[@]}"; do
        if [ "${GPU_QUEUE_PIDS[$idx]}" = "$finished_pid" ]; then
            GPU_QUEUE_AVAILABLE+=("${GPU_QUEUE_GPUS[$idx]}")
            echo "[$(date '+%F %T')] Done: ${GPU_QUEUE_LABELS[$idx]} (gpu=${GPU_QUEUE_GPUS[$idx]}, status=${wait_status})" \
                | tee -a "$GPU_QUEUE_LOG"
            unset 'GPU_QUEUE_PIDS[idx]' 'GPU_QUEUE_GPUS[idx]' 'GPU_QUEUE_LABELS[idx]'
            GPU_QUEUE_PIDS=("${GPU_QUEUE_PIDS[@]}")
            GPU_QUEUE_GPUS=("${GPU_QUEUE_GPUS[@]}")
            GPU_QUEUE_LABELS=("${GPU_QUEUE_LABELS[@]}")
            return
        fi
    done
}

# gpu_queue_dispatch <label> <func> [args...]
# Optional: set GPU_QUEUE_USE_TRAP=1 before sourcing to wrap jobs (for ruler/ppl).
gpu_queue_dispatch() {
    local label="$1" func="$2"
    shift 2
    while [ "${#GPU_QUEUE_AVAILABLE[@]}" -eq 0 ]; do
        gpu_queue_reap_one
    done
    local gpu="${GPU_QUEUE_AVAILABLE[0]}"
    GPU_QUEUE_AVAILABLE=("${GPU_QUEUE_AVAILABLE[@]:1}")
    echo "[$(date '+%F %T')] Launch ${label} on gpu=${gpu}" | tee -a "$GPU_QUEUE_LOG"
    if [ "${GPU_QUEUE_USE_TRAP:-0}" = "1" ]; then
        ( trap 'kill_descendants TERM $BASHPID; exit 130' INT TERM
          "$func" "$gpu" "$@" ) &
    else
        "$func" "$gpu" "$@" &
    fi
    GPU_QUEUE_PIDS+=($!)
    GPU_QUEUE_GPUS+=("$gpu")
    GPU_QUEUE_LABELS+=("$label")
}

gpu_queue_wait_all() {
    while [ "${#GPU_QUEUE_PIDS[@]}" -gt 0 ]; do
        gpu_queue_reap_one
    done
}

# Multi-GPU group queue (OpenCompass): gpu_group_queue_init <log> <gpus_per_task> <gpu_id> ...
gpu_group_queue_init() {
    GPU_GROUP_QUEUE_LOG="$1"
    GPU_GROUP_GPUS_PER_TASK="$2"
    shift 2
    GPU_GROUP_QUEUE_FAILED=0
    GPU_GROUP_AVAILABLE=("$@")
    GPU_GROUP_PIDS=()
    GPU_GROUP_GPU_GROUPS=()
    GPU_GROUP_LABELS=()
    GPU_GROUP_REAPED=""
}

gpu_group_queue_pop() {
    local take="$GPU_GROUP_GPUS_PER_TASK"
    local -a taken=("${GPU_GROUP_AVAILABLE[@]:0:take}")
    GPU_GROUP_AVAILABLE=("${GPU_GROUP_AVAILABLE[@]:take}")
    GPU_GROUP_REAPED=$(IFS=,; echo "${taken[*]}")
}

gpu_group_queue_reap_one() {
    local finished_pid wait_status=0
    wait -n -p finished_pid "${GPU_GROUP_PIDS[@]}" || wait_status=$?
    [ "$wait_status" -ne 0 ] && GPU_GROUP_QUEUE_FAILED=1
    local idx
    for idx in "${!GPU_GROUP_PIDS[@]}"; do
        if [ "${GPU_GROUP_PIDS[$idx]}" = "$finished_pid" ]; then
            local -a released=()
            IFS=',' read -r -a released <<< "${GPU_GROUP_GPU_GROUPS[$idx]}"
            GPU_GROUP_AVAILABLE+=("${released[@]}")
            echo "[$(date '+%F %T')] Done: ${GPU_GROUP_LABELS[$idx]} (gpus=${GPU_GROUP_GPU_GROUPS[$idx]}, status=${wait_status})" \
                | tee -a "$GPU_GROUP_QUEUE_LOG"
            unset 'GPU_GROUP_PIDS[idx]' 'GPU_GROUP_GPU_GROUPS[idx]' 'GPU_GROUP_LABELS[idx]'
            GPU_GROUP_PIDS=("${GPU_GROUP_PIDS[@]}")
            GPU_GROUP_GPU_GROUPS=("${GPU_GROUP_GPU_GROUPS[@]}")
            GPU_GROUP_LABELS=("${GPU_GROUP_LABELS[@]}")
            return
        fi
    done
}

# gpu_group_queue_dispatch <label> <func> <args...>
gpu_group_queue_dispatch() {
    local label="$1" func="$2"
    shift 2
    while [ "${#GPU_GROUP_AVAILABLE[@]}" -lt "$GPU_GROUP_GPUS_PER_TASK" ]; do
        gpu_group_queue_reap_one
    done
    gpu_group_queue_pop
    local gpu_group="$GPU_GROUP_REAPED"
    echo "[$(date '+%F %T')] Launch ${label} on gpus=${gpu_group}" | tee -a "$GPU_GROUP_QUEUE_LOG"
    "$func" "$gpu_group" "$@" &
    GPU_GROUP_PIDS+=($!)
    GPU_GROUP_GPU_GROUPS+=("$gpu_group")
    GPU_GROUP_LABELS+=("$label")
}

gpu_group_queue_wait_all() {
    while [ "${#GPU_GROUP_PIDS[@]}" -gt 0 ]; do
        gpu_group_queue_reap_one
    done
}
