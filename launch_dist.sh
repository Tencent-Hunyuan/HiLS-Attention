#!/bin/bash
# 分布式训练一键启动脚本，只需在 rank-0 节点上运行一次。
#
# 原理:
#   TaiJi 多机任务中，各节点容器之间无法直接 SSH（端口 22 不通），
#   但通过 TaiJi 管理端口（36000）可以用密码登录。
#   本脚本利用 expect 自动输入密码，SSH 到各 worker 节点，
#   以 nohup 后台启动训练进程，并传入正确的 INDEX（节点 rank）和 PATH
#   （worker SSH 进来是裸 shell，没有激活 conda，需要从 rank-0 继承 PATH）。
#   rank-0 自身在本地前台运行，作为 torchrun rendezvous master。
#
# 环境变量（由 init_env.sh / TaiJi 自动注入，无需手动设置）:
#   HOST_NUM     — 总节点数，例如 4
#   NODE_IP_0~N  — 各节点 IP，由 set_env.py 解析 NODE_IP_LIST 写入 ~/.bashrc
#   INDEX        — 当前节点 rank（rank-0 节点上为 0）
#
# 用法:
#   bash launch_dist.sh <训练脚本路径>
#   示例: bash launch_dist.sh scripts/cpt/cpt_olmo3_full_lhsa_20B_fixmlp_dist.sh
#
# 日志:
#   日志名: <训练脚本基名>_rank<节点号>.txt（基名 = basename 去 .sh）
#   rank-0: 同时打印到终端
#   rank-1~N: 在 WORK_DIR 下
#
# 推荐配合 tmux 使用，防止 IDE 断开后训练中断:
#   tmux new -s train
#   bash launch_dist.sh scripts/cpt/cpt_olmo3_full_lhsa_20B_fixmlp_dist.sh
#   # Ctrl+B D 可 detach，训练继续运行；tmux attach -t train 可重连

set -euo pipefail

TRAINING_SCRIPT="${1:-scripts/cpt/cpt_olmo3_full_lhsa_20B_fixmlp_dist.sh}"
# 用于日志文件名: 仅取脚本名、去掉 .sh
TRAINING_LOG_STEM="$(basename "${TRAINING_SCRIPT}" .sh)"
WORK_DIR="${WORK_DIR:-$(pwd)}"
NNODES="${HOST_NUM:?Please set HOST_NUM}"
MASTER_ADDR="${NODE_IP_0:?Please set NODE_IP_0}"
MASTER_PORT="${MASTER_PORT:-26752}"
SSH_PORT="${SSH_PORT:-36000}"
SSH_USER="${SSH_USER:-root}"
SSH_PASS="${SSH_PASS:-epUsleVZYDPXI6b,}"

# SSH helper using expect (no sshpass needed, handles password auth)
ssh_exec() {
    local ip="$1"
    local cmd="$2"
    expect -c "
        set timeout 30
        spawn ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no ${SSH_USER}@${ip} \"${cmd}\"
        expect {
            \"assword:\" { send \"${SSH_PASS}\r\"; exp_continue }
            eof
        }
        catch wait result
        exit [lindex \$result 3]
    "
}

echo "[launch] NNODES=${NNODES}, MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "[launch] WORK_DIR=${WORK_DIR}"
echo "[launch] TRAINING_SCRIPT=${TRAINING_SCRIPT}"
echo ""

# ── Capture the full PATH and key env vars from rank-0's conda env ───────────
# Workers' SSH non-interactive shells won't activate conda, so we pass PATH
# explicitly from rank-0 (same container image, same conda layout).
RANK0_PATH="${PATH}"

# Build the env export string (all NODE_IP_* vars)
NODE_IP_EXPORTS=""
for i in $(seq 0 $((NNODES - 1))); do
    var="NODE_IP_${i}"
    NODE_IP_EXPORTS+="export ${var}=${!var}; "
done

# ── Clean up stale GPU processes on worker nodes before launching ────────────
echo "[launch] Cleaning stale processes on worker nodes..."
for i in $(seq 1 $((NNODES - 1))); do
    var="NODE_IP_${i}"
    ip="${!var}"
    ssh_exec "$ip" "\
        pkill -f 'torchrun' 2>/dev/null; \
        pkill -f 'tasks/pretrain_with_ruler.py' 2>/dev/null; \
        pkill -f 'tasks/pretrain.py' 2>/dev/null; \
        pkill -f 'flash_hsa_run.py' 2>/dev/null; \
        pkill -f 'CPT_dist.sh' 2>/dev/null; \
        pkill -f 'train_dist.sh' 2>/dev/null; \
        true" &
done
wait
echo "[launch] Stale process cleanup done. Waiting 3s for GPU memory release..."
sleep 3

# ── Launch worker nodes (rank 1..N-1) via SSH ────────────────────────────────
for i in $(seq 1 $((NNODES - 1))); do
    var="NODE_IP_${i}"
    ip="${!var}"
    echo "[launch] Starting rank-${i} on ${ip}:${SSH_PORT} ..."

    ssh_exec "$ip" \
        "export PATH='${RANK0_PATH}'; \
         ${NODE_IP_EXPORTS} \
         export HOST_NUM=${NNODES}; \
         export INDEX=${i}; \
         export NODE_IP_0=${MASTER_ADDR}; \
         export MASTER_PORT=${MASTER_PORT}; \
         cd ${WORK_DIR} && \
         nohup bash ${TRAINING_SCRIPT} > ${WORK_DIR}/${TRAINING_LOG_STEM}_rank${i}.txt 2>&1 & \
         echo rank-${i} started PID=\$!" &
done

echo "[launch] Waiting 3s for workers to start..."
sleep 3

# ── Cleanup function: kill worker processes on all remote nodes ─────────────
cleanup_workers() {
    echo ""
    echo "[launch] Stopping workers on remote nodes..."
    for i in $(seq 1 $((NNODES - 1))); do
        var="NODE_IP_${i}"
        ip="${!var}"
        echo "[launch]   killing rank-${i} on ${ip} ..."
        # Kill the training script, CPT_dist.sh, torchrun, and the actual python
        # training processes. The nohup launch creates a process tree that pkill
        # on the top-level script alone may not fully clean up.
        ssh_exec "$ip" "\
            pkill -f '${TRAINING_SCRIPT}' 2>/dev/null; \
            pkill -f 'CPT_dist.sh' 2>/dev/null; \
            pkill -f 'train_dist.sh' 2>/dev/null; \
            pkill -f 'torchrun.*pretrain_with_ruler' 2>/dev/null; \
            pkill -f 'tasks/pretrain_with_ruler.py' 2>/dev/null; \
            pkill -f 'tasks/pretrain.py' 2>/dev/null; \
            pkill -f 'flash_hsa_run.py' 2>/dev/null; \
            true" &
    done
    wait
    echo "[launch] All workers stopped."
}

# Ensure cleanup runs on Ctrl+C (SIGINT), SIGTERM, or normal exit
trap cleanup_workers EXIT

# ── Launch rank-0 locally (foreground) ──────────────────────────────────────
echo "[launch] Starting rank-0 locally ..."
HOST_NUM=${NNODES} \
INDEX=0 \
NODE_IP_0=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
bash "${TRAINING_SCRIPT}" 2>&1 | tee "${WORK_DIR}/${TRAINING_LOG_STEM}_rank0.txt"
