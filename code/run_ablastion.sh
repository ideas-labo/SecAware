#!/bin/bash

# 🔥 消融实验批量运行脚本（并行版本）

echo "================================"
echo "Starting Ablation Study (Parallel)"
echo "================================"

# 基础配置
TARGET_MODEL="llama3-8B-instruct/"
VLLM_SERVER_URL="http://localhost:8001/v1"
MAX_QUERY=1000
SELECT_POLICY="mcts_normalized"

# 🔥 修改：使用固定的消融实验目录
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_BASE_DIR="${BASE_DIR}/logs/ablation_results/llama3"

# 创建时间戳子目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ABLATION_DIR="${ABLATION_BASE_DIR}/${TIMESTAMP}"
mkdir -p "${ABLATION_DIR}"

# 🔥 为每个配置创建独立子目录
mkdir -p "${ABLATION_DIR}/logs"
mkdir -p "${ABLATION_DIR}/results"

echo "Results will be saved to: ${ABLATION_DIR}"
echo ""

# 🔥 配置列表（包含新增的 no_anchors_no_cwe）
# CONFIGS=("full" "no_anchors" "no_cwe" "no_operators" "no_anchors_no_cwe")
CONFIGS=("no_anchors" "no_cwe" "no_anchors_no_cwe")

# 🔥 
declare -a PIDS

# 🔥 并行启动所有配置
echo "================================"
echo "Launching experiments in parallel..."
echo "================================"

for CONFIG in "${CONFIGS[@]}"; do
    echo "🚀 Starting: ${CONFIG}"
    
    # 🔥 日志文件保存在 logs 子目录
    LOG_FILE="${ABLATION_DIR}/logs/${CONFIG}.log"
    
    # 🔥 使用 nohup 在后台运行，防止连接断开
    nohup python run_secaware.py \
        --target_backend vllm_server \
        --target_model ${TARGET_MODEL} \
        --vllm_server_url ${VLLM_SERVER_URL} \
        --select_policy ${SELECT_POLICY} \
        --max_query ${MAX_QUERY} \
        --ablation_config ${CONFIG} \
        > "${LOG_FILE}" 2>&1 &
    
    # 记录PID
    PID=$!
    PIDS+=($PID)
    
    echo "   PID: ${PID}"
    echo "   Log: ${LOG_FILE}"
    
    # 短暂延迟，避免资源争抢
    sleep 2
done

echo ""
echo "================================"
echo "All experiments launched!"
echo "================================"
echo "PIDs: ${PIDS[@]}"
echo ""

# 🔥 创建进度监控脚本
MONITOR_SCRIPT="${ABLATION_DIR}/monitor_progress.sh"
cat > "${MONITOR_SCRIPT}" << 'MONITOR_EOF'
#!/bin/bash
ABLATION_DIR="__ABLATION_DIR__"
PIDS=(__PIDS__)
CONFIGS=(__CONFIGS__)

while true; do
    clear
    echo "================================"
    echo "Ablation Study Progress Monitor"
    echo "================================"
    date
    echo ""
    
    ALL_DONE=true
    for i in "${!PIDS[@]}"; do
        PID=${PIDS[$i]}
        CONFIG=${CONFIGS[$i]}
        
        if ps -p ${PID} > /dev/null 2>&1; then
            echo "⏳ ${CONFIG} (PID: ${PID}) - Running"
            ALL_DONE=false
        else
            echo "✅ ${CONFIG} (PID: ${PID}) - Completed"
        fi
    done
    
    echo ""
    echo "================================"
    echo "Log files:"
    ls -lh "${ABLATION_DIR}/logs/"
    echo ""
    
    if [ "$ALL_DONE" = true ]; then
        echo "🎉 All experiments completed!"
        break
    fi
    
    echo "Press Ctrl+C to stop monitoring (experiments will continue)"
    sleep 10
done
MONITOR_EOF

# 替换占位符
sed -i "s|__ABLATION_DIR__|${ABLATION_DIR}|g" "${MONITOR_SCRIPT}"
sed -i "s|__PIDS__|${PIDS[@]}|g" "${MONITOR_SCRIPT}"
sed -i "s|__CONFIGS__|${CONFIGS[@]}|g" "${MONITOR_SCRIPT}"
chmod +x "${MONITOR_SCRIPT}"

echo "📊 Progress monitor script created: ${MONITOR_SCRIPT}"
echo ""

# 🔥 创建实验摘要（包含PID信息）
SUMMARY_FILE="${ABLATION_DIR}/experiment_summary.txt"
cat > "${SUMMARY_FILE}" << EOF
Ablation Study Summary
======================
Timestamp: ${TIMESTAMP}
Target Model: ${TARGET_MODEL}
Max Query: ${MAX_QUERY}
Select Policy: ${SELECT_POLICY}
Execution Mode: Parallel

Configurations & PIDs:
EOF

for i in "${!CONFIGS[@]}"; do
    echo "  - ${CONFIGS[$i]} (PID: ${PIDS[$i]})" >> "${SUMMARY_FILE}"
done

cat >> "${SUMMARY_FILE}" << EOF

Configuration Details:
  - full                : 完整版本 (Anchors ✅, CWE ✅, Operators ✅)
  - no_anchors          : 移除 Semantic Anchors (Anchors ❌, CWE ✅, Operators ✅)
  - no_cwe              : 移除 CWE 知识 (Anchors ✅, CWE ❌, Operators ✅)
  - no_operators        : 移除变异算子 (Anchors ✅, CWE ✅, Operators ❌)
  - no_anchors_no_cwe   : 移除 Anchors + CWE (Anchors ❌, CWE ❌, Operators ✅)

Directory Structure:
  - logs/     : Execution logs for each configuration
  - results/  : CSV results for each configuration

To monitor progress, run:
  ${MONITOR_SCRIPT}

To check if experiments are still running:
  ps -p ${PIDS[@]}

To kill all experiments:
  kill ${PIDS[@]}

To analyze results (after completion), run:
  python analyze_ablation_results.py --results_dir "${ABLATION_DIR}/results"
EOF

echo "📄 Experiment summary saved to: ${SUMMARY_FILE}"
echo ""

# 🔥 询问是否启动监控
read -p "Start progress monitor? (y/N): " START_MONITOR

if [ "${START_MONITOR}" = "y" ] || [ "${START_MONITOR}" = "Y" ]; then
    echo ""
    echo "Starting progress monitor..."
    exec "${MONITOR_SCRIPT}"
else
    echo ""
    echo "================================"
    echo "Experiments running in background"
    echo "================================"
    echo "To monitor progress manually:"
    echo "  bash ${MONITOR_SCRIPT}"
    echo ""
    echo "To check logs in real-time:"
    echo "  tail -f ${ABLATION_DIR}/logs/*.log"
    echo ""
    echo "To wait for all experiments to complete:"
    echo "  wait ${PIDS[@]}"
    echo ""
    echo "Experiment directory: ${ABLATION_DIR}"
    echo "================================"
fi