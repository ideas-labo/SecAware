# Use LocalVLLMServer as target model
# CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server   --model ./Qwen2.5-Coder-32B-Instruct/ --tensor-parallel-size 2   --port 8000

# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
#  --model llama3-8B-instruct/ \
#  --host 0.0.0.0 \
#   --port 8001 \
#   --gpu-memory-utilization 0.90 \
#   --max-model-len 4096

nohup python run_secaware.py \
    --target_backend vllm_server \
    --target_model phi4-14b/ \
    --vllm_server_url http://localhost:8000/v1 \
    --select_policy mcts_normalized \
    --ablation_config full \
    --novelty_mode log \
    --enable_monitoring \
    > "./logs/run1.log" 2>&1 &


# # 比较不同策略
# python code/run_secaware.py --select_policy mcts     # baseline
# python code/run_secaware.py --select_policy ucb
# python code/run_secaware.py --select_policy mcts_novelty  
# python code/run_secaware.py --select_policy mcts_normalized  