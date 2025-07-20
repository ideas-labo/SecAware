CUDA_VISIBLE_DEVICES='0,1' nohup python3 -u run.py \
    --api_key 'sk-370d60e20fd840b18356cdc9e135e530' \
    --model_path qwen2.5-7b-instruct-1m\
    --target_model "codellama:7b" \
    --select_policy mcts_cwe \
    --max_query 1000 \
    > "./logs/codellama_7b_query1000.log" 2>&1 &