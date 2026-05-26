import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from fastchat.model import add_model_args
import argparse
import pandas as pd
from fuzzer.selection import (
    RandomSelectPolicy, 
    UCBSelectPolicy, 
    MCTSExploreSelectPolicy, 
    MCTSExploreCWESelectPolicy,
    MCTSNoveltyWeightedSelectPolicy,
    MCTSNormalizedRewardSelectPolicy
)
from fuzzer.secaware import Fuzzer
from fuzzer.new_mutators import MutationConfig, get_ablation_configs, print_ablation_summary
from llm import QwenLLM, LocalOllamaLLM, LocalLLM, LocalVLLMServer
import random
random.seed(100)
import logging
httpx_logger: logging.Logger = logging.getLogger("httpx")
import asyncio
httpx_logger.setLevel(logging.WARNING)
import datetime

async def main(args):
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_results_dir = os.path.join(current_dir, 'results')
    
    target_model_safe = args.target_model.replace(':', '-').replace('/', '-')
    
    run_dir = os.path.join(project_results_dir, target_model_safe, f'run_{run_timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    result_path = os.path.join(run_dir, f'results.csv')
    
    monitoring_dir = os.path.join(run_dir, 'monitoring')
    os.makedirs(monitoring_dir, exist_ok=True)

    print(f"Current directory: {current_dir}")
    print(f"Run directory: {run_dir}")
    print(f"Results will be saved to: {result_path}")
    print(f"Monitoring directory: {monitoring_dir}")

    templates = pd.read_csv(args.template_path)['text'].tolist()
    
    task_file = args.task_path
    if not os.path.exists(task_file):
        raise FileNotFoundError(f"Task file/directory not found: {task_file}")
    if os.path.isdir(task_file):
        task_file = os.path.join(task_file, 'seed.json')
        if not os.path.exists(task_file):
            raise FileNotFoundError(f"seed.json not found in {args.task_path}")

    mutator_model = QwenLLM(args.model_path, args.api_key, temperature=args.mutator_temperature)

    prompt_mutator_model = None
    if args.prompt_mutator_model:
        if args.prompt_mutator_backend == "qwen":
            prompt_mutator_model = QwenLLM(
                args.prompt_mutator_model, 
                args.prompt_mutator_api_key or args.api_key,
                temperature=args.mutator_temperature
            )
        elif args.prompt_mutator_backend == "vllm_server":
            prompt_mutator_model = LocalVLLMServer(
                base_url=args.vllm_server_url,
                model_name=args.prompt_mutator_model,
                api_key="EMPTY"
            )
        elif args.prompt_mutator_backend == "local":
            prompt_mutator_model = LocalLLM(args.prompt_mutator_model)
        else:
            prompt_mutator_model = LocalOllamaLLM(args.prompt_mutator_model)
    else:
        prompt_mutator_model = mutator_model
        logging.info("PromptMutator will use the same model as traditional mutators")

    if args.target_backend == "vllm_server":
        target_model = LocalVLLMServer(
            base_url=args.vllm_server_url,
            model_name=args.target_model,
            api_key="EMPTY"
        )
    elif args.target_backend == "local":
        target_model = LocalLLM(args.target_model)
    elif args.target_backend == "qwen":
        target_model = QwenLLM(args.target_model, args.target_api_key or args.api_key)
    else:
        target_model = LocalOllamaLLM(args.target_model)

    if args.select_policy == 'ucb':
        select_policy = UCBSelectPolicy()
    elif args.select_policy == 'mcts':
        select_policy = MCTSExploreSelectPolicy()
    elif args.select_policy == 'random':
        select_policy = RandomSelectPolicy()
    elif args.select_policy == 'mcts_normalized':
        select_policy = MCTSNormalizedRewardSelectPolicy(
            normalization_mode=args.normalization_mode,
            max_cwe_count=args.max_cwe_count,
            enable_monitoring=args.enable_monitoring
        )

    if args.ablation_config:
        ablation_configs = get_ablation_configs()
        if args.ablation_config not in ablation_configs:
            raise ValueError(f"Unknown ablation config: {args.ablation_config}. "
                           f"Available: {list(ablation_configs.keys())}")
        
        mutation_config = ablation_configs[args.ablation_config]
        logging.info(f"Using ablation config: {args.ablation_config}")
    else:
        mutation_config = MutationConfig(
            with_cwe_knowledge=args.with_cwe_knowledge,
            include_safe_example=args.include_safe_example,
            temperature=args.mutator_temperature,
            max_tokens=args.max_tokens,
            max_prompt_chars=args.max_prompt_chars,
            use_semantic_anchors=args.use_semantic_anchors,
            use_optional_context=args.use_optional_context,
            use_mutation_operators=args.use_mutation_operators,
            experiment_tag=args.experiment_tag
        )
    
    logging.info("Using PromptMutator with neutral operators")

    fuzzer = Fuzzer(
        templates=templates,
        target=target_model,
        task_file=task_file,
        result_path=result_path,
        select_policy=select_policy,
        energy=args.energy,
        max_jailbreak=args.max_jailbreak,
        max_query=args.max_query,
        max_iteration=args.max_iteration,
        dashscope_api_key=args.dashscope_api_key,
        kb_path=args.kb_path, 
        prompt_mutator_model=prompt_mutator_model,
        mutation_config=mutation_config,
        enable_monitoring=args.enable_monitoring,
        run_dir=run_dir,  
        monitoring_dir=monitoring_dir  
    )

    await fuzzer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SecAware Fuzzing with Oracle Evaluation and PromptMutator')
    
    # ========== API Keys ==========
    parser.add_argument('--api_key', type=str, default='sk-', 
                       help='Mutator model API Key')
    parser.add_argument('--dashscope_api_key', type=str, default='sk-',
                       help='DashScope API Key for Oracle LLM verification')
    parser.add_argument('--target_api_key', type=str, default=None,
                       help='Target model API Key (if using cloud API)')
    parser.add_argument('--prompt_mutator_api_key', type=str, default=None,
                       help='PromptMutator model API Key (if different from main API key)')
    
    # ========== Mutator Model ==========
    parser.add_argument('--model_path', type=str, default='qwen-plus',
                       help='Mutator model path or name')
    parser.add_argument('--mutator_temperature', type=float, default=0.7,
                       help='Temperature for mutation models')
    
    # ========== Target Model ==========
    parser.add_argument('--target_backend', type=str, default='vllm_server',
                       choices=['ollama', 'vllm', 'vllm_server', 'local', 'qwen'],
                       help='Target LLM backend')
    parser.add_argument('--target_model', type=str, 
                       default='Qwen2.5-Coder-7B-Instruct',
                       help='Target model path or name')
    
    #  vLLM Server 配置
    parser.add_argument('--vllm_server_url', type=str, 
                       default='http://localhost:8000/v1',
                       help='vLLM server URL (for vllm_server backend)')
    
    # ========== Fuzzing Parameters ==========
    parser.add_argument('--max_query', type=int, default=1000,
                       help='Maximum number of queries')
    parser.add_argument('--max_jailbreak', type=int, default=1000, 
                       help='Maximum jailbreak number')
    parser.add_argument('--max_iteration', type=int, default=-1,
                       help='Maximum iterations (-1 for unlimited)')
    parser.add_argument('--energy', type=int, default=1,
                       help='Energy of the fuzzing process')
    
    # ========== Data Paths ==========
    parser.add_argument("--template_path", type=str,
                       default="./datasets/coding_template.csv",
                       help="Path to prompt templates")
    parser.add_argument("--task_path", type=str,
                       default="./datasets/seed.json",
                       help="Path to seed.json file")
    
    # ========== Policies ==========
    parser.add_argument('--select_policy', type=str, default='mcts_cwe',
                       choices=['ucb', 'mcts', 'mcts_cwe', 'random', 
                               'mcts_novelty', 'mcts_normalized', 'mcts_novelty_normalized'],
                       help='Selection policy')
    
    # Selection Policy 参数
    parser.add_argument('--novelty_mode', type=str, default='log',
                       choices=['log', 'sqrt'],
                       help='Novelty weight calculation mode')
    parser.add_argument('--normalization_mode', type=str, default='sigmoid',
                       choices=['sigmoid', 'division', 'minmax'],
                       help='Reward normalization mode')
    parser.add_argument('--max_cwe_count', type=int, default=5,
                       help='Max CWE count for normalization')
    parser.add_argument('--enable_monitoring', action='store_true', default=True,
                       help='Enable fuzzing monitoring')
    
    # ========== Analysis ==========
    parser.add_argument('--enable_realistic_analysis', type=bool, default=True,
                       help='Enable realistic ratio analysis')
    parser.add_argument('--realistic_interval', type=int, default=200,
                       help='Realistic analysis interval (iterations)')
    
    # ========== Results ==========
    parser.add_argument("--result_dir", type=str, default="results/", 
                       help="Directory to save results")
    
    # ========== PromptMutator Parameters ==========
    parser.add_argument('--prompt_mutator_backend', type=str, default='qwen',
                       choices=['ollama', 'vllm', 'vllm_server', 'local', 'qwen'],
                       help='PromptMutator model backend')
    parser.add_argument('--prompt_mutator_model', type=str, default=None,
                       help='PromptMutator model path (if different from mutator_model)')
    parser.add_argument('--with_cwe_knowledge', type=bool, default=True,
                       help='Include CWE knowledge in PromptMutator')
    parser.add_argument('--include_safe_example', type=bool, default=False,
                       help='Include safe code examples in CWE knowledge')
    parser.add_argument('--max_tokens', type=int, default=4096,
                       help='Max tokens for mutation generation')
    parser.add_argument('--max_prompt_chars', type=int, default=4000,
                       help='Max characters for mutated prompts')
    
    parser.add_argument('--ablation_config', type=str, default=None,
                       choices=['full', 'no_anchors', 'no_cwe', 'no_operators', 'no_anchors_no_cwe',  
                               'anchors_only', 'cwe_only', 'operators_only', 'minimal'],
                       help='Use preset ablation configuration')
    parser.add_argument('--use_semantic_anchors', type=bool, default=True,
                       help='Include Semantic Anchors (for manual ablation)')
    parser.add_argument('--use_optional_context', type=bool, default=True,
                       help='Include CWE background context (for manual ablation)')
    parser.add_argument('--use_mutation_operators', type=bool, default=True,
                       help='Include mutation operators (for manual ablation)')
    parser.add_argument('--experiment_tag', type=str, default='full',
                       help='Experiment tag for identification')
    
    parser.add_argument('--kb_path', type=str, default=None,
                       help='Path to CWE knowledge base JSON file (default: datasets/cwe_knowledge_distilled.json)')
    
    
    add_model_args(parser)

    args = parser.parse_args()
    
    if args.ablation_config:
        print_ablation_summary()
    
    print("=" * 80)
    print("SecAware Fuzzing Configuration:")
    # ...existing prints...
    print(f"  PromptMutator Backend: {args.prompt_mutator_backend}")
    print(f"    - Model: {args.prompt_mutator_model or 'Same as mutator_model'}")
    print(f"    - CWE Knowledge: {args.with_cwe_knowledge}")
    print(f"    - Safe Examples: {args.include_safe_example}")

    if args.ablation_config:
        print(f"  Ablation Config: {args.ablation_config}")
    else:
        print(f"   Manual Ablation:")
        print(f"    - Semantic Anchors: {args.use_semantic_anchors}")
        print(f"    - Optional Context: {args.use_optional_context}")
        print(f"    - Mutation Operators: {args.use_mutation_operators}")
    print(f"  Experiment Tag: {args.experiment_tag}")
    
    print("=" * 80)
    print("SecAware Fuzzing Configuration:")
    print(f"  Target Model: {args.target_model} ({args.target_backend})")
    if args.target_backend == 'vllm_server':
        print(f"    - vLLM Server URL: {args.vllm_server_url}")
    print(f"  Mutator Model: {args.model_path}")
    print(f"  PromptMutator Backend: {args.prompt_mutator_backend}")
    print(f"    - Model: {args.prompt_mutator_model or 'Same as mutator_model'}")
    print(f"    - CWE Knowledge: {args.with_cwe_knowledge}")
    print(f"    - Safe Examples: {args.include_safe_example}")
    print(f"  Selection Policy: {args.select_policy}")
    if args.select_policy in ['mcts_novelty', 'mcts_novelty_normalized']:
        print(f"    - Novelty Mode: {args.novelty_mode}")
    if args.select_policy in ['mcts_normalized', 'mcts_novelty_normalized']:
        print(f"    - Normalization Mode: {args.normalization_mode}")
        print(f"    - Max CWE Count: {args.max_cwe_count}")
    print(f"  Monitoring: {args.enable_monitoring}")
    print(f"  Task File: {args.task_path}")
    print(f"  Max Query: {args.max_query}")
    print(f"  Max Jailbreak: {args.max_jailbreak}")
    print(f"  Max Iteration: {args.max_iteration}")
    print("=" * 80)
    
    asyncio.run(main(args))