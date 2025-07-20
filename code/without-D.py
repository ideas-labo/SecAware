import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # for debugging

from fastchat.model import add_model_args
import argparse
import pandas as pd
# from fuzzer.selection import (MCTSPriorityNoAverageMultiCWESelectPolicy, MCTSPrioritySelectPolicy, MCTSExploreSelectPolicy, UCBSelectPolicy)
from fuzzer.selection import MCTSExploreCWESelectPolicy
from fuzzer.code_mutator import (MutateRandomSinglePolicy, ProgressiveMutatePolicy, CWEProgressiveThreatMutatePolicy,OpenAIMutatorCrossOverCode,OpenAIMutatorGuidedGenerate, OpenAIMutatorGuidedMutation, 
                                      OpenAIMutatorGuidedExpansion, OpenAIMutatorAdversarialMutation)
from fuzzer import Fuzzer
from llm import QwenLLM, LocalOllamaLLM
from utils.VulAnalyser import VulAnalyser
import random
random.seed(100)
import logging
httpx_logger: logging.Logger = logging.getLogger("httpx")
import asyncio
# disable httpx logging
httpx_logger.setLevel(logging.WARNING)
import datetime

async def main(args):
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create directories for results and logs based on the target model
    model_dir = os.path.join(args.result_dir, args.target_model.replace(':', '-'))
    os.makedirs(model_dir, exist_ok=True)

    # Update paths for results and logs
    result_path = os.path.join(model_dir, f'results_{run_timestamp}.csv')

    reward_history_path = os.path.join(model_dir, f'reward_history_{run_timestamp}.json')

    # Log paths for debugging
    print(f"Results will be saved to: {result_path}")

    templates = pd.read_csv(args.template_path)['text'].tolist()
    task_file = args.task_path
    # code_base = args.code_base


    mutator_model = QwenLLM(args.model_path, args.api_key,temperature=0.7)
    target_model = LocalOllamaLLM(args.target_model)

    mcts_policy = MCTSExploreCWESelectPolicy()
    
    fuzzer = Fuzzer(
        templates =templates,
        target=target_model,
        predictor=VulAnalyser(),
        task_file=task_file,
        result_path=result_path,
        mutate_policy=MutateRandomSinglePolicy([
            OpenAIMutatorCrossOverCode(mutator_model, temperature=0.7),  
            OpenAIMutatorGuidedGenerate(mutator_model, temperature=0.7),
            OpenAIMutatorGuidedMutation(mutator_model, temperature=0.7),
            OpenAIMutatorGuidedExpansion(mutator_model, temperature=0.7),
            OpenAIMutatorAdversarialMutation(mutator_model, temperature=0.7)]
        ),
        select_policy= mcts_policy,
        energy=args.energy,
        max_jailbreak=args.max_jailbreak,
        max_query=args.max_query,
        # generate_in_batch=False,
    )

    await fuzzer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--api_key', type=str,  help='API Key')
    parser.add_argument('--model_path', type=str, default='qwen2.5-72b-instruct',
                        help='mutate model path')
    parser.add_argument('--target_model', type=str, default='codellama:7b',
                        help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--max_query', type=int, default=1000,
                        help='The maximum number of queries')
    parser.add_argument('--max_jailbreak', type=int,
                        default=1000, help='The maximum jailbreak number')
    parser.add_argument('--energy', type=int, default=1,
                        help='The energy of the fuzzing process')
    # parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--template_path", type=str,
                        default="datasets/coding_template.csv")
    parser.add_argument("--task_path", type=str,
                        default="datasets/instruct_subset.json")
    parser.add_argument('--select_policy', type=str, default='mcts_cwe',
                        choices=['ucb', 'mcts', 'mcts_cwe', 'random'],
                        help='Selection policy: ucb, mcts, mcts_cwe, or random')
    parser.add_argument("--result_dir", type=str,
                        default="results/", help="Directory to save results and logs")
    add_model_args(parser)

    args = parser.parse_args()
    asyncio.run(main(args))