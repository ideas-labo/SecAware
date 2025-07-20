import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 用于调试

from fastchat.model import add_model_args
import argparse
import pandas as pd
from fuzzer.vanilla_mutator import *
from fuzzer.code_core import PromptNode, Fuzzer
from fuzzer.selection import MCTSExploreCWESelectPolicy, MCTSExploreSelectPolicy, UCBSelectPolicy
from llm import QwenLLM, LocalOllamaLLM
from utils.VulAnalyser import VulAnalyser
import random
import logging
import asyncio
import datetime

httpx_logger: logging.Logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

random.seed(100)

def create_vanilla_mutators(model):
    return {
        'crossover': VanillaMutatorCrossOverCode(model=model),
        'guided_generate': VanillaMutatorGuidedGenerate(model=model),
        'guided_mutation': VanillaMutatorGuidedMutation(model=model),
        'guided_expansion': VanillaMutatorGuidedExpansion(model=model),
        'adversarial': VanillaMutatorAdversarialMutation(model=model)
    }

def get_select_policy(policy_name):
    if policy_name == 'MCTSPriorityNoAverageMultiCWESelectPolicy':
        return MCTSExploreCWESelectPolicy()
    elif policy_name == 'MCTSExploreSelectPolicy':
        return MCTSExploreSelectPolicy()
    elif policy_name == 'UCBSelectPolicy':
        return UCBSelectPolicy()
    else:
        print(f"Unknown Policy:'{policy_name}', use default MCTSPriorityNoAverageMultiCWESelectPolicy")
        return MCTSExploreCWESelectPolicy()

async def main(args):
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    model_dir = os.path.join(args.result_dir, args.target_model.replace(':', '-'))
    os.makedirs(model_dir, exist_ok=True)

    result_path = os.path.join(model_dir, f'vanilla_results_{run_timestamp}.csv')

    print(f"Results will be saved to: {result_path}")

    templates = pd.read_csv(args.template_path)['text'].tolist()
    task_file = args.task_path

    mutator_model = QwenLLM(args.model_path, args.api_key)
    target_model = LocalOllamaLLM(args.target_model)

    vanilla_mutators = create_vanilla_mutators(mutator_model)
    
    simple_mutators = [
        vanilla_mutators['crossover'],
        vanilla_mutators['guided_generate'],
        vanilla_mutators['guided_mutation'],
        vanilla_mutators['guided_expansion'],
    ]
    
    complex_mutators = [
        vanilla_mutators['adversarial'],
    ]
    
    mutate_policy = VanillaProgressiveMutatePolicy(
        simple_mutators=simple_mutators,
        complex_mutators=complex_mutators
    )

    select_policy = get_select_policy(args.select_policy)
    
    fuzzer = Fuzzer(
        templates=templates,
        target=target_model,
        predictor=VulAnalyser(),
        task_file=task_file,
        result_path=result_path,
        mutate_policy=mutate_policy, 
        select_policy=select_policy,  
        energy=args.energy,
        max_jailbreak=args.max_jailbreak,
        max_query=args.max_query,
    )


    await fuzzer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vanilla Mutator Fuzzing')
    parser.add_argument('--api_key', type=str, help='API Key')
    parser.add_argument('--model_path', type=str, default='qwen2.5-72b-instruct',
                        help='mutator model path')
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