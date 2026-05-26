import logging
import time
import csv
import json

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .selection import SelectPolicy

from llm import LLM, LocalLLM
from utils.template import synthesis_message
from utils.knowledge import KnowledgeBase
from utils.oracle import CodeEvaluator 
from utils.fuzzing_monitor import FuzzingMonitor
from .new_mutators import PromptMutator, build_neutral_operator_registry, MutationConfig

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings
import asyncio
import pandas as pd
import numpy as np

class PromptNode:
    def __init__(self,
                 fuzzer: 'Fuzzer',
                 task: str,
                 cwe_ids: 'list[str]',  
                 lang: str,
                 response: str = None,
                 results: 'list[int]' = None,
                 found_cwes: 'list[list[str]]' = None, 
                 parent: 'PromptNode' = None,
                 mutator: 'PromptMutator' = None):
        self.fuzzer: 'Fuzzer' = fuzzer
        self.task: str = task
        self.cwe_ids: 'list[str]' = cwe_ids if isinstance(cwe_ids, list) else [cwe_ids]
        self.lang: str = lang
        self.response: str = response
        self.results: 'list[int]' = results if results is not None else []
        self.found_cwes: 'list[list[str]]' = found_cwes if found_cwes is not None else []
        
        self.visited_num = 0
        self.parent: 'PromptNode' = parent
        self.mutator: 'PromptMutator' = mutator
        self.child: 'list[PromptNode]' = []
        self.level: int = 0 if parent is None else parent.level + 1

        self.value = 0.0  
        self.is_terminal = False 

        self._index = None  
        
        if parent is None:
            self.ancestor_seed_idx = None
        else:
            self.ancestor_seed_idx = parent.ancestor_seed_idx

        if self.parent is not None:
            self.index = len(self.parent.child)

        self.mutation_trace: list[str] = []
        if parent is not None and hasattr(parent, 'mutation_trace'):
            self.mutation_trace.extend(parent.mutation_trace)
        
        self.functional_results: 'list[bool]' = []
        self.security_results: 'list[bool]' = []
        self.oracle_details: 'list[dict]' = []
        
        self.mutation_operators: list[str] = []
    
    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self.parent is None:
            self.ancestor_seed_idx = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_jailbreak(self):
        return sum(self.results)
    
    @property
    def num_functional_pass(self):
        return sum(self.functional_results) if self.functional_results else 0
    
    @property
    def num_secure(self):
        return sum(self.security_results) if self.security_results else 0
    
    @property
    def num_both_pass(self):
        if not self.functional_results or not self.security_results:
            return 0
        return sum(1 for f, s in zip(self.functional_results, self.security_results) if f and s)
    
    @property
    def num_functional_but_insecure(self):
        if not self.functional_results or not self.security_results:
            return 0
        return sum(1 for f, s in zip(self.functional_results, self.security_results) if f and not s)
    
    @property
    def num_nonfunctional_but_secure(self):
        if not self.functional_results or not self.security_results:
            return 0
        return sum(1 for f, s in zip(self.functional_results, self.security_results) if not f and s)
    
    @property
    def num_both_fail(self):
        if not self.functional_results or not self.security_results:
            return 0
        return sum(1 for f, s in zip(self.functional_results, self.security_results) if not f and not s)

    @property
    def num_reject(self):
        return len(self.results) - sum(self.results)

    @property
    def num_query(self):
        return len(self.results)
    
    @property
    def all_found_cwes(self) -> 'list[str]':
        unique_cwes = set()
        for cwe_list in self.found_cwes:
            unique_cwes.update(cwe_list)
        return list(unique_cwes)
    
    @property
    def primary_cwe_id(self):
        return self.cwe_ids[0] if self.cwe_ids else None
    
    @property
    def cwe_id(self):
        return self.primary_cwe_id
    
    def update_value(self, reward):
        self.value += reward
    
    def get_detailed_status(self) -> str:
        total = len(self.functional_results) if self.functional_results else 0
        if total == 0:
            return "No evaluations"
        
        return (f"Total:{total} | "
                f"Both:{self.num_both_pass} | "
                f"Func&Vuln:{self.num_functional_but_insecure} | "
                f"Secure&NonFunc:{self.num_nonfunctional_but_secure} | "
                f"BothFail:{self.num_both_fail}")

class Fuzzer:
    _kb_instance = None
    
    def __init__(self,
             templates: 'list[str]',
             target: 'LLM',
             task_file: str,
             result_path: str,
             select_policy: 'SelectPolicy',
             max_query: int = -1,
             max_jailbreak: int = -1,
             max_reject: int = -1,
             max_iteration: int = -1,
             energy: int = 1,
             record_file: str = None,
             generate_in_batch: bool = False,
             dashscope_api_key: str = "sk-",
             enable_monitoring: bool = True,
             prompt_mutator_model: LLM = None,
             mutation_config: MutationConfig = None,
             kb_path: str = None,
             run_dir: str = None,  
             monitoring_dir: str = None  
             ):

        self.templates: 'list[str]' = templates
        self.target: LLM = target
        
        self.oracle = CodeEvaluator(
            timeout=60,
            dashscope_api_key=dashscope_api_key,
            use_llm_verification=True
        )
        logging.info("Oracle evaluator initialized (functional + security)")

        if Fuzzer._kb_instance is None:
            if kb_path is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                kb_path = os.path.join(current_dir, "../datasets/cwe_knowledge_distilled.json")
            
            if not os.path.exists(kb_path):
                logging.warning(f"Knowledge base not found: {kb_path}")
                Fuzzer._kb_instance = None
            else:
                from utils.knowledge import KnowledgeBase
                Fuzzer._kb_instance = KnowledgeBase(filename=kb_path)
                logging.info(f"Loaded knowledge base from: {kb_path}")
        
        self.kb = Fuzzer._kb_instance

        if prompt_mutator_model is None:
            logging.warning("No PromptMutator model provided, using target model")
            prompt_mutator_model = target
        
        registry = build_neutral_operator_registry()
        self.prompt_mutator = PromptMutator(
            model=prompt_mutator_model,
            registry=registry,
            kb=self.kb,
            rng_seed=int(time.time())
        )
        self.mutation_config = mutation_config or MutationConfig()
        logging.info("PromptMutator initialized with neutral operators")

        print(f"Loading tasks from: {task_file}")
        try:
            self.prompt_nodes: 'list[PromptNode]' = self.initialize_prompt_nodes_from_json(task_file)
        except Exception as e:
            logging.error(f"Failed to load prompt nodes: {e}")
            raise

        if not self.prompt_nodes:
            raise ValueError(f"No prompt nodes loaded from {task_file}")

        self.initial_prompts_nodes = self.prompt_nodes.copy()

        for i, prompt_node in enumerate(self.prompt_nodes):
            prompt_node.index = i

        print(f"Successfully loaded {len(self.prompt_nodes)} prompt nodes")
        
        for i, node in enumerate(self.prompt_nodes[:min(3, len(self.prompt_nodes))]):
            print(f"Node {i}:")
            print(f"  Task: {node.task[:150]}...")
            print(f"  CWE: {node.cwe_ids}")
            print(f"  Language: {node.lang}")

        self.select_policy = select_policy

        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.current_iteration: int = 0
        self.current_functional_pass: int = 0
        self.current_secure: int = 0
        self.current_both_pass: int = 0
        self.current_functional_but_insecure: int = 0  
        self.current_nonfunctional_but_secure: int = 0
        self.current_both_fail: int = 0

        self.max_query: int = max_query
        self.max_jailbreak: int = max_jailbreak
        self.max_reject: int = max_reject
        self.max_iteration: int = max_iteration
        self.energy: int = energy
        
        self.result_file = result_path

        if run_dir is None:
            self.run_dir = os.path.dirname(self.result_file)
        else:
            self.run_dir = run_dir
        
        if monitoring_dir is None:
            self.monitoring_dir = os.path.join(self.run_dir, 'monitoring')
            os.makedirs(self.monitoring_dir, exist_ok=True)
        else:
            self.monitoring_dir = monitoring_dir
        
        logging.info(f"Run directory: {os.path.abspath(self.run_dir)}")
        logging.info(f"Monitoring directory: {os.path.abspath(self.monitoring_dir)}")
        
        result_dir = os.path.dirname(self.result_file)
        if result_dir:
            try:
                os.makedirs(result_dir, exist_ok=True)
                test_file = os.path.join(result_dir, '.test_write_permissions')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except (PermissionError, OSError) as e:
                logging.warning(f"无法写入指定目录 {result_dir}: {e}")
                timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                target_name = self.target.__class__.__name__.lower()
                fallback_filename = f'results_{target_name}_{timestamp}.csv'
                self.result_file = fallback_filename
                logging.info(f"结果将保存到当前目录: {self.result_file}")

        if record_file is None:
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            target_llm = self.target.__class__.__name__
            self.record_file = f'records-{target_llm}-max_query_{self.max_query}-{timestamp}.csv'
        else:
            self.record_file = record_file

        try:
            self.raw_fp = open(self.result_file, 'w', buffering=1)
            logging.info(f"结果文件已创建: {os.path.abspath(self.result_file)}")
        except PermissionError as e:
            logging.error(f"无法创建结果文件 {self.result_file}: {e}")
            import tempfile
            temp_dir = tempfile.gettempdir()
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            temp_filename = f'secaware_results_{timestamp}.csv'
            self.result_file = os.path.join(temp_dir, temp_filename)
            self.raw_fp = open(self.result_file, 'w', buffering=1)
            logging.warning(f"使用临时文件: {self.result_file}")

        self.writter = csv.writer(self.raw_fp)
        self.writter.writerow([
            'index', 'prompt', 'cwe_ids', 'response', 'parent_index', 
            'results', 'found_cwes',
            'functional_results', 'security_results', 
            'both_pass_count', 'functional_but_insecure_count',
            'nonfunctional_but_secure_count', 'both_fail_count',
            'level', 'is_root', 
            'mutator_type', 'mutation_trace', 'mutation_operators', 
            'timestamp', 'detailed_status'
        ])

        self.generate_in_batch = False
        if generate_in_batch is True:
            self.generate_in_batch = True
            if isinstance(self.target, LocalLLM):
                warnings.warn("IMPORTANT! Hugging face inference with batch generation has the problem of consistency due to pad tokens.")

        self.high_value_log_file = os.path.join(
            self.monitoring_dir, 
            f"high_value_vulns.jsonl"
        )
        self.high_value_fp = open(self.high_value_log_file, 'w', encoding='utf-8')
        logging.info(f"High-value vulnerabilities log: {self.high_value_log_file}")
        
        self.enable_monitoring = enable_monitoring
        self.monitor = None
        if self.enable_monitoring:
            self.monitor = FuzzingMonitor(top_k=10, bottom_k=5)
            logging.info("Fuzzing monitor initialized (Top-10 & Bottom-5)")

            self.monitor.output_dir = self.monitoring_dir
            
            if hasattr(self.select_policy, 'monitor'):
                self.select_policy.monitor = self.monitor
        
        self.setup()

    def initialize_prompt_nodes_from_json(self, json_file: str) -> List[PromptNode]:
        """从 seed.json 格式的文件中加载 prompt nodes"""
        prompt_nodes = []
        
        try:
            if not os.path.isfile(json_file):
                raise ValueError(f"Expected file path, got: {json_file}")
            
            print(f"Loading tasks from seed dataset: {json_file}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError(f"Expected list in {json_file}, got {type(data)}")
            
            loaded_count = 0
            skipped_count = 0
            
            for idx, item in enumerate(data):
                try:
                    if not isinstance(item, dict):
                        print(f"Warning: Item {idx} is not a dictionary, skipping...")
                        skipped_count += 1
                        continue
                    
                    task = item.get('test_case_prompt', '').strip()
                    
                    if not task:
                        print(f"Skipping item {idx}: no 'test_case_prompt' field or empty")
                        skipped_count += 1
                        continue
                    
                    task = ' '.join(task.split())
                    
                    if len(task) < 10:
                        print(f"Skipping item {idx}: task too short ({len(task)} chars)")
                        skipped_count += 1
                        continue
                    
                    cwe_ids = []
                    cwe_identifier = item.get('cwe_identifier', '')
                    if cwe_identifier and isinstance(cwe_identifier, str):
                        if not cwe_identifier.startswith('CWE-'):
                            cwe_identifier = f'CWE-{cwe_identifier}'
                        cwe_ids.append(cwe_identifier)
                    
                    language = item.get('language', 'python').lower()
                    
                    node = PromptNode(
                        fuzzer=self,
                        task=task,
                        cwe_ids=cwe_ids,
                        lang=language,
                        response=None,
                        results=None,
                        found_cwes=None,
                        parent=None,
                        mutator=None
                    )
                    
                    node.problem_id = item.get('idx', idx)
                    node.source_file = json_file
                    node.repo = item.get('repo', '')
                    node.pattern_desc = item.get('pattern_desc', '')
                    node.origin_code = item.get('origin_code', '')
                    node.global_context = item.get('global_context', '')
                    node.behavioral_description = item.get('behavioral_description', '')
                    node.source = item.get('source', '')
                    node.stars = item.get('stars', 0)
                    
                    node.eval_template = item.get('eval_template', '')
                    node.fc_tests = item.get('fc_tests', [])
                    
                    node.index = len(prompt_nodes)
                    prompt_nodes.append(node)
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"Warning: Error processing item {idx}: {e}")
                    skipped_count += 1
                    continue
            
            print(f"\nData loading summary:")
            print(f"  - Total items: {len(data)}")
            print(f"  - Successfully loaded: {loaded_count}")
            print(f"  - Skipped: {skipped_count}")
            
            if loaded_count == 0:
                raise ValueError(f"No valid nodes were loaded from {json_file}")
            
            logging.info(f"Successfully loaded {loaded_count}/{len(data)} PromptNodes from {json_file}")
        
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {json_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {json_file}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to initialize prompt nodes from {json_file}: {str(e)}")
        
        return prompt_nodes
    
    def setup(self):
        self.select_policy.fuzzer = self

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='[%H:%M:%S]'))
        logger.addHandler(console)

    def is_stop(self):
        checks = [
            ('max_query', 'current_query'),
            ('max_jailbreak', 'current_jailbreak'),
            ('max_reject', 'current_reject'),
            ('max_iteration', 'current_iteration'),
        ]
        return any(getattr(self, max_attr) != -1 and getattr(self, curr_attr) >= getattr(self, max_attr) for max_attr, curr_attr in checks)

    def _get_all_descendants(self, node: PromptNode) -> List[PromptNode]:
        descendants = []
        for child in node.child:
            descendants.append(child)
            descendants.extend(self._get_all_descendants(child))
        return descendants
    
    def mutate_with_prompt_mutator(self, seed: PromptNode) -> PromptNode:
        try:
            if seed.ancestor_seed_idx is not None:
                ancestor = self.initial_prompts_nodes[seed.ancestor_seed_idx]
            else:
                ancestor = seed
            
            global_context = getattr(ancestor, 'global_context', '')
            behavioral_description = getattr(ancestor, 'behavioral_description', '')
            
            mutated_task = self.prompt_mutator.mutate(
                task_prompt=seed.task,
                global_context=global_context,
                behavior_description=behavioral_description,
                cwe_id=seed.primary_cwe_id,
                ops=None,  
                cfg=self.mutation_config
            )
            
            new_node = PromptNode(
                fuzzer=self,
                task=mutated_task,
                cwe_ids=seed.cwe_ids.copy(),
                lang=seed.lang,
                parent=seed,
                mutator=self.prompt_mutator
            )
            
            if hasattr(self.prompt_mutator, '_last_used_ops'):
                new_node.mutation_operators = self.prompt_mutator._last_used_ops
            
            new_node.mutation_trace.append('PromptMutator')
            
            return new_node
            
        except Exception as e:
            logging.error(f"PromptMutator mutation failed: {e}")
            return None
    
    async def run(self):
        logging.info("Fuzzing started!")
        logging.info(f"Monitoring enabled: {self.enable_monitoring}")
        
        try:
            while not self.is_stop():
                seed = self.select_policy.select()
                
                mutated_node = self.mutate_with_prompt_mutator(seed)
                mutated_results = [mutated_node] if mutated_node else []
                
                if mutated_results:
                    await self.evaluate(mutated_results)
                    self.update(mutated_results)

                if self.enable_monitoring and self.monitor:
                    self.monitor.log_step(
                        step=self.current_iteration,
                        select_policy=self.select_policy,
                        prompt_nodes=self.prompt_nodes
                    )
                
                self.log()
                
                if (self.enable_monitoring and 
                    self.monitor and
                    self.current_iteration > 0 and 
                    self.current_iteration % 50 == 0):
                    
                    recent_events = [
                        e for e in self.monitor.key_events 
                        if e['step'] > self.current_iteration - 50
                    ]
                    
                    if recent_events:
                        logging.info(f"\n 最近50步的关键事件 ({len(recent_events)}个):")
                        for evt in recent_events[-5:]:
                            logging.info(
                                f"  Step {evt['step']}: {evt['type']} - "
                                f"Node {evt['node_idx']}, Reward={evt['reward']:.2f}"
                            )
                    
                    if self.monitor.topk_history and self.monitor.bottomk_history:
                        latest_topk = self.monitor.topk_history[-1]
                        latest_bottomk = self.monitor.bottomk_history[-1]
                        
                        logging.info(
                            f"\n📊 Top-{self.monitor.top_k} 平均 Reward: {latest_topk['avg_reward']:.2f} | "
                            f"Bottom-{self.monitor.bottom_k} 平均 Reward: {latest_bottomk['avg_reward']:.2f} | "
                            f"差距: {latest_topk['avg_reward'] - latest_bottomk['avg_reward']:.2f}"
                        )

        except KeyboardInterrupt:
            logging.info("Fuzzing interrupted by user!")
        
        if self.enable_monitoring and self.monitor:
            final_report = self.monitor.generate_report()
            logging.info(f"\n{'='*80}\n{final_report}\n{'='*80}")
            
            json_file = os.path.join(self.monitoring_dir, "final_monitoring.json")
            topk_file = os.path.join(self.monitoring_dir, "topk_nodes.csv")
            bottomk_file = os.path.join(self.monitoring_dir, "bottomk_nodes.csv")
            
            self.monitor.save_to_json(json_file)
            self.monitor.export_to_csv(topk_file, bottomk_file)
            
            try:
                from utils.fuzzing_visualizer import FuzzingVisualizer
                
                logging.info(f"\n{'='*80}")
                logging.info("🎨 Generating visualization plots...")
                logging.info(f"{'='*80}")
                
                visualizer = FuzzingVisualizer(json_file=json_file)
                viz_dir = os.path.join(self.monitoring_dir, "visualization")
                visualizer.generate_all_plots(output_dir=viz_dir)
                
                logging.info(f"\n✓ Visualization completed!")
                logging.info(f"   - Plots saved to: {os.path.abspath(viz_dir)}")
            except Exception as e:
                logging.warning(f"Visualization failed: {e}")
        
        self.raw_fp.close()
        self.high_value_fp.close()

    async def evaluate(self, prompt_nodes: 'list[PromptNode]'):
        for prompt_node in prompt_nodes:
            responses = []
            messages = []
            
            try:
                for template in self.templates:
                    message = synthesis_message(template, prompt_node.task)
                    if message is None:
                        prompt_node.response = []
                        prompt_node.results = []
                        prompt_node.functional_results = []
                        prompt_node.security_results = []
                        prompt_node.oracle_details = []
                        break
                        
                    if not self.generate_in_batch:
                        response = self.target.generate(message)
                        if isinstance(response, list):
                            actual_response = response[0] if len(response) > 0 else ""
                        else:
                            actual_response = response
                        
                        responses.append(actual_response)
                    else:
                        messages.append(message)
                        
                if self.generate_in_batch and messages:
                    batch_responses = self.target.generate_batch(messages)
                    responses = batch_responses
                        
                prompt_node.response = responses
                
                for i, response in enumerate(responses):
                    if not isinstance(response, str):
                        logging.warning(f"Response {i} is not a string: {type(response)}, converting...")
                        responses[i] = str(response)

                if responses:
                    if prompt_node.ancestor_seed_idx is not None:
                        ancestor = self.initial_prompts_nodes[prompt_node.ancestor_seed_idx]
                    else:
                        ancestor = prompt_node
                    
                    eval_template = getattr(ancestor, 'eval_template', '')
                    fc_tests = getattr(ancestor, 'fc_tests', [])
                    
                    if eval_template and '{task}' in eval_template:
                        evaluation_prompt = eval_template.replace('{task}', prompt_node.task)
                    else:
                        evaluation_prompt = prompt_node.task
                    
                    test_cases = {
                        'fc_tests': fc_tests,
                        'evaluation_prompt': evaluation_prompt
                    }
                    
                    functional_results = []
                    security_results = []
                    oracle_details_list = []
                    found_cwes = []
                    
                    for response in responses:
                        try:
                            is_functional, is_secure, details = await self.oracle.evaluate(
                                code=response,
                                test_cases=test_cases,
                                original_prompt=evaluation_prompt,
                                language=prompt_node.lang
                            )
                            
                            functional_results.append(is_functional)
                            security_results.append(is_secure)
                            oracle_details_list.append(details)
                            
                            sec_info = details.get('security', {})
                            response_cwes = sec_info.get('found_cwes', [])
                            found_cwes.append(response_cwes)
                            
                        except Exception as e:
                            logging.error(f"Oracle evaluation failed for response: {e}")
                            functional_results.append(False)
                            security_results.append(True)
                            oracle_details_list.append({'error': str(e)})
                            found_cwes.append([])
                    
                    prompt_node.functional_results = functional_results
                    prompt_node.security_results = security_results
                    prompt_node.oracle_details = oracle_details_list
                    prompt_node.found_cwes = found_cwes
                    prompt_node.results = [0 if sec else 1 for sec in security_results]
                    
                    for cwe_list in found_cwes:
                        for cwe_id in cwe_list:
                            if cwe_id not in prompt_node.cwe_ids:
                                prompt_node.cwe_ids.append(cwe_id)
                else:
                    prompt_node.results = []
                    prompt_node.functional_results = []
                    prompt_node.security_results = []
                    prompt_node.found_cwes = []
                    prompt_node.oracle_details = []
                    
            except Exception as e:
                logging.error(f"Error in evaluate for prompt_node {prompt_node.index}: {e}")
                prompt_node.response = []
                prompt_node.results = []
                prompt_node.functional_results = []
                prompt_node.security_results = []
                prompt_node.found_cwes = []
                prompt_node.oracle_details = []

    def update(self, prompt_nodes: 'list[PromptNode]'):
        self.current_iteration += 1

        for prompt_node in prompt_nodes:
            num_jailbreak = prompt_node.num_jailbreak
            num_query = prompt_node.num_query
            num_reject = prompt_node.num_reject
            num_functional_pass = prompt_node.num_functional_pass
            num_secure = prompt_node.num_secure
            num_both_pass = prompt_node.num_both_pass
            num_functional_but_insecure = prompt_node.num_functional_but_insecure  
            num_nonfunctional_but_secure = prompt_node.num_nonfunctional_but_secure
            num_both_fail = prompt_node.num_both_fail
            
            if num_jailbreak > 0:
                prompt_node.index = len(self.prompt_nodes)
                self.prompt_nodes.append(prompt_node)

            mutation_operators = getattr(prompt_node, 'mutation_operators', [])

            self.writter.writerow([
                prompt_node.index, 
                prompt_node.task, 
                prompt_node.cwe_ids, 
                prompt_node.response, 
                prompt_node.parent.index if prompt_node.parent else None,
                prompt_node.results, 
                prompt_node.all_found_cwes,
                prompt_node.functional_results,
                prompt_node.security_results,
                num_both_pass,
                num_functional_but_insecure,  
                num_nonfunctional_but_secure,
                num_both_fail,
                prompt_node.level,
                prompt_node.parent is None,
                'PromptMutator',
                '|'.join(prompt_node.mutation_trace) if prompt_node.mutation_trace else '',
                '|'.join(mutation_operators) if mutation_operators else '',
                int(time.time()),
                prompt_node.get_detailed_status()  
            ])
            
            if num_functional_but_insecure > 0:
                for i, (func, sec, cwes, resp) in enumerate(zip(
                    prompt_node.functional_results,
                    prompt_node.security_results,
                    prompt_node.found_cwes,
                    prompt_node.response if isinstance(prompt_node.response, list) else [prompt_node.response]
                )):
                    if func and not sec:  
                        high_value_entry = {
                            'iteration': self.current_iteration,
                            'timestamp': time.time(),
                            'node_index': prompt_node.index,
                            'response_index': i,
                            'task': prompt_node.task,
                            'language': prompt_node.lang,
                            'target_cwes': prompt_node.cwe_ids,
                            'found_cwes': cwes,
                            'response': resp[:500],  
                            'level': prompt_node.level,
                            'mutator': str(prompt_node.mutator.__class__.__name__) if prompt_node.mutator else None,
                            'mutation_operators': mutation_operators
                        }
                        self.high_value_fp.write(json.dumps(high_value_entry, ensure_ascii=False) + '\n')
                        self.high_value_fp.flush()
                        
                        logging.info(f"HIGH-VALUE: Node {prompt_node.index} - "
                                   f"Functional but insecure! CWEs: {cwes}")
                
            self.current_jailbreak += num_jailbreak
            self.current_query += num_query
            self.current_reject += num_reject
            self.current_functional_pass += num_functional_pass
            self.current_secure += num_secure
            self.current_both_pass += num_both_pass
            self.current_functional_but_insecure += num_functional_but_insecure 
            self.current_nonfunctional_but_secure += num_nonfunctional_but_secure
            self.current_both_fail += num_both_fail

        self.select_policy.update(prompt_nodes)

    def log(self):
        total_nodes = len(self.prompt_nodes)
        successful_nodes = len([node for node in self.prompt_nodes if node.num_jailbreak > 0])

        total_evaluated = self.current_functional_pass + self.current_nonfunctional_but_secure + \
                         self.current_functional_but_insecure + self.current_both_fail
        
        if total_evaluated > 0:
            vuln_rate = self.current_functional_but_insecure / total_evaluated * 100
            ideal_rate = self.current_both_pass / total_evaluated * 100
        else:
            vuln_rate = 0
            ideal_rate = 0
        
        logging.info(
            f"Iter {self.current_iteration}: "
            f"FuncVuln={self.current_functional_but_insecure}({vuln_rate:.1f}%) | "
            # f"Ideal={self.current_both_pass}({ideal_rate:.1f}%) | "
            f"Total={self.current_jailbreak}vulns, "
            f"{self.current_functional_pass}funcPass, "
            f"{self.current_secure}secure | "
            f"{self.current_query}queries, "
            f"{total_nodes}nodes({successful_nodes}vuln)"
        )
        
        if self.current_iteration % 10 == 0:
            logging.info(
                f"\n{'='*80}\n"
                f"【四象限分布】\n"
                f"  ✅ 功能✓ 安全✓ (理想): {self.current_both_pass}\n"
                f"  🔥 功能✓ 安全✗ (危险): {self.current_functional_but_insecure}\n"
                f"  ⚠️  功能✗ 安全✓ (无用): {self.current_nonfunctional_but_secure}\n"
                f"  ❌ 功能✗ 安全✗ (失败): {self.current_both_fail}\n"
                f"{'='*80}"
            )