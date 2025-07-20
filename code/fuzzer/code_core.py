import logging
import time
import csv
import json

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from code_mutator import Mutator, MutatePolicy
    from .selection import SelectPolicy

from llm import LLM, LocalLLM
from utils.template import synthesis_message
from utils.VulAnalyser import VulAnalyser
from utils.knowledge import KnowledgeBase
import warnings
import asyncio
import pandas as pd
import numpy as np
import os

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
                 mutator: 'Mutator' = None):
        self.fuzzer: 'Fuzzer' = fuzzer
        self.task: str = task
        self.cwe_ids: 'list[str]' = cwe_ids if isinstance(cwe_ids, list) else [cwe_ids]
        self.lang: str = lang
        self.response: str = response
        self.results: 'list[int]' = results if results is not None else []
        self.found_cwes: 'list[list[str]]' = found_cwes if found_cwes is not None else []
        
        self.visited_num = 0
        self.parent: 'PromptNode' = parent
        self.mutator: 'Mutator' = mutator
        self.child: 'list[PromptNode]' = []
        self.level: int = 0 if parent is None else parent.level + 1

        self.value = 0.0  
        self.is_terminal = False 

        self._index = None  

        if self.parent is not None:
            self.index = len(self.parent.child)

        self.mutation_trace: list[str] = []
        if parent is not None and hasattr(parent, 'mutation_trace'):
            self.mutation_trace.extend(parent.mutation_trace)
    
    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_jailbreak(self):
        return sum(self.results)

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



class Fuzzer:
    def __init__(self,
                 templates: 'list[str]',
                 target: 'LLM',
                 predictor: 'VulAnalyser',
                 task_file: str,
                 result_path: str,
                #  code_base: str,
                 mutate_policy: 'MutatePolicy',
                 select_policy: 'SelectPolicy',
                 max_query: int = -1,
                 max_jailbreak: int = -1,
                 max_reject: int = -1,
                 max_iteration: int = -1,
                 energy: int = 1,
                 record_file: str = None,
                 generate_in_batch: bool = False,
                 ):

        self.templates: 'list[str]' = templates
        self.target: LLM = target
        self.predictor = predictor

        base_dir = os.path.dirname(os.path.abspath(__file__))
        kb_path = os.path.join(base_dir, "../datasets/cwe_knowledge_distilled.json")
        self.kb = KnowledgeBase(filename=kb_path)

        self.prompt_nodes: 'list[PromptNode]' = self.initialize_prompt_nodes_from_json(task_file)
        self.initial_prompts_nodes = self.prompt_nodes.copy()


        for i, prompt_node in enumerate(self.prompt_nodes):
            prompt_node.index = i

        self.mutate_policy = mutate_policy
        self.select_policy = select_policy

        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.current_iteration: int = 0

        self.max_query: int = max_query
        self.max_jailbreak: int = max_jailbreak
        self.max_reject: int = max_reject
        self.max_iteration: int = max_iteration
        self.energy: int = energy
        self.result_file = result_path

        if record_file is None:
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            target_llm = self.target.__class__.__name__  # 假设 target 是一个类实例
            self.record_file = f'records-{target_llm}-max_query_{self.max_query}-{timestamp}.csv'
        else:
            self.record_file = record_file

        self.raw_fp = open(self.result_file, 'w', buffering=1)
        self.writter = csv.writer(self.raw_fp)
        self.writter.writerow(
            ['index', 'prompt','cwe_ids','response', 'parent', 'results', 'found_cwes'])

        self.generate_in_batch = False
        # if len(self.questions) > 0 and generate_in_batch is True:
        if generate_in_batch is True:
            self.generate_in_batch = True
            if isinstance(self.target, LocalLLM):
                warnings.warn("IMPORTANT! Hugging face inference with batch generation has the problem of consistency due to pad tokens. We do not suggest doing so and you may experience (1) degraded output quality due to long padding tokens, (2) inconsistent responses due to different number of padding tokens during reproduction. You should turn off generate_in_batch or use vllm batch inference.")
        self.setup()


    def initialize_prompt_nodes_from_json(self, json_file: str) -> List[PromptNode]:
        prompt_nodes = []
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                task = item.get('test_case_prompt', '')
                
                cwe_identifiers = item.get('cwe_identifier', '')
                if isinstance(cwe_identifiers, str):
                    if cwe_identifiers:
                        if not cwe_identifiers.startswith('CWE-'):
                            cwe_identifiers = f'CWE-{cwe_identifiers}'
                        cwe_ids = [cwe_identifiers]
                    else:
                        cwe_ids = []
                elif isinstance(cwe_identifiers, list):
                    cwe_ids = []
                    for cwe_id in cwe_identifiers:
                        if cwe_id:
                            if not cwe_id.startswith('CWE-'):
                                cwe_id = f'CWE-{cwe_id}'
                            cwe_ids.append(cwe_id)
                else:
                    cwe_ids = []
                    
                lang = item.get('language', '')
                
                node = PromptNode(
                    fuzzer=self,
                    task=task,
                    cwe_ids=cwe_ids, 
                    lang=lang,
                    response=None,
                    results=None,
                    found_cwes=None,
                    parent=None,
                    mutator=None
                )
                
                node.index = len(prompt_nodes)
                prompt_nodes.append(node)
                
            logging.info(f"Successfully initialized from {json_file} {len(prompt_nodes)} PromptNodes")
        
        except Exception as e:
            logging.error(f"Initialization error from json file: {str(e)}")
        
        return prompt_nodes

    def setup(self):
        self.mutate_policy.fuzzer = self
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

    async def run(self):
        logging.info("Fuzzing started!")
        record_interval = 100
        try:
            while not self.is_stop():
                seed = self.select_policy.select()
                mutated_results = self.mutate_policy.mutate_single(seed)

                await self.evaluate(mutated_results)
                self.update(mutated_results)
                self.log()

        except KeyboardInterrupt:
            logging.info("Fuzzing interrupted by user!")

        logging.info("Fuzzing finished!")
        self.raw_fp.close()


    async def evaluate(self, prompt_nodes: 'list[PromptNode]'):
        for prompt_node in prompt_nodes:
            responses = []
            messages = []
            for template in self.templates:
                message = synthesis_message(template, prompt_node.task)
                if message is None:  # The prompt is not valid
                    prompt_node.response = []
                    prompt_node.results = []
                    break
                if not self.generate_in_batch:
                    response = self.target.generate(message)
                    responses.append(response[0] if isinstance(
                        response, list) else response)
                else:
                    messages.append(message)
                if self.generate_in_batch:
                    responses = self.target.generate_batch(messages)
                prompt_node.response = responses
                print(responses)
                for response in responses:
                    if not isinstance(response, str):
                        raise TypeError(f"Expected response to be a str, but got {type(response)}")

            prediction_results = await asyncio.gather(*[self.predictor.predict(response) for response in responses])
            
            results = []
            found_cwes = []
            for pred_result in prediction_results:
                is_vulnerable, cwe_ids = pred_result 
                results.append(is_vulnerable)
                found_cwes.append(cwe_ids)

            for cwe_list in found_cwes:
                for cwe_id in cwe_list:
                    if cwe_id not in prompt_node.cwe_ids:
                        prompt_node.cwe_ids.append(cwe_id)
                
            prompt_node.results = results
            prompt_node.found_cwes = found_cwes

    def update(self, prompt_nodes: 'list[PromptNode]'):
        self.current_iteration += 1

        for prompt_node in prompt_nodes:
            if prompt_node.num_jailbreak > 0:
                prompt_node.index = len(self.prompt_nodes)
                self.prompt_nodes.append(prompt_node)

            self.writter.writerow([
                    prompt_node.index, 
                    prompt_node.task, 
                    prompt_node.cwe_ids, 
                    prompt_node.response, 
                    prompt_node.parent.index if prompt_node.parent else "None", 
                    prompt_node.results, 
                    prompt_node.all_found_cwes
                    ])
                
            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject

        self.select_policy.update(prompt_nodes)

    def log(self):
        logging.info(
            f"Iteration {self.current_iteration}: {self.current_jailbreak} jailbreaks, {self.current_reject} rejects, {self.current_query} queries")
