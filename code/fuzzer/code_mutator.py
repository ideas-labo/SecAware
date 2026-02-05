import random
import json
from .code_core import Fuzzer, PromptNode
from utils.template import synthesis_message
from utils.openai_utils import openai_request
from utils.template import QUESTION_PLACEHOLDER
from llm import OpenAILLM, LLM
from utils.prompt import *


class Mutator:
    def __init__(self, fuzzer: 'Fuzzer'):
        self._fuzzer = fuzzer
        self.n = None

    def mutate_single(self, seed) -> 'list[str]':
        raise NotImplementedError("Mutator must implement mutate method.")

    def mutate_batch(self, seeds) -> 'list[list[str]]':
        return [self.mutate_single(seed) for seed in seeds]

    @property
    def fuzzer(self):
        return self._fuzzer

    @fuzzer.setter
    def fuzzer(self, Fuzzer):
        self._fuzzer = Fuzzer
        self.n = Fuzzer.energy


class OpenAIMutatorBase(Mutator):
    def __init__(self,
                 model: 'LLM',  # 修改类型提示，支持更广泛的LLM类型
                 temperature: int = 1,
                 max_tokens: int = 512,
                 max_trials: int = 100,
                 failure_sleep_time: int = 5,
                 fuzzer: 'Fuzzer' = None):
        super().__init__(fuzzer)

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_trials = max_trials
        self.failure_sleep_time = failure_sleep_time

    def mutate_single(self, seed) -> 'list[str]':
        # 确保 seed 是字符串或 PromptNode
        if hasattr(seed, 'task'):
            prompt_text = seed.task
        elif isinstance(seed, str):
            prompt_text = seed
        else:
            prompt_text = str(seed)
            
        print(f"Mutating with prompt: {prompt_text[:100]}...")
        
        try:
            # 调用模型生成
            result = self.model.generate(prompt_text, self.temperature, self.max_tokens)
            
            # 确保返回字符串列表
            if isinstance(result, str):
                return [result]
            elif isinstance(result, list):
                return [str(r) for r in result]
            else:
                return [str(result)]
                
        except Exception as e:
            print(f"Generation error in {self.__class__.__name__}: {e}")
            return [prompt_text]  # 失败时返回原始输入

    def _format_prompt(self, template, **kwargs):
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print("Formatting with keys:", list(kwargs.keys()))
            print(f"Prompt formatting error: missing parameter {e}")
            return ""

class OpenAIMutatorCrossOverCode(OpenAIMutatorBase):
    def __init__(self,
                 model: 'OpenAILLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 max_trials: int = 100,
                 failure_sleep_time: int = 5,
                 fuzzer: 'Fuzzer' = None):
        super().__init__(model, temperature, max_tokens, max_trials, failure_sleep_time, fuzzer)

    def cross_over(self, seed: PromptNode, prompt_nodes: 'list[PromptNode]'):
        # # 过滤出语言相同的节点
        # same_lang_nodes = [node for node in prompt_nodes if node.lang == seed.lang]
        
        # # 如果没有语言相同的节点，回退到使用所有节点
        # if not same_lang_nodes:
        #     same_lang_nodes = prompt_nodes
            
        # # 从过滤后的节点中随机选择
        # candidate_prompt = random.choice(same_lang_nodes)

        candidate_prompt = random.choice(prompt_nodes)
        
        return self._format_prompt(
            crossover_prompt,
            task1=seed.task,
            task2=candidate_prompt.task
        )

    def mutate_single(self, seed):
        return super().mutate_single(self.cross_over(seed, self.fuzzer.prompt_nodes))
    

# CWE指导的变异器
class OpenAIMutatorGuidedGenerate(OpenAIMutatorBase):
    def __init__(self,
                 model: 'OpenAILLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 max_trials: int = 100,
                 failure_sleep_time: int = 5,
                 fuzzer: 'Fuzzer' = None):
        super().__init__(model, temperature, max_tokens, max_trials, failure_sleep_time, fuzzer)
        
    def guided_generate(self, seed: PromptNode, _: 'list[PromptNode]'):
        cwe_entry = self.fuzzer.kb.get_entrys(seed.cwe_id)
        template = modified_guided_generate_similar_prompt
        formatted_prompt = self._format_prompt(
            template,
            task=seed.task,
            cwe_id=cwe_entry.get("cwe_id", ""),
            cwe_title=cwe_entry.get("cwe_title", ""),
            description=cwe_entry.get("description", ""),
            extended_description=cwe_entry.get("extended_description", ""),
            usage_scenarios=cwe_entry.get("usage_scenarios", ""),
            design_challenges=cwe_entry.get("design_challenges", ""),
            engineering_tradeoffs=cwe_entry.get("engineering_tradeoffs", ""),
            demonstrative_examples_unsafe_code=cwe_entry.get("demonstrative_examples_unsafe_code", "")
        )
        return formatted_prompt
        
    def mutate_single(self, seed):
        return super().mutate_single(self.guided_generate(seed, self.fuzzer.prompt_nodes))

class OpenAIMutatorGuidedMutation(OpenAIMutatorBase):
    def __init__(self,
                 model: 'OpenAILLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 max_trials: int = 100,
                 failure_sleep_time: int = 5,
                 fuzzer: 'Fuzzer' = None):
        super().__init__(model, temperature, max_tokens, max_trials, failure_sleep_time, fuzzer)
        
    def guided_mutate(self, seed: PromptNode, _: 'list[PromptNode]'):
        cwe_entry = self.fuzzer.kb.get_entrys(seed.cwe_id)
        template = modified_guided_mutation_prompt
        formatted_prompt = self._format_prompt(
            template,
            task=seed.task,
            cwe_id=cwe_entry.get("cwe_id", ""),
            cwe_title=cwe_entry.get("cwe_title", ""),
            description=cwe_entry.get("description", ""),
            extended_description=cwe_entry.get("extended_description", ""),
            expanded_use_cases=cwe_entry.get("expanded_use_cases", ""),
            design_challenges=cwe_entry.get("design_challenges", ""),
            engineering_tradeoffs=cwe_entry.get("engineering_tradeoffs", ""),
            demonstrative_examples_unsafe_code=cwe_entry.get("demonstrative_examples_unsafe_code", "")
        )
        return formatted_prompt

    def mutate_single(self, seed):
        return super().mutate_single(self.guided_mutate(seed, self.fuzzer.prompt_nodes))


# 2. 实现基于CWE知识的对抗性变异算子  
class OpenAIMutatorAdversarialMutation(OpenAIMutatorBase):
    def __init__(self,
                 model: 'OpenAILLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 max_trials: int = 100,
                 failure_sleep_time: int = 5,
                 fuzzer: 'Fuzzer' = None):
        super().__init__(model, temperature, max_tokens, max_trials, failure_sleep_time, fuzzer)
        
    def adversarial_mutate(self, seed: PromptNode, _: 'list[PromptNode]'):
        cwe_entry = self.fuzzer.kb.get_entrys(seed.cwe_id)
        template = modified_adversarial_mutation_prompt
        formatted_prompt = self._format_prompt(
            template,
            task=seed.task,
            cwe_id=cwe_entry.get("cwe_id", ""),
            cwe_title=cwe_entry.get("cwe_title", ""),
            description=cwe_entry.get("description", ""),
            extended_description=cwe_entry.get("extended_description", ""),
            design_challenges=cwe_entry.get("design_challenges", ""),
            engineering_tradeoffs=cwe_entry.get("engineering_tradeoffs", ""),
        )
        return formatted_prompt 
        
    def mutate_single(self, seed):
        return super().mutate_single(self.adversarial_mutate(seed, self.fuzzer.prompt_nodes))


# 3. 实现基于CWE知识的引导扩展算子
class OpenAIMutatorGuidedExpansion(OpenAIMutatorBase):
    def __init__(self,
                 model: 'OpenAILLM',
                 temperature: int = 1,
                 max_tokens: int = 512,
                 max_trials: int = 100,
                 failure_sleep_time: int = 5,
                 fuzzer: 'Fuzzer' = None):
        super().__init__(model, temperature, max_tokens, max_trials, failure_sleep_time, fuzzer)
        
    def guided_expand(self, seed: PromptNode, _: 'list[PromptNode]'):
        cwe_entry = self.fuzzer.kb.get_entrys(seed.cwe_id)
        template = modified_guided_expansion_prompt
        formatted_prompt = self._format_prompt(
            template,
            task=seed.task,
            cwe_id=cwe_entry.get("cwe_id", ""),
            cwe_title=cwe_entry.get("cwe_title", ""),
            description=cwe_entry.get("description", ""),
            extended_description=cwe_entry.get("extended_description", ""),
            usage_scenarios=cwe_entry.get("usage_scenarios", ""),
            design_challenges=cwe_entry.get("design_challenges", ""),
            engineering_tradeoffs=cwe_entry.get("engineering_tradeoffs", ""),
            expanded_use_cases=cwe_entry.get("expanded_use_cases", ""),
            external_dependencies=cwe_entry.get("external_dependencies", ""),
            demonstrative_examples_unsafe_code=cwe_entry.get("demonstrative_examples_unsafe_code", "")
        )

        return formatted_prompt

    def mutate_single(self, seed):
        return super().mutate_single(self.guided_expand(seed, self.fuzzer.prompt_nodes))



class MutatePolicy:
    def __init__(self,
                 mutators: 'list[Mutator]',
                 fuzzer: 'Fuzzer' = None):
        self.mutators = mutators
        self._fuzzer = fuzzer

    def mutate_single(self, seed):
        raise NotImplementedError("MutatePolicy must implement mutate method.")

    def mutate_batch(self, seeds):
        raise NotImplementedError("MutatePolicy must implement mutate method.")

    @property
    def fuzzer(self):
        return self._fuzzer

    @fuzzer.setter
    def fuzzer(self, Fuzzer):
        self._fuzzer = Fuzzer
        for mutator in self.mutators:
            mutator.fuzzer = Fuzzer


class MutateRandomSinglePolicy(MutatePolicy):
    def __init__(self,
                 mutators: 'list[Mutator]',
                 fuzzer: 'Fuzzer' = None,
                 concatentate: bool = False):
        super().__init__(mutators, fuzzer)
        self.concatentate = concatentate

    # def mutate_single(self, seed: PromptNode) -> PromptNode:
    #     mutated_prompt = self.mutator.mutate(seed.prompt)
    #     new_prompt_node = PromptNode(
    #         fuzzer=self.fuzzer,
    #         prompt=mutated_prompt,
    #         parent=seed,
    #         mutator=self.mutator,
    #         seed_prompt=seed.seed_prompt  # 传递 seed_prompt
    #     )
    #     return new_prompt_node

    def mutate_single(self, prompt_node: 'PromptNode') -> 'list[PromptNode]':
        mutator = random.choice(self.mutators)
        results = mutator.mutate_single(prompt_node)
        if self.concatentate:
            results = [result + prompt_node.task  for result in results]

        return [PromptNode(self.fuzzer, result, prompt_node.cwe_ids, prompt_node.lang, parent=prompt_node, mutator=mutator) for result in results]
    


class ProgressiveMutatePolicy(MutatePolicy):
    def __init__(self,
                 guided_mutators: 'list[Mutator]',
                 adversarial_mutators: 'list[Mutator]',
                 fuzzer: 'Fuzzer' = None):
        if not guided_mutators or not adversarial_mutators:
            raise ValueError("Both guided and adversarial mutators lists cannot be empty")
        self.guided_mutators = guided_mutators
        self.adversarial_mutators = adversarial_mutators
        all_mutators = guided_mutators + adversarial_mutators
        super().__init__(all_mutators, fuzzer)
    
    def mutate_single(self, prompt_node: 'PromptNode') -> 'list[PromptNode]':

        if not prompt_node.mutation_trace:
            mutator = random.choice(self.guided_mutators)
            if hasattr(mutator, 'guided_generate'):
                new_prompt = mutator.guided_generate(prompt_node, self.fuzzer.prompt_nodes)
            elif hasattr(mutator, 'guided_mutate'):
                new_prompt = mutator.guided_mutate(prompt_node, self.fuzzer.prompt_nodes)
            elif hasattr(mutator, 'guided_expand'):
                new_prompt = mutator.guided_expand(prompt_node, self.fuzzer.prompt_nodes)
            elif hasattr(mutator, 'cross_over'):
                new_prompt = mutator.cross_over(prompt_node, self.fuzzer.prompt_nodes)
            elif hasattr(mutator, 'expand'):
                new_prompt = mutator.expand(prompt_node, self.fuzzer.prompt_nodes)
            else:
                raise ValueError(f"Unsupported guided mutator: {mutator.__class__.__name__}")
            mutation_type = "guided"
        else:
            mutator = random.choice(self.adversarial_mutators)
            new_prompt = mutator.adversarial_mutate(prompt_node, self.fuzzer.prompt_nodes)
            mutation_type = "adversarial"
        
        results = mutator.model.generate(new_prompt, mutator.temperature, mutator.max_tokens)
        new_nodes = []
        for result in results:
            new_node = PromptNode(
                self.fuzzer,
                result,
                prompt_node.cwe_ids,
                prompt_node.lang,
                parent=prompt_node,
                mutator=mutator
            )
            new_node.mutation_trace = prompt_node.mutation_trace.copy()
            new_node.mutation_trace.append(mutator.__class__.__name__)
            new_nodes.append(new_node)
        
        return new_nodes

class ChainedMutatePolicy(MutatePolicy):
    def __init__(self,
                mutator_chain: 'list[Mutator]',
                fuzzer: 'Fuzzer' = None):
        super().__init__(mutator_chain, fuzzer)
        self.mutator_chain = mutator_chain  # 按顺序执行的变异器链
    
    def mutate_single(self, prompt_node: 'PromptNode') -> 'list[PromptNode]':
        """
        Apply a chain of mutations to a single PromptNode
        
        This implementation applies each mutator in sequence to produce only final results,
        without preserving intermediate mutations.
        """
        results = [prompt_node.task]
        last_mutator = prompt_node.mutator
        mutation_trace = prompt_node.mutation_trace.copy()  # 复制现有的 mutation_trace
        
        for mutator in self.mutator_chain:
            new_results = []
            
            for result in results:
                temp_node = PromptNode(self.fuzzer, result, prompt_node.cwe_ids,
                                      prompt_node.lang, parent=prompt_node, 
                                      mutator=last_mutator)
                
                mutated = mutator.mutate_single(temp_node)
                new_results.extend(mutated)
            
            # 更新 mutation_trace
            mutation_trace.append(mutator.__class__.__name__)
            results = new_results
            last_mutator = mutator
        
        # 为最终结果创建新的 PromptNode，并更新 mutation_trace

        new_nodes = [PromptNode(self.fuzzer, result, prompt_node.cwe_ids,
                          prompt_node.lang, parent=prompt_node, 
                          mutator=self.mutator_chain[-1]) for result in results]
        
        for node in new_nodes:
            # 复制 mutation_trace
            node.mutation_trace = mutation_trace.copy()

        return new_nodes
    
    
    def mutate_batch(self, seeds: 'list[PromptNode]') -> 'list[list[PromptNode]]':
        return [self.mutate_single(seed) for seed in seeds]


# class CWEProgressiveThreatMutatePolicy(ChainedMutatePolicy):
#     def __init__(self,
#                 guided_mutators: 'list[Mutator]',
#                 adversarial_mutators: 'list[Mutator]',
#                 fuzzer: 'Fuzzer' = None):

#         for mutator in guided_mutators:
#             if not isinstance(mutator, (OpenAIMutatorCrossOverCode,OpenAIMutatorGuidedGenerate, OpenAIMutatorGuidedMutation, 
#                                       OpenAIMutatorGuidedExpansion)):
#                 print(f"Warning: {mutator.__class__.__name__} may not be designed for guidance mutation")
                
#         for mutator in adversarial_mutators:
#             if not isinstance(mutator, (OpenAIMutatorAdversarialMutation)):
#                 print(f"Warning: {mutator.__class__.__name__} may not be designed for adversarial mutation")
        
#         mutator_chain = []
        
#         if guided_mutators:
#             mutator_chain.append(random.choice(guided_mutators))
        
#         if adversarial_mutators:
#             mutator_chain.append(random.choice(adversarial_mutators))
            
#         if not mutator_chain:
#             raise ValueError("Both guidance mutators and adversarial mutators lists cannot be empty")
            
#         super().__init__(mutator_chain, fuzzer)

class CWEProgressiveThreatMutatePolicy(MutatePolicy):
    def __init__(self,
                 guided_mutators: 'list[Mutator]',
                 adversarial_mutators: 'list[Mutator]',
                 fuzzer: 'Fuzzer' = None):
        self.guided_mutators = guided_mutators
        self.adversarial_mutators = adversarial_mutators
        all_mutators = guided_mutators + adversarial_mutators
        super().__init__(all_mutators, fuzzer) 

        for mutator in guided_mutators:
            if not isinstance(mutator, (OpenAIMutatorCrossOverCode, OpenAIMutatorGuidedGenerate,
                                        OpenAIMutatorGuidedMutation, OpenAIMutatorGuidedExpansion)):
                print(f"Warning: {mutator.__class__.__name__} may not be designed for guidance mutation")

        for mutator in adversarial_mutators:
            if not isinstance(mutator, (OpenAIMutatorAdversarialMutation,)):
                print(f"Warning: {mutator.__class__.__name__} may not be designed for adversarial mutation")

    def mutate_single(self, prompt_node: 'PromptNode') -> 'list[PromptNode]':
        mutator_chain = []

        if self.guided_mutators:
            mutator_chain.append(random.choice(self.guided_mutators))

        if self.adversarial_mutators:
            mutator_chain.append(random.choice(self.adversarial_mutators))

        if not mutator_chain:
            raise ValueError("No valid mutator in either guided or adversarial list")

        # 修复：正确处理变异链
        current_nodes = [prompt_node]  # 从 PromptNode 开始，而不是字符串
        mutation_trace = prompt_node.mutation_trace.copy()
        
        for i, mutator in enumerate(mutator_chain):
            next_nodes = []
            
            for node in current_nodes:
                try:
                    # 每个 mutator 的 mutate_single 返回 list[PromptNode]
                    mutated_nodes = mutator.mutate_single(node)
                    
                    # 确保返回的是 PromptNode 列表
                    if isinstance(mutated_nodes, list):
                        for mutated_node in mutated_nodes:
                            if hasattr(mutated_node, 'task'):  # 确认是 PromptNode
                                next_nodes.append(mutated_node)
                            else:
                                # 如果不是 PromptNode，尝试创建一个
                                new_node = PromptNode(
                                    self.fuzzer, 
                                    str(mutated_node), 
                                    node.cwe_ids,
                                    node.lang, 
                                    parent=node,
                                    mutator=mutator
                                )
                                next_nodes.append(new_node)
                    else:
                        # 处理单个返回值
                        if hasattr(mutated_nodes, 'task'):
                            next_nodes.append(mutated_nodes)
                        else:
                            new_node = PromptNode(
                                self.fuzzer, 
                                str(mutated_nodes), 
                                node.cwe_ids,
                                node.lang, 
                                parent=node,
                                mutator=mutator
                            )
                            next_nodes.append(new_node)
                            
                except Exception as e:
                    print(f"Error in mutator {mutator.__class__.__name__}: {e}")
                    # 如果变异失败，保留原节点
                    next_nodes.append(node)
            
            mutation_trace.append(mutator.__class__.__name__)
            current_nodes = next_nodes
            
            print(f"After mutator {i+1} ({mutator.__class__.__name__}): {len(current_nodes)} nodes")
            
            # 调试：打印第一个节点的任务内容
            if current_nodes:
                first_task = current_nodes[0].task
                print(f"  First node task: {first_task[:100] if isinstance(first_task, str) else str(first_task)[:100]}...")

        # 更新所有最终节点的 mutation_trace
        for node in current_nodes:
            node.mutation_trace = mutation_trace.copy()
            # 确保父节点设置正确
            if node.parent is None:
                node.parent = prompt_node

        return current_nodes

    def mutate_batch(self, seeds: 'list[PromptNode]') -> 'list[list[PromptNode]]':
        return [self.mutate_single(seed) for seed in seeds]