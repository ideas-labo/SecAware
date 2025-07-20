from .code_core import Fuzzer, PromptNode
from .code_mutator import OpenAIMutatorBase  # 直接使用现有的基类
from utils.template import synthesis_message
from utils.openai import openai_request
from utils.template import QUESTION_PLACEHOLDER
from llm import OpenAILLM, LLM
from utils.prompt import *
import random

class VanillaMutatorCrossOverCode(OpenAIMutatorBase):
    """无domain knowledge的CrossOver变异器"""
    
    def cross_over(self, seed: PromptNode, prompt_nodes: 'list[PromptNode]'):
        """
        对两个prompt进行crossover操作，不使用任何CWE知识
        """
        # 随机选择另一个prompt进行crossover
        other_prompts = [p for p in prompt_nodes if p != seed]
        if not other_prompts:
            return seed.task
        
        other_prompt = random.choice(other_prompts)
        
        # 使用已定义的crossover_prompt
        return self._format_prompt(
            crossover_prompt,
            task1=seed.task,
            task2=other_prompt.task
        )
    
    def mutate_single(self, seed):
        return super().mutate_single(self.cross_over(seed, self.fuzzer.prompt_nodes))

class VanillaMutatorGuidedGenerate(OpenAIMutatorBase):
    """无domain knowledge的引导生成变异器"""
    
    def guided_generate(self, seed: PromptNode, prompt_nodes: 'list[PromptNode]'):
        """
        基于vanilla模板生成相似的编程任务
        """
        # 使用已定义的vanilla_guided_generate_similar_prompt
        return self._format_prompt(
            vanilla_guided_generate_similar_prompt,
            task=seed.task
        )
    
    def mutate_single(self, seed):
        return super().mutate_single(self.guided_generate(seed, self.fuzzer.prompt_nodes))

class VanillaMutatorGuidedMutation(OpenAIMutatorBase):
    """无domain knowledge的引导变异器"""
    
    def guided_mutate(self, seed: PromptNode, prompt_nodes: 'list[PromptNode]'):
        """
        对现有任务进行变异，只关注工程复杂性
        """
        # 使用已定义的vanilla_guided_mutation_prompt
        return self._format_prompt(
            vanilla_guided_mutation_prompt,
            task=seed.task
        )
    
    def mutate_single(self, seed):
        return super().mutate_single(self.guided_mutate(seed, self.fuzzer.prompt_nodes))

class VanillaMutatorGuidedExpansion(OpenAIMutatorBase):
    """无domain knowledge的引导扩展变异器"""
    
    def guided_expand(self, seed: PromptNode, prompt_nodes: 'list[PromptNode]'):
        """
        扩展现有任务的复杂性，不使用CWE知识
        """
        # 使用已定义的vanilla_guided_expansion_prompt
        return self._format_prompt(
            vanilla_guided_expansion_prompt,
            task=seed.task
        )
    
    def mutate_single(self, seed):
        return super().mutate_single(self.guided_expand(seed, self.fuzzer.prompt_nodes))

class VanillaMutatorAdversarialMutation(OpenAIMutatorBase):
    """无domain knowledge的对抗性变异器"""
    
    def adversarial_mutate(self, seed: PromptNode, prompt_nodes: 'list[PromptNode]'):
        """
        创建对抗性变异，只关注工程挑战
        """
        # 使用已定义的vanilla_adversarial_prompt
        return self._format_prompt(
            vanilla_adversarial_prompt,
            task=seed.task
        )
    
    def mutate_single(self, seed):
        return super().mutate_single(self.adversarial_mutate(seed, self.fuzzer.prompt_nodes))


# 变异策略类，专门用于vanilla mutators
class VanillaMutateRandomSinglePolicy:
    """vanilla变异器的随机单一策略"""
    
    def __init__(self, mutators: 'list[OpenAIMutatorBase]'):
        self.mutators = mutators
        self.last_used_mutator = None  # 初始化为None
    
    def mutate(self, seed: PromptNode, prompt_nodes: 'list[PromptNode]'):
        """随机选择一个vanilla mutator进行变异"""
        if not self.mutators:
            return seed.task
        
        mutator = random.choice(self.mutators)
        self.last_used_mutator = mutator  # 记录使用的变异器
        
        try:
            # 根据mutator类型调用相应的方法
            if isinstance(mutator, VanillaMutatorCrossOverCode):
                prompt = mutator.cross_over(seed, prompt_nodes)
                # 记录使用的方法
                mutator.last_method_used = 'cross_over'
            elif isinstance(mutator, VanillaMutatorGuidedGenerate):
                prompt = mutator.guided_generate(seed, prompt_nodes)
                mutator.last_method_used = 'guided_generate'
            elif isinstance(mutator, VanillaMutatorGuidedMutation):
                prompt = mutator.guided_mutate(seed, prompt_nodes)
                mutator.last_method_used = 'guided_mutate'
            elif isinstance(mutator, VanillaMutatorGuidedExpansion):
                prompt = mutator.guided_expand(seed, prompt_nodes)
                mutator.last_method_used = 'guided_expand'
            elif isinstance(mutator, VanillaMutatorAdversarialMutation):
                prompt = mutator.adversarial_mutate(seed, prompt_nodes)
                mutator.last_method_used = 'adversarial_mutate'
            else:
                return seed.task
            
            mutator.last_prompt_sent = prompt
            
            results = mutator.model.generate(prompt, mutator.temperature, mutator.max_tokens)
            result = results[0] if results else seed.task
            
            mutator.last_response = result
            
            return result
            
        except Exception as e:
            print(f"Mutation error in {mutator.__class__.__name__}: {e}")
            return seed.task


class VanillaProgressiveMutatePolicy:
    
    def __init__(self, simple_mutators: 'list[OpenAIMutatorBase]', 
                 complex_mutators: 'list[OpenAIMutatorBase]'):
        self.simple_mutators = simple_mutators
        self.complex_mutators = complex_mutators
    
    def mutate_single(self, seed: PromptNode):
        prompt_nodes = []
        if hasattr(seed, 'fuzzer') and seed.fuzzer and hasattr(seed.fuzzer, 'prompt_nodes'):
            prompt_nodes = seed.fuzzer.prompt_nodes
        
        mutated_task = self.mutate(seed, prompt_nodes)
        
        mutated_node = PromptNode(
            fuzzer=seed.fuzzer,
            task=mutated_task,
            cwe_ids=seed.cwe_ids, 
            lang=seed.lang,
            parent=seed,
            mutator=None 
        )
        
        if hasattr(seed, 'mutation_trace'):
            mutated_node.mutation_trace = seed.mutation_trace.copy()
            mutated_node.mutation_trace.append('VanillaProgressiveMutate')
        
        return [mutated_node]
    
    def mutate(self, seed: PromptNode, prompt_nodes: 'list[PromptNode]'):

        mutator_chain = []
        
        if self.simple_mutators:
            mutator_chain.append(random.choice(self.simple_mutators))
        
        if self.complex_mutators:
            mutator_chain.append(random.choice(self.complex_mutators))
        
        if not mutator_chain:
            return seed.task
        
        current_task = seed.task
        for mutator in mutator_chain:
            temp_node = PromptNode(seed.fuzzer, current_task, seed.cwe_ids, seed.lang)
            policy = VanillaMutateRandomSinglePolicy([mutator])
            current_task = policy.mutate(temp_node, prompt_nodes)
        
        return current_task