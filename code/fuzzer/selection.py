import random
import numpy as np
import math
from typing import Dict, List, Set, Any
from collections import defaultdict


from fuzzer import Fuzzer, PromptNode
import datetime
import pandas as pd

class SelectPolicy:
    def __init__(self, fuzzer: Fuzzer):
        self.fuzzer = fuzzer

    def select(self) -> PromptNode:
        raise NotImplementedError(
            "SelectPolicy must implement select method.")

    def update(self, prompt_nodes: 'list[PromptNode]'):
        pass


class RoundRobinSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: Fuzzer = None):
        super().__init__(fuzzer)
        self.index: int = 0

    def select(self) -> PromptNode:
        seed = self.fuzzer.prompt_nodes[self.index]
        seed.visited_num += 1
        return seed

    def update(self, prompt_nodes: 'list[PromptNode]'):
        self.index = (self.index - 1 + len(self.fuzzer.prompt_nodes)
                      ) % len(self.fuzzer.prompt_nodes)


class RandomSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: Fuzzer = None):
        super().__init__(fuzzer)

    def select(self) -> PromptNode:
        seed = random.choice(self.fuzzer.prompt_nodes)
        seed.visited_num += 1
        return seed


class UCBSelectPolicy(SelectPolicy):
    def __init__(self,
                 explore_coeff: float = 1.0,
                 fuzzer: Fuzzer = None):
        super().__init__(fuzzer)

        self.step = 0
        self.last_choice_index = None
        self.explore_coeff = explore_coeff
        if self.fuzzer is not None:
            self.rewards = [0 for _ in range(len(self.fuzzer.prompt_nodes))]
        else:
            self.rewards = []


    def select(self) -> PromptNode:
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(self.fuzzer.prompt_nodes) - len(self.rewards))])

        self.step += 1
        scores = np.zeros(len(self.fuzzer.prompt_nodes))
        for i, prompt_node in enumerate(self.fuzzer.prompt_nodes):
            smooth_visited_num = prompt_node.visited_num + 1
            scores[i] = self.rewards[i] / smooth_visited_num + \
                self.explore_coeff * \
                np.sqrt(2 * np.log(self.step) / smooth_visited_num)

        self.last_choice_index = np.argmax(scores)
        self.fuzzer.prompt_nodes[self.last_choice_index].visited_num += 1
        return self.fuzzer.prompt_nodes[self.last_choice_index]

    def update(self, prompt_nodes: 'list[PromptNode]'):
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])
        self.rewards[self.last_choice_index] += \
            succ_num / len(self.fuzzer.templates)

            


class MCTSExploreSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: Fuzzer = None, ratio=0.5, alpha=0.1, beta=0.2):
        super().__init__(fuzzer)

        self.step = 0
        self.mctc_select_path: 'list[PromptNode]' = []
        self.last_choice_index = None
        self.rewards = []
        self.ratio = ratio  # balance between exploration and exploitation
        self.alpha = alpha  # penalty for level
        self.beta = beta   # minimal reward after penalty

    def select(self) -> PromptNode:
        self.step += 1
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(self.fuzzer.prompt_nodes) - len(self.rewards))])

        self.mctc_select_path.clear()
        cur = max(
            self.fuzzer.initial_prompts_nodes,
            key=lambda pn:
            self.rewards[pn.index] / (pn.visited_num + 1) +
            self.ratio * np.sqrt(2 * np.log(self.step) /
                                 (pn.visited_num + 0.01))
        )
        self.mctc_select_path.append(cur)

        while len(cur.child) > 0:
            if np.random.rand() < self.alpha:
                break
            cur = max(
                cur.child,
                key=lambda pn:
                self.rewards[pn.index] / (pn.visited_num + 1) +
                self.ratio * np.sqrt(2 * np.log(self.step) /
                                     (pn.visited_num + 0.01))
            )
            self.mctc_select_path.append(cur)

        for pn in self.mctc_select_path:
            pn.visited_num += 1

        self.last_choice_index = cur.index
        return cur

    def update(self, prompt_nodes: 'list[PromptNode]'):
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])
        # cwe_ratios = self.fuzzer.calculate_jailbreak_ratio_by_cwe()

        last_choice_node = self.fuzzer.prompt_nodes[self.last_choice_index]
        for prompt_node in reversed(self.mctc_select_path):
            reward = succ_num / (len(self.fuzzer.templates)
                                 * len(prompt_nodes))
            # cwe_ratio = cwe_ratios.get(prompt_node.cwe_id, 1)
            self.rewards[prompt_node.index] += reward * \
                max(self.beta, (1 - 0.1 * last_choice_node.level))
                # max(self.beta, (1 - 0.1 * last_choice_node.level) * cwe_ratio)


class MCTSExploreCWESelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: Fuzzer = None, ratio=0.5, alpha=0.1, beta=0.2):
        super().__init__(fuzzer)
        self.step = 0
        self.mctc_select_path = []
        self.last_choice_index = None
        self.rewards = []
        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta
        self.cwe_success_count = defaultdict(int)
        self.cwe_attempt_count = defaultdict(int)
        self.cwe_last_success = defaultdict(int)
        self.history = []  

    def select(self) -> PromptNode:
        self.step += 1
        self._expand_rewards()
        self.mctc_select_path.clear()
        cur = max(self.fuzzer.initial_prompts_nodes, key=self._calculate_ucb)
        self.mctc_select_path.append(cur)
        while cur.child and np.random.rand() >= self.alpha:
            cur = max(cur.child, key=self._calculate_ucb)
            self.mctc_select_path.append(cur)
        for pn in self.mctc_select_path:
            pn.visited_num += 1
        self.last_choice_index = cur.index
        return cur

    def update(self, prompt_nodes: 'list[PromptNode]'):
        succ_num = sum(pn.num_jailbreak for pn in prompt_nodes)
        last_choice_node = self.fuzzer.prompt_nodes[self.last_choice_index]
        self._update_cwe_stats(prompt_nodes)
        cwe_ratios = self._calculate_jailbreak_ratio_by_cwe()
        for pn in reversed(self.mctc_select_path):
            node_cwe_ids = self._get_node_cwe_ids(pn)
            avg_cwe_ratio = sum([cwe_ratios.get(cid, 1) for cid in node_cwe_ids]) if node_cwe_ids else 1
            penalty = max(self.beta, (1 - 0.1 * getattr(last_choice_node, 'level', 0)) + avg_cwe_ratio)
            reward = succ_num / (len(self.fuzzer.templates) * len(prompt_nodes)) * penalty
            self.rewards[pn.index] += reward
        self.history.append({
            'step': self.step,
            'rewards': self.rewards.copy(),
            'cwe_ratios': cwe_ratios.copy()
        })

    def _expand_rewards(self):
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend([0] * (len(self.fuzzer.prompt_nodes) - len(self.rewards)))

    def _calculate_ucb(self, node):
        return self.rewards[node.index] / (node.visited_num + 1) + \
               self.ratio * np.sqrt(2 * np.log(self.step) / (node.visited_num + 0.01))

    @staticmethod
    def _get_node_cwe_ids(node):
        if hasattr(node, 'cwe_ids') and node.cwe_ids:
            return node.cwe_ids if isinstance(node.cwe_ids, list) else [node.cwe_ids]
        elif hasattr(node, 'cwe_id') and node.cwe_id:
            return [node.cwe_id]
        return []

    def _update_cwe_stats(self, prompt_nodes):
        for pn in prompt_nodes:
            for cwe in self._get_node_cwe_ids(pn):
                self.cwe_attempt_count[cwe] += 1
                if hasattr(pn, 'all_found_cwes') and cwe in pn.all_found_cwes:
                    self.cwe_success_count[cwe] += 1
                    self.cwe_last_success[cwe] = self.step

    def _calculate_jailbreak_ratio_by_cwe(self):
        return {cwe: self.cwe_success_count[cwe] / self.cwe_attempt_count[cwe]
                if self.cwe_attempt_count[cwe] > 0 else 0
                for cwe in self.cwe_attempt_count}
    

# class MCTSPriorityNoAverageMultiCWESelectPolicy(SelectPolicy):
#     def __init__(self, fuzzer: Fuzzer = None, ratio=0.5, alpha=0.1, beta=0.2,
#                  w_cwe_global=0.3, w_coverage=0.2):
#         super().__init__(fuzzer)
#         self.step = 0
#         self.mctc_select_path: 'list[PromptNode]' = []
#         self.last_choice_index = None
#         self.rewards = []
#         self.ratio = ratio  # balance between exploration and exploitation
#         self.alpha = alpha  # penalty for level
#         self.beta = beta   # minimal reward after penalty
        
#         # 新增多CWE权重参数（主要用于UCB计算）
#         self.w_cwe_global = w_cwe_global  # 全局CWE成功率权重
#         self.w_coverage = w_coverage  # CWE覆盖奖励权重
        
#         self.cwe_success_count = defaultdict(int)  # 每个CWE成功次数
#         self.cwe_attempt_count = defaultdict(int)  # 每个CWE尝试次数
#         self.cwe_last_success = defaultdict(int)  # 每个CWE最后成功的步骤
        
#         self.reward_history = []  # 记录每次更新后的完整rewards
#         self.step_rewards = []    # 记录每步的奖励变化
#         self.reward_updates = []  # 记录每次更新的详细信息

#     def select(self) -> PromptNode:
#         self.step += 1
#         if len(self.fuzzer.prompt_nodes) > len(self.rewards):
#             self.rewards.extend(
#                 [0 for _ in range(len(self.fuzzer.prompt_nodes) - len(self.rewards))])

#         self.mctc_select_path.clear()
#         cur = max(
#             self.fuzzer.initial_prompts_nodes,
#             key=lambda pn: self._calculate_multi_cwe_ucb(pn)
#         )
#         self.mctc_select_path.append(cur)

#         while len(cur.child) > 0:
#             if np.random.rand() < self.alpha:
#                 break
#             cur = max(
#                 cur.child,
#                 key=lambda pn: self._calculate_multi_cwe_ucb(pn)
#             )
#             self.mctc_select_path.append(cur)

#         for pn in self.mctc_select_path:
#             pn.visited_num += 1

#         self.last_choice_index = cur.index
#         self.record_current_state()
#         return cur

#     def _calculate_multi_cwe_ucb(self, node):
#         base_ucb = (self.rewards[node.index] / (node.visited_num + 1) +
#                    self.ratio * np.sqrt(2 * np.log(self.step) / (node.visited_num + 0.01)))      
#         return base_ucb 
        

#     def _get_node_cwe_ids(self, node):
#         """获取节点的CWE ID列表，兼容单个和多个CWE"""
#         if hasattr(node, 'cwe_ids') and node.cwe_ids:
#             return node.cwe_ids if isinstance(node.cwe_ids, list) else [node.cwe_ids]
#         elif hasattr(node, 'cwe_id') and node.cwe_id:
#             return [node.cwe_id]
#         return []

#     def update(self, prompt_nodes: 'list[PromptNode]'):
#         """使用与MCTSPrioritySelectPolicy相同的奖励计算方式，但支持多CWE统计"""
#         succ_num = sum([prompt_node.num_jailbreak for prompt_node in prompt_nodes])
#         last_choice_node = self.fuzzer.prompt_nodes[self.last_choice_index]
        
#         # 记录更新前的状态
#         rewards_before = self.rewards.copy()
    
#         # 处理多CWE统计更新
#         for prompt_node in prompt_nodes:
#             target_cwe_ids = self._get_node_cwe_ids(prompt_node)
#             found_cwes = getattr(prompt_node, 'all_found_cwes', [])
            
#             # 记录当前节点的所有目标CWE被尝试
#             for target_cwe in target_cwe_ids:
#                 if target_cwe:
#                     self.cwe_attempt_count[target_cwe] += 1
                    
#                     # 如果成功触发该目标CWE，更新成功计数
#                     if target_cwe in found_cwes:
#                         self.cwe_success_count[target_cwe] += 1
#                         self.cwe_last_success[target_cwe] = self.step
            
#             # 针对所有触发的CWE更新统计(包括非目标CWE)
#             for cwe in found_cwes:
#                 if cwe not in target_cwe_ids:  # 避免重复计数目标CWE
#                     self.cwe_success_count[cwe] += 1
#                     # 增加一次尝试次数
#                     self.cwe_attempt_count[cwe] += 1
#                     self.cwe_last_success[cwe] = self.step
        
#         cwe_ratios = self.calculate_jailbreak_ratio_by_cwe()
        
#         for prompt_node in reversed(self.mctc_select_path):
#             # 基础奖励计算（与原版完全一致）
#             reward = succ_num / (len(self.fuzzer.templates) * len(prompt_nodes))
            
#             # 获取节点的CWE信息
#             node_cwe_ids = self._get_node_cwe_ids(prompt_node)
            
#             # 对于多CWE节点，计算平均CWE比率
#             if node_cwe_ids:
#                 cwe_ratios_for_node = [cwe_ratios.get(cwe_id, 1) for cwe_id in node_cwe_ids]
#                 # avg_cwe_ratio = sum(cwe_ratios_for_node) / len(cwe_ratios_for_node)
#                 avg_cwe_ratio = sum(cwe_ratios_for_node)
#             else:
#                 # 兼容原版：如果没有CWE信息，使用传统方式
#                 avg_cwe_ratio = cwe_ratios.get(getattr(prompt_node, 'cwe_id', None), 1)
            
#             # 计算penalty_factor（与原版一致）
#             penalty_factor = max(self.beta, (1 - 0.1 * last_choice_node.level + avg_cwe_ratio))
#             reward_increment = reward * penalty_factor
            
#             # 记录详细的更新信息
#             self.reward_updates.append({
#                 'step': self.step,
#                 'node_index': prompt_node.index,
#                 'node_cwe_ids': node_cwe_ids,  # 多CWE信息
#                 'node_cwe_id': node_cwe_ids[0] if len(node_cwe_ids) == 1 else None,  # 兼容性
#                 'reward_before': self.rewards[prompt_node.index],
#                 'reward_increment': reward_increment,
#                 'raw_reward': reward,
#                 'cwe_ratio': avg_cwe_ratio,  # 对于多CWE是平均值
#                 'individual_cwe_ratios': {cwe_id: cwe_ratios.get(cwe_id, 1) for cwe_id in node_cwe_ids},
#                 'penalty_factor': penalty_factor,
#                 'node_level': prompt_node.level if hasattr(prompt_node, 'level') else None,
#                 'calculation_details': {
#                     'success_num': succ_num,
#                     'templates_count': len(self.fuzzer.templates),
#                     'prompt_nodes_count': len(prompt_nodes),
#                     'level_penalty': 1 - 0.1 * last_choice_node.level,
#                     'final_factor': penalty_factor,
#                     'is_multi_cwe': len(node_cwe_ids) > 1
#                 }
#             })
            
#             # 更新奖励（与原版完全一致）
#             self.rewards[prompt_node.index] += reward_increment
        
#         # 记录完整的rewards历史
#         self.reward_history.append({
#             'step': self.step,
#             'rewards': self.rewards.copy(),
#             'rewards_before': rewards_before,
#             'path_nodes': [pn.index for pn in self.mctc_select_path],
#             'success_num': succ_num,
#             'cwe_ratios': cwe_ratios.copy() if cwe_ratios else {},
#             'last_choice_node_level': last_choice_node.level if hasattr(last_choice_node, 'level') else None
#         })
    
#     def calculate_jailbreak_ratio_by_cwe(self) -> dict:
#         cwe_ratios = {}
#         for cwe_id in self.cwe_attempt_count:
#             attempts = self.cwe_attempt_count[cwe_id]
#             successes = self.cwe_success_count[cwe_id]
#             cwe_ratios[cwe_id] = successes / attempts if attempts > 0 else 0
#         return cwe_ratios
    
#     def get_cwe_statistics(self) -> dict:
#         return {
#             'cwe_success_count': dict(self.cwe_success_count),
#             'cwe_attempt_count': dict(self.cwe_attempt_count),
#             'cwe_ratios': self.calculate_jailbreak_ratio_by_cwe(),
#             'cwe_last_success': dict(self.cwe_last_success),
#             'cwe_asr': {
#                 cwe_id: (self.cwe_success_count[cwe_id] / self.cwe_attempt_count[cwe_id])
#                 for cwe_id in self.cwe_attempt_count if self.cwe_attempt_count[cwe_id] > 0
#             }
#         }
    
#     def record_current_state(self):
#         if self.mctc_select_path:
#             # 收集路径中所有节点的CWE信息
#             path_cwe_info = []
#             for pn in self.mctc_select_path:
#                 node_cwe_ids = self._get_node_cwe_ids(pn)
#                 path_cwe_info.append(node_cwe_ids)
            
#             self.step_rewards.append({
#                 'step': self.step,
#                 'rewards_snapshot': self.rewards.copy(),
#                 'selected_path': [pn.index for pn in self.mctc_select_path],
#                 'last_choice': self.last_choice_index,
#                 'path_cwe_info': path_cwe_info,  # 多CWE信息
#                 'path_cwe_ids': path_cwe_info[-1] if path_cwe_info else []  # 保持兼容性
#             })
    
#     def get_reward_statistics(self):
#         if not self.reward_history:
#             return None
        
#         stats = {
#             'total_steps': len(self.reward_history),
#             'final_rewards': self.rewards.copy(),
#             'max_reward': max(self.rewards) if self.rewards else 0,
#             'min_reward': min(self.rewards) if self.rewards else 0,
#             'avg_reward': sum(self.rewards) / len(self.rewards) if self.rewards else 0,
#             'reward_distribution': {},
#             'cwe_analysis': {},
#             'multi_cwe_analysis': {}  # 新增多CWE分析
#         }
        
#         # 统计每个节点的奖励变化
#         for i, reward in enumerate(self.rewards):
#             stats['reward_distribution'][i] = reward
        
#         # 分析CWE相关的奖励分布
#         cwe_rewards = {}
#         multi_cwe_rewards = {}  # 多CWE奖励统计
        
#         for update in self.reward_updates:
#             node_cwe_ids = update.get('node_cwe_ids', [])
            
#             # 单CWE统计（保持兼容）
#             if len(node_cwe_ids) == 1:
#                 cwe_id = node_cwe_ids[0]
#                 if cwe_id not in cwe_rewards:
#                     cwe_rewards[cwe_id] = []
#                 cwe_rewards[cwe_id].append(update['reward_increment'])
            
#             # 多CWE统计
#             elif len(node_cwe_ids) > 1:
#                 cwe_combo = tuple(sorted(node_cwe_ids))
#                 if cwe_combo not in multi_cwe_rewards:
#                     multi_cwe_rewards[cwe_combo] = []
#                 multi_cwe_rewards[cwe_combo].append(update['reward_increment'])
        
#         # 处理单CWE统计
#         for cwe_id, rewards_list in cwe_rewards.items():
#             stats['cwe_analysis'][cwe_id] = {
#                 'total_updates': len(rewards_list),
#                 'total_reward': sum(rewards_list),
#                 'avg_reward': sum(rewards_list) / len(rewards_list),
#                 'max_reward': max(rewards_list),
#                 'min_reward': min(rewards_list)
#             }
        
#         # 处理多CWE统计
#         for cwe_combo, rewards_list in multi_cwe_rewards.items():
#             combo_key = '+'.join(map(str, cwe_combo))
#             stats['multi_cwe_analysis'][combo_key] = {
#                 'cwe_combination': list(cwe_combo),
#                 'total_updates': len(rewards_list),
#                 'total_reward': sum(rewards_list),
#                 'avg_reward': sum(rewards_list) / len(rewards_list),
#                 'max_reward': max(rewards_list),
#                 'min_reward': min(rewards_list)
#             }
        
#         return stats