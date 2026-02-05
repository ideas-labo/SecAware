import random
import numpy as np
import math
import logging
from typing import Dict, List, Set, Any
from collections import defaultdict

from fuzzer import Fuzzer, PromptNode
from utils.fuzzing_monitor import FuzzingMonitor
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

    @staticmethod
    def calculate_gini_coefficient(cwe_success_count: Dict[str, int]) -> float:
        if not cwe_success_count:
            return 0.0
        
        counts = sorted(cwe_success_count.values())
        n = len(counts)
        
        if n == 0:
            return 0.0
        
        if n == 1:
            return 1.0
        
        total_count = np.sum(counts)
        
        if total_count == 0:
            return 0.0
        
        if np.all(counts == counts[0]):
            return 0.0
        
        index = np.arange(1, n + 1)
        
        try:
            gini = (2 * np.sum(index * counts)) / (n * total_count) - (n + 1) / n
            gini = np.clip(gini, 0.0, 1.0)
            return float(gini)
        except Exception as e:
            import logging
            logging.warning(f"Error calculating Gini coefficient: {e}")
            return 0.0


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
        
        denominator = len(self.fuzzer.templates) if len(self.fuzzer.templates) > 0 else 1
        self.rewards[self.last_choice_index] += succ_num / denominator


class MCTSExploreSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: Fuzzer = None, ratio=0.5, alpha=0.1, beta=0.2):
        super().__init__(fuzzer)

        self.step = 0
        self.mctc_select_path: 'list[PromptNode]' = []
        self.last_choice_index = None
        self.rewards = []
        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta

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

        last_choice_node = self.fuzzer.prompt_nodes[self.last_choice_index]
        for prompt_node in reversed(self.mctc_select_path):
            reward = succ_num / (len(self.fuzzer.templates)
                                 * len(prompt_nodes))
            self.rewards[prompt_node.index] += reward * \
                max(self.beta, (1 - 0.1 * last_choice_node.level))


class MCTSNormalizedRewardSelectPolicy(SelectPolicy):
    
    def __init__(self, fuzzer: Fuzzer = None, ratio=0.5, alpha=0.1, beta=0.2,
                 normalization_mode='linear', max_cwe_count=5,
                 cwe_prior_alpha=1.0, cwe_prior_beta=1.0,
                 enable_monitoring=True):
        super().__init__(fuzzer)
        self.step = 0
        self.mctc_select_path = []
        self.last_choice_index = None
        self.rewards = []
        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta
        
        self.normalization_mode = normalization_mode
        self.max_cwe_count = max_cwe_count
        self.cwe_prior_alpha = cwe_prior_alpha
        self.cwe_prior_beta = cwe_prior_beta
        
        self.cwe_success_count = defaultdict(int)
        self.cwe_attempt_count = defaultdict(int)
        self.cwe_last_success = defaultdict(int)
        
        self._reward_history = []
        self._reward_mean = 0.0
        self._reward_std = 1.0
        
        self.monitor = FuzzingMonitor(top_k=10, bottom_k=5) if enable_monitoring else None

    def select(self) -> PromptNode:
        if not self.fuzzer.initial_prompts_nodes:
            raise ValueError("No initial prompt nodes available for selection")
        
        self.step += 1
        self._expand_rewards()
        self.mctc_select_path.clear()
        
        cur = max(self.fuzzer.initial_prompts_nodes, key=self._calculate_ucb)
        self.mctc_select_path.append(cur)
        
        while cur.child and np.random.rand() >= self.alpha:
            try:
                cur = max(cur.child, key=self._calculate_ucb)
            except ValueError:
                break
            self.mctc_select_path.append(cur)
            
        for pn in self.mctc_select_path:
            pn.visited_num += 1
            
        self.last_choice_index = cur.index
        return cur

    def update(self, prompt_nodes: 'list[PromptNode]'):
        succ_num = sum(pn.num_jailbreak for pn in prompt_nodes)
        last_choice_node = self.fuzzer.prompt_nodes[self.last_choice_index]
        
        self._update_cwe_stats(prompt_nodes)
        cwe_ratios = self._calculate_smoothed_cwe_ratios()
        
        if succ_num == 0:
            pass  
        else:
            for pn in reversed(self.mctc_select_path):
                node_cwe_ids = self._get_node_cwe_ids(pn)
                
                if node_cwe_ids:
                    scarcity_weights = [1.0 - cwe_ratios.get(cid, 0.5) for cid in node_cwe_ids]
                    avg_scarcity = np.mean(scarcity_weights)
                else:
                    avg_scarcity = 0.5
                
                base_reward = succ_num / (len(self.fuzzer.templates) * len(prompt_nodes))
                level_discount = (1 - 0.1 * getattr(last_choice_node, 'level', 0))
                
                raw_reward = base_reward * level_discount * (1.0 + avg_scarcity)
                
                if len(self._reward_history) > 10:
                    normalized_reward = self._normalize_reward(raw_reward)
                else:
                    normalized_reward = raw_reward
                
                final_reward = max(self.beta, normalized_reward)
                
                self.rewards[pn.index] += final_reward
                self._reward_history.append(raw_reward)
                
                if len(self._reward_history) % 50 == 0:
                    recent = self._reward_history[-200:]
                    self._reward_mean = np.mean(recent)
                    self._reward_std = np.std(recent) + 1e-6
        
        if self.monitor:
            self.monitor.log_step(self.step, self, self.fuzzer.prompt_nodes)

    def _normalize_reward(self, raw_reward: float) -> float:

        if self.normalization_mode == 'linear':
            max_possible = 1.0 * (1.0 + self.max_cwe_count)
            return np.clip(raw_reward / max_possible, 0.0, 1.0)
        
        elif self.normalization_mode == 'zscore':
            if self._reward_std > 1e-6:
                z = (raw_reward - self._reward_mean) / self._reward_std
                return 1.0 / (1.0 + np.exp(-z))
            return 0.5
        
        elif self.normalization_mode == 'adaptive_sigmoid':
            return 1.0 / (1.0 + np.exp(-2 * (raw_reward - self._reward_mean)))
        
        elif self.normalization_mode == 'minmax':
            if len(self._reward_history) < 2:
                return raw_reward
            min_r = min(self._reward_history[-100:])
            max_r = max(self._reward_history[-100:])
            if max_r - min_r < 1e-6:
                return 0.5
            return (raw_reward - min_r) / (max_r - min_r)
        
        else:
            return raw_reward

    def _calculate_smoothed_cwe_ratios(self) -> Dict[str, float]:
        ratios = {}
        for cwe in self.cwe_attempt_count:
            success = self.cwe_success_count.get(cwe, 0)
            attempt = self.cwe_attempt_count[cwe]
            smoothed_ratio = (success + self.cwe_prior_alpha) / \
                           (attempt + self.cwe_prior_alpha + self.cwe_prior_beta)
            ratios[cwe] = smoothed_ratio
        return ratios

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
            node_cwe_ids = self._get_node_cwe_ids(pn)
            for cwe in node_cwe_ids:
                self.cwe_attempt_count[cwe] += 1
                if hasattr(pn, 'all_found_cwes') and cwe in pn.all_found_cwes:
                    self.cwe_success_count[cwe] += 1
                    self.cwe_last_success[cwe] = self.step
    
    def get_balance_statistics(self):
        if self.monitor:
            return self.monitor.get_summary_statistics()
        
        if not self._reward_history:
            return None
        
        avg_exploitation = np.mean([r / (pn.visited_num + 1) 
                                   for r, pn in zip(self.rewards, self.fuzzer.prompt_nodes)])
        avg_exploration = self.ratio * np.sqrt(2 * np.log(self.step) / (1 + 0.01))
        
        return {
            'avg_exploitation_term': avg_exploitation,
            'avg_exploration_term': avg_exploration,
            'exploitation_to_exploration_ratio': avg_exploitation / avg_exploration if avg_exploration > 0 else float('inf'),
            'raw_reward_stats': {
                'mean': self._reward_mean,
                'std': self._reward_std,
                'recent_mean': np.mean(self._reward_history[-50:]) if len(self._reward_history) > 0 else 0
            }
        }
    
    def _calculate_gini_coefficient(self):
        return self.calculate_gini_coefficient(self.cwe_success_count)

class RandomSearchSelectPolicy(SelectPolicy):
    """
    Random Search Baseline Policy
    """
    
    def __init__(self, fuzzer: Fuzzer = None, enable_monitoring: bool = True):

        super().__init__(fuzzer)
        self.step = 0
        
        self.cwe_success_count = defaultdict(int)
        self.cwe_attempt_count = defaultdict(int)
        self.cwe_last_success = defaultdict(int)
        
        self.monitor = FuzzingMonitor(top_k=10, bottom_k=5) if enable_monitoring else None
        
        self.selection_history = []
        
        logging.info("🎲 RandomSearchSelectPolicy initialized (Pure Random Baseline)")
    
    def select(self) -> PromptNode:
 
        if not self.fuzzer.prompt_nodes:
            raise ValueError("No prompt nodes available for selection")
        
        self.step += 1
        
        selected_node = random.choice(self.fuzzer.prompt_nodes)
        selected_node.visited_num += 1
        
        self.selection_history.append({
            'step': self.step,
            'node_index': selected_node.index,
            'node_level': selected_node.level,
            'node_cwe_ids': self._get_node_cwe_ids(selected_node),
            'visited_num': selected_node.visited_num
        })
        
        if self.step % 100 == 0:
            self._log_statistics()
        
        return selected_node
    
    def update(self, prompt_nodes: 'list[PromptNode]'):
        self._update_cwe_stats(prompt_nodes)
        
        if self.monitor:
            self.monitor.log_step(self.step, self, self.fuzzer.prompt_nodes)
    
    def _update_cwe_stats(self, prompt_nodes: List[PromptNode]):
        for pn in prompt_nodes:
            node_cwe_ids = self._get_node_cwe_ids(pn)
            
            for cwe in node_cwe_ids:
                self.cwe_attempt_count[cwe] += 1
                
                if hasattr(pn, 'all_found_cwes') and cwe in pn.all_found_cwes:
                    self.cwe_success_count[cwe] += 1
                    self.cwe_last_success[cwe] = self.step
    
    @staticmethod
    def _get_node_cwe_ids(node: PromptNode) -> List[str]:
        if hasattr(node, 'cwe_ids') and node.cwe_ids:
            return node.cwe_ids if isinstance(node.cwe_ids, list) else [node.cwe_ids]
        elif hasattr(node, 'cwe_id') and node.cwe_id:
            return [node.cwe_id]
        return []
    
    def _log_statistics(self):
        total_cwes = len(self.cwe_success_count)
        total_nodes = len(self.fuzzer.prompt_nodes)
        
        visit_counts = [pn.visited_num for pn in self.fuzzer.prompt_nodes]
        avg_visits = np.mean(visit_counts) if visit_counts else 0
        std_visits = np.std(visit_counts) if visit_counts else 0
        
        logging.info(
            f"\n🎲 [Random Search Baseline] Step {self.step}:\n"
            f"   - Total nodes: {total_nodes}\n"
            f"   - CWE types discovered: {total_cwes}\n"
            f"   - Avg visits per node: {avg_visits:.2f} (std: {std_visits:.2f})\n"
            f"   - CWE distribution: {dict(sorted(self.cwe_success_count.items(), key=lambda x: x[1], reverse=True)[:5])}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:

        if self.monitor:
            return self.monitor.get_summary_statistics()
        
        visit_counts = [pn.visited_num for pn in self.fuzzer.prompt_nodes]
        
        return {
            'policy_type': 'RandomSearch',
            'total_steps': self.step,
            'total_cwe_types_discovered': len(self.cwe_success_count),
            'cwe_distribution': dict(self.cwe_success_count),
            'cwe_attempt_count': dict(self.cwe_attempt_count),
            'gini_coefficient': self._calculate_gini_coefficient(),
            'visit_statistics': {
                'mean': float(np.mean(visit_counts)) if visit_counts else 0,
                'std': float(np.std(visit_counts)) if visit_counts else 0,
                'min': int(np.min(visit_counts)) if visit_counts else 0,
                'max': int(np.max(visit_counts)) if visit_counts else 0
            },
            'coverage': {
                'nodes_visited_at_least_once': sum(1 for v in visit_counts if v > 0),
                'total_nodes': len(self.fuzzer.prompt_nodes)
            }
        }
    
    def _calculate_gini_coefficient(self) -> float:
        return self.calculate_gini_coefficient(self.cwe_success_count)
    
    def export_selection_history(self, filepath: str):
        if not self.selection_history:
            logging.warning("No selection history to export")
            return
        
        df = pd.DataFrame(self.selection_history)
        df.to_csv(filepath, index=False)
        logging.info(f"✓ Selection history exported to: {filepath}")
