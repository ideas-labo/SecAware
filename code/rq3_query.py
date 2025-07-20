import pandas as pd
import requests
import json
import time
import os
import argparse
import re
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Any
import random

# Import LLM module, assuming DeepSeekLLM is in the llm package
sys.path.append(str(Path(__file__).parent.parent))
from llm.llm import DeepSeekLLM

def evaluate_tasks_complexity_batch(tasks_data: List[Dict], api_key: str, 
                                   model_path: str = "deepseek-coder"):
    """
    批量评估多个编程任务的复杂度
    
    Args:
        tasks_data: 包含任务信息的字典列表
        api_key: DeepSeek API密钥
        model_path: 模型路径
    
    Returns:
        批量评估结果
    """
    
    task_descriptions = []
    for i, task in enumerate(tasks_data):
        
        prompt_preview = task.get('prompt', '')
            
        task_descriptions.append(f"""
**Task {i+1}:**
CWE ID: {task.get('cwe_id', 'Unknown')}
Task Description: {prompt_preview}
""")
    
    tasks_text = "\n".join(task_descriptions)
    
    prompt = f"""
You are an expert security researcher and programmer. Evaluate the complexity of the following {len(tasks_data)} programming tasks. 

**CRITICAL: Evaluate all tasks relative to each other to ensure consistent scoring across the batch. A task should only get a higher score than another if it is genuinely more complex in that dimension.**

**EVALUATION DIMENSIONS (1-10 scale):**

1. **Conceptual Complexity**: Programming concepts, algorithms, data structures difficulty
   - 1-2: Basic syntax, simple variables, linear logic
   - 3-4: Control structures, basic data structures (arrays, objects)
   - 5-6: OOP concepts, basic algorithms, file I/O
   - 7-8: Advanced data structures, complex algorithms, design patterns
   - 9-10: Advanced algorithms, complex system design, mathematical concepts

2. **Implementation Complexity**: Code implementation difficulty and estimated size
   - 1-2: <20 lines, single function, no dependencies
   - 3-4: 20-50 lines, 2-3 functions, basic libraries
   - 5-6: 50-100 lines, multiple modules, moderate integration
   - 7-8: 100-200 lines, complex logic flow, significant libraries
   - 9-10: >200 lines, multi-module architecture, extensive integration

3. **Edge Case & Error Handling**: Special cases and error handling complexity
   - 1-2: No edge cases, basic happy path
   - 3-4: Simple input validation, basic error messages
   - 5-6: Multiple input scenarios, structured error handling
   - 7-8: Complex validation, resource management, recovery mechanisms
   - 9-10: Comprehensive fault tolerance, concurrent error handling

4. **Security Risk Assessment**: Potential security risks and necessary measures
   - 1-2: No security risks, read-only operations
   - 3-4: Minor risks, basic input handling
   - 5-6: Moderate risks (file operations, network I/O)
   - 7-8: High risks (memory management, authentication, database)
   - 9-10: Critical risks (cryptography, privilege escalation, system calls)

5. **Performance & Resource Requirements**: Efficiency and optimization needs
   - 1-2: Minimal resources, simple operations
   - 3-4: Basic loops, small data sets
   - 5-6: Moderate data processing, some optimization needed
   - 7-8: Large data sets, performance-critical operations
   - 9-10: High-performance computing, real-time constraints

**Programming Tasks:**
{tasks_text}

**EVALUATION FORMAT:**
For each task, provide evaluation in this exact format:

=== Task X Evaluation ===
Conceptual Complexity: X/10
Implementation Complexity: X/10
Edge Case Handling: X/10
Security Risk: X/10
Performance Requirements: X/10
Overall: X/10
Justification: [Brief explanation of the overall complexity assessment]

**Important Guidelines:**
- Compare tasks relatively - if Task A requires more complex algorithms than Task B, Task A should score higher in Conceptual Complexity
- Consider actual implementation requirements, not just description length
- Be consistent with your scoring scale across all tasks
- The Overall score should reflect a weighted average of the individual dimensions with emphasis on the most critical aspects
"""

    llm = DeepSeekLLM(
        model_path=model_path,
        api_key=api_key,
        system_message="You are a senior security researcher and software architect. Provide consistent, objective evaluations with relative comparison across tasks in the same batch. Focus on implementation complexity rather than description complexity."
    )
    
    responses = llm.generate(
        prompt=prompt,
        temperature=0.1,
        max_tokens=10240,  
        n=1
    )
    
    return responses[0]

def extract_batch_complexity_scores(evaluation_text: str, num_tasks: int) -> List[Dict]:
    """
    从批量评估文本中提取每个任务的复杂度分数
    
    Args:
        evaluation_text: LLM批量评估文本
        num_tasks: 任务数量
        
    Returns:
        每个任务的分数字典列表
    """
    results = []
    
    task_pattern = r"=== Task (\d+) Evaluation ===(.*?)(?==== Task \d+ Evaluation ===|$)"
    task_matches = re.findall(task_pattern, evaluation_text, re.DOTALL)
    
    aspects = [
        "Conceptual Complexity",
        "Implementation Complexity", 
        "Edge Case Handling",
        "Security Risk",
        "Performance Requirements"
    ]
    
    for task_num, task_evaluation in task_matches:
        scores = {"Task_Number": int(task_num)}
        
        for aspect in aspects:
            patterns = [
                rf"{aspect}:\s*(\d+(?:\.\d+)?)/10",
                rf"{aspect}:\s*(\d+(?:\.\d+)?)"
            ]
            
            found = False
            for pattern in patterns:
                match = re.search(pattern, task_evaluation, re.IGNORECASE)
                if match:
                    scores[aspect] = float(match.group(1))
                    found = True
                    break
            
            if not found:
                scores[aspect] = 0.0

        overall_patterns = [
            r"Overall:\s*(\d+(?:\.\d+)?)/10",
            r"Overall:\s*(\d+(?:\.\d+)?)"
        ]
        
        found_overall = False
        for pattern in overall_patterns:
            match = re.search(pattern, task_evaluation, re.IGNORECASE)
            if match:
                scores["Overall"] = float(match.group(1))
                found_overall = True
                break
        
        if not found_overall:
            scores["Overall"] = 0.0
            
        justification_pattern = r"Justification:\s*(.*?)(?:\n\n|$)"
        match = re.search(justification_pattern, task_evaluation, re.DOTALL)
        justification = match.group(1).strip() if match else "No justification provided"
        scores["Justification"] = justification
        
        results.append(scores)
    
    results.sort(key=lambda x: x.get("Task_Number", 0))
    
    while len(results) < num_tasks:
        empty_scores = {
            "Task_Number": len(results) + 1,
            "Conceptual Complexity": 0.0,
            "Implementation Complexity": 0.0,
            "Edge Case Handling": 0.0,
            "Security Risk": 0.0,
            "Performance Requirements": 0.0,
            "Overall": 0.0,
            "Justification": "Evaluation parsing failed"
        }
        results.append(empty_scores)
    
    return results[:num_tasks]

def evaluate_tasks_with_multiple_batches(df: pd.DataFrame, api_key: str, 
                                       model_path: str = "deepseek-coder", 
                                       batch_size: int = 5, 
                                       num_rounds: int = 1) -> pd.DataFrame:
    """
    使用多批次评估提高评估效率和质量
    
    Args:
        df: 包含任务的DataFrame
        api_key: API密钥  
        model_path: 模型路径
        batch_size: 批量大小
        num_rounds: 每个批次的评估轮数
        
    Returns:
        包含评估结果的DataFrame
    """
    
    # 准备任务数据
    tasks_data = []
    for index, row in df.iterrows():
        tasks_data.append({
            'task_id': row['index'] if 'index' in row else index,
            'cwe_id': row.get('cwe_id', ''),
            'prompt': row['prompt'],
            'results': row.get('results'),
            'found_cwes': row.get('found_cwes'),
            'original_index': index
        })
    
    all_results = []
    
    print(f"开始批量评估 {len(tasks_data)} 个任务 (批量大小: {batch_size})")
    
    # 分批处理
    for batch_start in range(0, len(tasks_data), batch_size):
        batch_end = min(batch_start + batch_size, len(tasks_data))
        batch = tasks_data[batch_start:batch_end]
        
        print(f"处理批次 {batch_start//batch_size + 1} (任务 {batch_start+1}-{batch_end})")
        
        batch_results = []
        
        # 多轮评估（如果需要）
        for round_num in range(num_rounds):
            if num_rounds > 1:
                print(f"  第 {round_num + 1} 轮评估...")
                # 随机打乱批次内任务顺序以减少位置偏差
                batch_shuffled = batch.copy()
                random.shuffle(batch_shuffled)
            else:
                batch_shuffled = batch
            
            try:
                evaluation_text = evaluate_tasks_complexity_batch(
                    batch_shuffled, api_key, model_path
                )
                
                scores_list = extract_batch_complexity_scores(evaluation_text, len(batch_shuffled))
                
                # 将结果映射回原始任务
                round_results = []
                for i, task in enumerate(batch_shuffled):
                    if i < len(scores_list):
                        result = scores_list[i].copy()
                        result['task_id'] = task['task_id']
                        result['original_index'] = task['original_index']
                        result['cwe_id'] = task['cwe_id']
                        result['results'] = task['results']
                        result['found_cwes'] = task['found_cwes']
                        result['round'] = round_num + 1
                        round_results.append(result)
                
                batch_results.extend(round_results)
                
            except Exception as e:
                print(f"  批次评估失败: {e}")
                # 为失败的批次添加默认结果
                for task in batch_shuffled:
                    error_result = {
                        'task_id': task['task_id'],
                        'original_index': task['original_index'],
                        'cwe_id': task['cwe_id'],
                        'results': task['results'],
                        'found_cwes': task['found_cwes'],
                        'Conceptual Complexity': 0.0,
                        'Implementation Complexity': 0.0,
                        'Edge Case Handling': 0.0,
                        'Security Risk': 0.0,
                        'Performance Requirements': 0.0,
                        'Overall': 0.0,
                        'Justification': f"Batch evaluation failed: {str(e)}",
                        'round': round_num + 1
                    }
                    batch_results.append(error_result)
            
            # 延迟避免API限制
            time.sleep(2 if num_rounds > 1 else 1)
        
        all_results.extend(batch_results)
        
        # 批次间延迟
        if batch_end < len(tasks_data):
            time.sleep(1)
    
    # 如果有多轮评估，计算平均分数
    if num_rounds > 1:
        final_results = calculate_average_scores(all_results, tasks_data)
    else:
        final_results = all_results
    
    # 创建结果DataFrame
    result_df = df.copy()
    
    # 添加复杂度评估结果
    for result in final_results:
        mask = result_df.index == result['original_index']
        
        # 映射字段名
        field_mapping = {
            'Conceptual Complexity': 'complexity_conceptual',
            'Implementation Complexity': 'complexity_implementation', 
            'Edge Case Handling': 'complexity_edge_case',
            'Security Risk': 'complexity_security',
            'Performance Requirements': 'complexity_performance',
            'Overall': 'complexity_overall'
        }
        
        for original_field, new_field in field_mapping.items():
            if original_field in result:
                result_df.loc[mask, new_field] = result[original_field]
        
        if 'Justification' in result:
            result_df.loc[mask, 'justification'] = result['Justification']
        
        # 如果有多轮评估，也保存标准差信息
        if f'{original_field}_std' in result:
            result_df.loc[mask, f'{new_field}_std'] = result[f'{original_field}_std']
    
    return result_df

def calculate_average_scores(all_results: List[Dict], tasks_data: List[Dict]) -> List[Dict]:
    """计算多轮评估的平均分数"""
    
    final_results = []
    score_fields = ['Conceptual Complexity', 'Implementation Complexity', 
                   'Edge Case Handling', 'Security Risk', 
                   'Performance Requirements', 'Overall']
    
    for task in tasks_data:
        task_id = task['task_id']
        task_results = [r for r in all_results if r['task_id'] == task_id]
        
        if not task_results:
            continue
            
        avg_result = {
            'task_id': task_id,
            'original_index': task['original_index'],
            'cwe_id': task['cwe_id'],
            'results': task['results'],
            'found_cwes': task['found_cwes']
        }
        
        # 计算每个维度的平均分和标准差
        justifications = []
        for field in score_fields:
            scores = [r[field] for r in task_results if field in r and r[field] > 0]
            if scores:
                avg_result[field] = np.mean(scores)
                avg_result[f'{field}_std'] = np.std(scores) if len(scores) > 1 else 0.0
                avg_result[f'{field}_count'] = len(scores)
            else:
                avg_result[field] = 0.0
                avg_result[f'{field}_std'] = 0.0
                avg_result[f'{field}_count'] = 0
        
        # 合并所有轮次的理由
        for r in task_results:
            if 'Justification' in r and r['Justification']:
                justifications.append(f"Round {r.get('round', '?')}: {r['Justification']}")
        
        avg_result['Justification'] = " | ".join(justifications) if justifications else "No justification"
        final_results.append(avg_result)
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description='Batch evaluate task complexity using DeepSeek API')
    parser.add_argument('--api-key', type=str, help='DeepSeek API key')
    parser.add_argument('--model-path', type=str, default='deepseek-coder', 
                        help='Model path (deepseek-chat or deepseek-coder)')
    parser.add_argument('--input-csv', type=str, help='Input CSV file path')
    parser.add_argument('--output-csv', type=str, help='Output CSV file path')
    parser.add_argument('--batch-size', type=int, default=5, 
                        help='Number of tasks to evaluate in each batch')
    parser.add_argument('--num-rounds', type=int, default=1,
                        help='Number of evaluation rounds for each batch')
    parser.add_argument('--enable-analysis', action='store_true',
                        help='Enable correlation analysis after evaluation')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("DeepSeek API key is required")
    
    input_csv = args.input_csv 
    output_csv = args.output_csv or 'batch_complexity_evaluation.csv'
    
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} tasks from {input_csv}")
    
    result_df = evaluate_tasks_with_multiple_batches(
        df, api_key, args.model_path, 
        args.batch_size, args.num_rounds
    )
    
    result_df.to_csv(output_csv, index=False)
    print(f"Batch evaluation completed. Results saved to {output_csv}")
    
    # 打印统计信息
    print(f"\n===== Evaluation Summary =====")
    if 'complexity_overall' in result_df.columns:
        valid_overall = result_df['complexity_overall'].dropna()
        if len(valid_overall) > 0:
            print(f"Overall complexity: Mean {valid_overall.mean():.2f}, Std {valid_overall.std():.2f}")
            print(f"Complexity range: {valid_overall.min():.1f} - {valid_overall.max():.1f}")
            print(f"Valid evaluations: {len(valid_overall)}/{len(result_df)}")

if __name__ == "__main__":
    main()